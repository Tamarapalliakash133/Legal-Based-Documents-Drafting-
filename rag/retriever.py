import concurrent.futures as cf
import contextvars
import pickle
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable

import rag.config as config
from guardrails.input_rules import sanitize_context
from rag.cache_file import get_cached_embeddings, kv_get, kv_set, make_key
from store_logs.db import collection

_POOL = cf.ThreadPoolExecutor(max_workers=10, thread_name_prefix="retrieve")

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You help a legal search engine. Write ONE short passage (3-4 sentences, max 70 words) "
     "in the style of a legal reference document that would directly answer the question. "
     "Use precise legal terminology and the key terms a legal text would contain. "
     "If a conversation excerpt is given, resolve pronouns/follow-ups using it. "
     "Output only the passage."),
    ("human", "Conversation excerpt (may be empty):\n{history}\n\nQuestion: {question}"),
])

_STOP = set("a an the of to in on for and or is are was were be been it this that with as by at from what which who "
            "how do does did can could should would may shall i you we they my your our their me not no if any".split())
_tok = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t for t in _tok.findall(text.lower()) if t not in _STOP]


def _submit(fn, *a, **kw):
   
    ctx = contextvars.copy_context()
    return _POOL.submit(ctx.run, lambda: fn(*a, **kw))


def _doc_key(d: Document) -> str:
    return make_key(d.metadata.get("page"), d.page_content[:200])


def _to_json(docs: List[Document]):
    return [{"c": d.page_content, "m": d.metadata} for d in docs]


def _from_json(items) -> List[Document]:
    return [Document(page_content=i["c"], metadata=i["m"]) for i in items]


@dataclass
class RetrievalResult:
    docs: List[Document]
    hyde_used: bool = False
    hyde_passage: Optional[str] = None
    cached: bool = False
    timings: dict = field(default_factory=dict)


class HybridHydeRetriever:
    def __init__(self):
        t = time.perf_counter()
        self.embeddings = get_cached_embeddings()
        self.vector_store = FAISS.load_local(
            config.FAISS_DIR, self.embeddings, allow_dangerous_deserialization=True
        )
        with open(config.BM25_DOCS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        self.bm25 = BM25Retriever.from_documents(self.chunks, preprocess_func=tokenize)
        self.bm25.k = config.TOP_K_BM25

        self.hyde_llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0,
            max_tokens=config.HYDE_MAX_TOKENS,
            timeout=config.HYDE_TIMEOUT_SECONDS + 2,
            max_retries=0,
        )
        self._hyde_chain = HYDE_PROMPT | self.hyde_llm
        self._reranker = None
        self._reranker_lock = threading.Lock()
        self._reranker_failed = False

        collection.insert_one({
            "is_ready_hydrid_reterival": "yes",
            "hyde": config.HYDE_ENABLED,
            "top_k_semantic": config.TOP_K_SEMANTIC,
            "top_k_bm25": config.TOP_K_BM25,
            "ensemble_weights": config.ENSEMBLE_WEIGHTS,
            "load_seconds": round(time.perf_counter() - t, 3),
        })

    def warmup(self):
        """Load the cross-encoder + hit the embedding path once so the first user is not slow."""
        self._get_reranker()
        try:
            self.embeddings.embed_query("warmup legal agreement")
        except Exception:
            pass

    def _get_reranker(self):
        if self._reranker is not None or self._reranker_failed:
            return self._reranker
        with self._reranker_lock:
            if self._reranker is None and not self._reranker_failed:
                try:
                    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
                    self._reranker = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
                    self._reranker.score([("warm", "up")])
                    collection.insert_one({"reranking": "cross_encoder attached", "top_k": config.TOP_K_AFTER_RERANK})
                except Exception as e:  # missing torch / no network -> fuse-only ranking
                    self._reranker_failed = True
                    collection.insert_one({"reranking": "unavailable", "error": str(e)[:200]})
        return self._reranker


    @traceable(run_type="retriever", name="bm25_search")
    def _bm25(self, query: str) -> List[Document]:
        return self.bm25.invoke(query)

    @traceable(run_type="retriever", name="dense_search",
               process_inputs=lambda inputs: {"k": inputs.get("k"), "vector": "<embedding omitted>"})
    def _dense(self, vec: List[float], k: int) -> List[Document]:
        return self.vector_store.similarity_search_by_vector(vec, k=k)

    @traceable(run_type="retriever", name="dense_search_from_text")
    def _dense_text(self, text: str) -> List[Document]:
        return self._dense(self.embeddings.embed_query(text), config.TOP_K_SEMANTIC)

    @traceable(run_type="chain", name="hyde_generate")
    def _hyde_generate(self, question: str, history: str) -> str:
        key = "hyde:" + make_key(question.lower().strip(), history[-400:], config.LLM_MODEL, config.HYDE_MAX_TOKENS)
        hit = kv_get(key)
        if hit:
            return hit
        passage = self._hyde_chain.invoke({"question": question, "history": history or "(none)"}).content.strip()
        if passage:
            kv_set(key, passage, config.HYDE_CACHE_TTL_SECONDS)
        return passage

    @traceable(run_type="retriever", name="hyde_branch")
    def _hyde_branch(self, question: str, history: str):
        passage = self._hyde_generate(question, history)
        if not passage:
            return passage, [], []
        dense = self._dense_text(passage)
        lex = self._bm25(passage)
        return passage, dense, lex

    @traceable(run_type="chain", name="rrf_fusion")
    def _fuse(self, ranked_lists: List[List[Document]], weights: List[float]) -> List[Document]:
        scores: dict[str, float] = {}
        keep: dict[str, Document] = {}
        for docs, w in zip(ranked_lists, weights):
            for rank, d in enumerate(docs):
                k = _doc_key(d)
                keep.setdefault(k, d)
                scores[k] = scores.get(k, 0.0) + w / (config.RRF_K + rank + 1)
        order = sorted(scores, key=scores.get, reverse=True)
        return [keep[k] for k in order]

    @traceable(run_type="chain", name="cross_encoder_rerank")
    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        rr = self._get_reranker()
        cands = docs[: config.RERANK_CANDIDATES]
        if rr is None or len(cands) <= 1:
            return cands[: config.TOP_K_AFTER_RERANK]
        scores = rr.score([(query, d.page_content) for d in cands])
        ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[: config.TOP_K_AFTER_RERANK]]


    @traceable(run_type="retriever", name="retrieve")
    def retrieve(self, question: str, history: str = "", use_hyde: Optional[bool] = None,
                 rerank: bool = True) -> RetrievalResult:
        t0 = time.perf_counter()
        use_hyde = config.HYDE_ENABLED if use_hyde is None else use_hyde

        ckey = "ret:" + make_key(question.lower().strip(), history[-400:], use_hyde, rerank)
        hit = kv_get(ckey)
        if hit:
            return RetrievalResult(_from_json(hit["docs"]), hit["hyde"], hit.get("passage"), True,
                                   {"retrieval_ms": round((time.perf_counter() - t0) * 1000)})


        f_bm25 = _submit(self._bm25, question)
        f_dense = _submit(self._dense_text, question)
        f_hyde = _submit(self._hyde_branch, question, history) if use_hyde else None

        bm25_docs = f_bm25.result()
        dense_docs = f_dense.result()
        t_plain = time.perf_counter()

        lists = [dense_docs, bm25_docs]
        weights = [config.ENSEMBLE_WEIGHTS[0], config.ENSEMBLE_WEIGHTS[1]]
        passage, hyde_ok = None, False
        if f_hyde is not None:
            try:
                remaining = max(0.05, config.HYDE_TIMEOUT_SECONDS - (time.perf_counter() - t0))
                passage, h_dense, h_bm25 = f_hyde.result(timeout=remaining)
                if h_dense:
                    lists += [h_dense, h_bm25]
                    weights += [config.ENSEMBLE_WEIGHTS[0], config.ENSEMBLE_WEIGHTS[1] * 0.6]
                    hyde_ok = True
            except cf.TimeoutError:
                collection.insert_one({"hyde": "timeout", "q": question[:120]})
            except Exception as e:
                collection.insert_one({"hyde": "error", "error": str(e)[:200]})
        t_hyde = time.perf_counter()

        fused = self._fuse(lists, weights)
        t_fuse = time.perf_counter()

        rq = f"{question}" if not history else f"{history.splitlines()[-1][:200]} {question}"
        docs = self._rerank(rq, fused) if rerank else fused[: config.TOP_K_AFTER_RERANK]
        t_end = time.perf_counter()

       
        docs = [Document(page_content=sanitize_context(d.page_content), metadata=dict(d.metadata)) for d in docs]

        timings = {
            "plain_search_ms": round((t_plain - t0) * 1000),
            "hyde_wait_ms": round((t_hyde - t_plain) * 1000),
            "rerank_ms": round((t_end - t_fuse) * 1000),
            "retrieval_ms": round((t_end - t0) * 1000),
        }
        kv_set(ckey, {"docs": _to_json(docs), "hyde": hyde_ok, "passage": passage},
               config.RETRIEVAL_CACHE_TTL_SECONDS)
        return RetrievalResult(docs, hyde_ok, passage, False, timings)


_instance: Optional[HybridHydeRetriever] = None
_instance_lock = threading.Lock()


def get_retriever() -> HybridHydeRetriever:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = HybridHydeRetriever()
    return _instance



def build_retriever(llm=None) -> HybridHydeRetriever:
    return get_retriever()
