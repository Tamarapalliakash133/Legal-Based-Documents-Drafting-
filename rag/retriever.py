import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

import config
from cache_file import get_cached_embeddings
from store_logs import collection


def load_hybrid_retriever():
    embeddings = get_cached_embeddings()

    vector_store = FAISS.load_local(
        config.FAISS_DIR,embeddings
    )
    semantic_retriever = vector_store.as_retriever(search_kwargs = {"k": config.TOP_K_SEMANTIC})

    with open(config.BM25_DOCS_PATH,"rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.TOP_K_BM25

    hybrid = EnsembleRetriever(
        retrievers = [semantic_retriever,bm25_retriever],
        weights = config.ENSEMBLE_WEIGHTS
    )

    collection.insert_one({
        "is_ready_hydrid_reterival" : "yes",
        "top_k_semantic" : config.TOP_K_SEMANTIC,
        "top_k_bm25" : config.TOP_K_BM25,
        "ensemble_weights" : config.ENSEMBLE_WEIGHTS
    })

    return hybrid