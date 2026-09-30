import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-40-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_TEMPERATURE = 0.0

PDF_PATH = os.getenv("PDF_PATH","documents/law_documents.pdf")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(BASE_DIR,"store")
FAISS_DIR = os.path.join(STORE_DIR,"faiss_index")
BM25_DOCS_PATH = os.path.join(STORE_DIR,"bm25_docs.pkl")
EMBED_CACHE_DIR = os.path.join(STORE_DIR,"embed_cache")

TOP_K_SEMANTIC = 5
TOP_K_BM25 = 5
ENSEMBLE_WEIGHTS = [0.5,0.5]
TOP_K_AFTER_RERANK = 4
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MULTI_QUERY_COUNT = 3


REDIS_URL = os.getenv(REDIS_URL,"redis://localhost:6379/0")
REDIS_CACHE_TTL_SECONDS = 60 * 60 * 24

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set sorry for interruption we will solve very soon"
    )