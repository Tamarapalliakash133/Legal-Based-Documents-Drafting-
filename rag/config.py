import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_TEMPERATURE = 0.0

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.getenv("PDF_PATH", os.path.join(_ROOT, "documents", "law_documents.pdf"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


RERANK_CANDIDATES = 8          
RRF_K = 60
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "true").lower() == "true"
HYDE_MAX_TOKENS = 110   
HYDE_TIMEOUT_SECONDS = 1.6  
MAX_ANSWER_TOKENS = 700
DRAFT_MAX_TOKENS = 2200
DRAFT_MODEL = os.getenv("DRAFT_MODEL", LLM_MODEL)
HISTORY_TURNS = 6           
ANSWER_CACHE_TTL_SECONDS = 60 * 60 * 24 * 3
HYDE_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7
RETRIEVAL_CACHE_TTL_SECONDS = 60 * 60 * 6
PROMPT_VERSION = "v2"      

MAX_CHAT_INPUT_CHARS = 4000
MAX_DRAFT_INPUT_CHARS = 12000


APP_DB_PATH = os.getenv("APP_DB_PATH", os.path.join(STORE_DIR, "app.db"))

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "true"))
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "akash-project")


REDIS_URL = os.getenv("REDIS_URL","redis://localhost:6379/0")
REDIS_CACHE_TTL_SECONDS = 60 * 60 * 24

MONGODB_URL = os.getenv("MONGODB_URL","mongodb://localhost:27017/")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set sorry for interruption we will solve very soon"
    )