import os
from dotenv import load_dotenv
load_dotenv()

import rag.config as config
from store_logs.db import collection


def setup_observability() -> bool:
    """Turn LangSmith tracing on only when a key exists (otherwise the tracer just spams errors
    and wastes a thread). Sets both the LANGCHAIN_* and the newer LANGSMITH_* variables.
    @traceable spans are used on guardrails, retrieval (bm25 / dense / hyde / rerank), generation
    and the root pipeline; LangChain LLM calls nest under them automatically."""
    want = config.LANGCHAIN_TRACING_V2.lower() == "true"
    enabled = want and bool(config.LANGCHAIN_API_KEY)

    flag = "true" if enabled else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = flag
    os.environ["LANGSMITH_TRACING"] = flag

    if enabled:
        for prefix in ("LANGCHAIN", "LANGSMITH"):
            os.environ[f"{prefix}_API_KEY"] = config.LANGCHAIN_API_KEY
            os.environ[f"{prefix}_PROJECT"] = config.LANGCHAIN_PROJECT
        os.environ.setdefault("LANGSMITH_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))
        collection.insert_one({"langsmith": "activated", "project": config.LANGCHAIN_PROJECT})
    else:
        collection.insert_one({"langsmith": "not activated", "reason": "no key" if want else "disabled"})
    return enabled
