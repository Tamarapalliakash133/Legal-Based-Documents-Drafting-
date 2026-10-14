import os
from dotenv import load_dotenv
load_dotenv()

import rag.config as config
from store_logs.db import collection
def setup_observability():
    os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2

    if config.LANGCHAIN_TRACING_V2.lower() == "true":
        os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
        
        collection.insert_one({
            "langsmith" : "activated"
        })
    else:
        collection.insert_one({
            "langsmith" : "not actiavted"
        })