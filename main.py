from rag.memory_chain import build_converstional_rag_chain
from rag.load_chunk import build_index
from rag.cache_file import set_llm_cache

import rag.config as config
import time
import uuid


build_index(force_rebuild=False)

session_id = str(uuid.uuid4())

chain = build_converstional_rag_chain()

result = chain.invoke(
    {"input" : "hi"},
    config = {"configurable" : {"session_id" : session_id}}
)

print(result['answer'])