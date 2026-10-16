from rag.memory_chain import build_converstional_rag_chain
from rag.load_chunk import build_index
from rag.cache_file import set_llm_cache

import rag.config as config
from observability.monitor import setup_observability
import time
import uuid


build_index(force_rebuild=False)

session_id = str(uuid.uuid4())

chain = build_converstional_rag_chain()
setup_observability()
start = time.perf_counter()
result = chain.invoke(
    {"input" : "what is a security deposit"},
    config = {"configurable" : {"session_id" : session_id}}
)
end = time.perf_counter()
print(result['answer'])
print(end-start)