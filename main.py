import asyncio
import sys
import time
import os
from observability.monitor import setup_observability

setup_observability()


async def _ask(question: str):
    from rag.cache_file import setup_llm_cache
    from rag.load_chunk import build_index
    from rag.memory_chain import build_converstional_rag_chain

    setup_llm_cache()
    build_index(force_rebuild=False)
    engine = build_converstional_rag_chain()
    start = time.perf_counter()
    async for ev in engine.answer_stream(question, []):
        if ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "replace":
            print("\n[output guardrail replaced answer]\n", ev["text"])
    print(f"\n\n{time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(_ask(" ".join(sys.argv[1:])))
    else:
        import uvicorn

        uvicorn.run(
            "api.server:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "10000")),
            reload=False
        )