import hashlib
import json
import threading
import time
from functools import lru_cache
from typing import Any, Iterator, Optional, Sequence

import redis
import rag.config as config
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.stores import BaseStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from store_logs.db import collection

_state = {"ok": False, "checked": 0.0}
_state_lock = threading.Lock()
_RECHECK_SECONDS = 30


def get_redis_client():
    return redis.Redis.from_url(
        config.REDIS_URL,
        socket_connect_timeout=0.25,
        socket_timeout=0.6,
    )


@lru_cache(maxsize=1)
def _client():
    return get_redis_client()


def redis_available() -> bool:
    now = time.time()
    if now - _state["checked"] < _RECHECK_SECONDS:
        return _state["ok"]
    with _state_lock:
        if now - _state["checked"] < _RECHECK_SECONDS:
            return _state["ok"]
        try:
            _client().ping()
            _state["ok"] = True
        except Exception:
            _state["ok"] = False
        _state["checked"] = time.time()
    return _state["ok"]


def _redis_failed():
    _state["ok"] = False
    _state["checked"] = time.time()



_mem: dict[str, tuple[float, str]] = {}
_MEM_MAX = 5000
_mem_lock = threading.Lock()
_NS = "law:"


def make_key(*parts: Any) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def kv_get(key: str) -> Optional[Any]:
    k = _NS + key
    now = time.time()
    with _mem_lock:
        hit = _mem.get(k)
        if hit and hit[0] > now:
            return json.loads(hit[1])
        if hit:
            _mem.pop(k, None)
    if redis_available():
        try:
            raw = _client().get(k)
            if raw is not None:
                s = raw.decode("utf-8")
                with _mem_lock:
                    _mem[k] = (now + 600, s)  
                return json.loads(s)
        except Exception:
            _redis_failed()
    return None


def kv_set(key: str, value: Any, ttl: int) -> None:
    k = _NS + key
    s = json.dumps(value, ensure_ascii=False)
    with _mem_lock:
        if len(_mem) >= _MEM_MAX:

            for old in sorted(_mem, key=lambda x: _mem[x][0])[: _MEM_MAX // 10]:
                _mem.pop(old, None)
        _mem[k] = (time.time() + ttl, s)
    if redis_available():
        try:
            _client().setex(k, ttl, s)
        except Exception:
            _redis_failed()


def setup_llm_cache():
    if redis_available():
        try:
            from langchain_redis import RedisCache
            set_llm_cache(RedisCache(redis_url=config.REDIS_URL, ttl=config.REDIS_CACHE_TTL_SECONDS))
            collection.insert_one({"redis_enable": "true", "redis_url": config.REDIS_URL})
            return "redis"
        except Exception:
            pass
    set_llm_cache(InMemoryCache())
    collection.insert_one({"redis_enable": "false", "redis_url": config.REDIS_URL})
    return "memory"


class TieredByteStore(BaseStore[str, bytes]):
    """memory first, Redis second; never raises when Redis disappears."""

    def __init__(self, namespace: str = "embed_cache"):
        self.ns = namespace + ":"
        self._d: dict[str, bytes] = {}

    def mget(self, keys: Sequence[str]) -> list[Optional[bytes]]:
        out: list[Optional[bytes]] = [self._d.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing and redis_available():
            try:
                vals = _client().mget([self.ns + keys[i] for i in missing])
                for i, v in zip(missing, vals):
                    if v is not None:
                        self._d[keys[i]] = v
                        out[i] = v
            except Exception:
                _redis_failed()
        return out

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        for k, v in key_value_pairs:
            self._d[k] = v
        if redis_available():
            try:
                pipe = _client().pipeline()
                for k, v in key_value_pairs:
                    pipe.set(self.ns + k, v)
                pipe.execute()
            except Exception:
                _redis_failed()

    def mdelete(self, keys: Sequence[str]) -> None:
        for k in keys:
            self._d.pop(k, None)
        if redis_available():
            try:
                _client().delete(*[self.ns + k for k in keys])
            except Exception:
                _redis_failed()

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        for k in list(self._d):
            if prefix is None or k.startswith(prefix):
                yield k


@lru_cache(maxsize=1)
def get_cached_embeddings():
    underlying = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
    store = TieredByteStore("embed_cache")
    cached = CacheBackedEmbeddings.from_bytes_store(
        underlying,
        store,
        namespace=config.EMBEDDING_MODEL,
        query_embedding_cache=store,
    )
    collection.insert_one({"embedding_cache": "tiered(memory+redis)"})
    return cached
