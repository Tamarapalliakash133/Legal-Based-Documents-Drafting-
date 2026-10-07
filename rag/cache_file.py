import redis
import rag.config as config
from langchain_redis import RedisCache
from langchain_community.storage import RedisStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.globals import set_llm_cache
from store_logs.db import collection

def get_redis_client():
    return redis.Redis.from_url(config.REDIS_URL)

def setup_llm_cache():
    try:
        client = get_redis_client()
        client.ping()

        cache = RedisCache(
            redis_url = config.REDIS_URL,
            ttl = config.REDIS_CACHE_TTL_SECONDS
        )
        set_llm_cache(cache)

        collection.insert_one({
            "redis_enable" : "true",
            "redis_url" : config.REDIS_URL
        })
    except redis.exceptions.ConnectionError:
        collection.insert_one({
            "redis_enable" : "false",
            "redis_url" : config.REDIS_URL
        })

def get_cached_embeddings():

    underlying = OpenAIEmbeddings(
        model = config.EMBEDDING_MODEL,
        api_key = config.OPENAI_API_KEY
    )

    try:
        client = get_redis_client()
        client.ping()

        store = RedisStore(
            client = client,
            namespace = "embed_cache"
        )

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying,
            store,
            namespace = config.EMBEDDING_MODEL
        )

        collection.insert_one({
            "embedding_cache" : "true"
        })
        return cached_embeddings

    except redis.exceptions.ConnectionError:
        collection.insert_one({
            "embedding_cache" : "false"
        })
        return underlying


