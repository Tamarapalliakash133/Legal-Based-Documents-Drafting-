import queue
import threading
import time
from datetime import datetime, timezone

import rag.config as config

_DOWN_FOR_SECONDS = 60


class _SafeCollection:
    def __init__(self):
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
        self._coll = None
        self._down_until = 0.0
        self._started = False
        self._lock = threading.Lock()

    def _ensure_worker(self):
        if self._started:
            return
        with self._lock:
            if not self._started:
                threading.Thread(target=self._worker, name="mongo-logger", daemon=True).start()
                self._started = True

    def _connect(self):
        from pymongo import MongoClient
        client = MongoClient(
            config.MONGODB_URL,
            serverSelectionTimeoutMS=800,
            connectTimeoutMS=800,
            socketTimeoutMS=2000,
        )
        return client["store_logs"]["db_collection"]

    def _worker(self):
        while True:
            doc = self._q.get()
            if time.time() < self._down_until:
                continue  
            try:
                if self._coll is None:
                    self._coll = self._connect()
                self._coll.insert_one(doc)
            except Exception:
                self._coll = None
                self._down_until = time.time() + _DOWN_FOR_SECONDS

    def insert_one(self, doc: dict):
        doc = dict(doc)
        doc.setdefault("ts", datetime.now(timezone.utc))
        self._ensure_worker()
        try:
            self._q.put_nowait(doc)
        except queue.Full:
            pass


collection = _SafeCollection()


def log_event(event: str, **fields):
    collection.insert_one({"event": event, **fields})
