import json
import os
import sqlite3
import threading
import time
import uuid

import rag.config as config

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS workspaces(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT DEFAULT '', created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS chats(
  id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  title TEXT NOT NULL, created_at REAL, updated_at REAL);
CREATE INDEX IF NOT EXISTS idx_chats_ws ON chats(workspace_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages(
  id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT DEFAULT '[]', meta TEXT DEFAULT '{}', created_at REAL);
CREATE INDEX IF NOT EXISTS idx_msgs_chat ON messages(chat_id, created_at);
CREATE TABLE IF NOT EXISTS drafts(
  id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  title TEXT NOT NULL, doc_type TEXT DEFAULT '', details TEXT DEFAULT '', content TEXT DEFAULT '',
  created_at REAL, updated_at REAL);
CREATE INDEX IF NOT EXISTS idx_drafts_ws ON drafts(workspace_id, updated_at DESC);
"""


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                os.makedirs(os.path.dirname(config.APP_DB_PATH), exist_ok=True)
                c = sqlite3.connect(config.APP_DB_PATH, check_same_thread=False)
                c.row_factory = sqlite3.Row
                c.executescript(SCHEMA)
                _conn = c
    return _conn


def _q(sql, args=(), one=False):
    with _lock:
        cur = _db().execute(sql, args)
        rows = cur.fetchall()
        _db().commit()
    return (rows[0] if rows else None) if one else rows


def _id() -> str:
    return uuid.uuid4().hex[:12]


def _d(row):
    return dict(row) if row else None


# --------------------------------------------------------------------------- workspaces
def list_workspaces():
    return [dict(r) for r in _q(
        """SELECT w.*, (SELECT COUNT(*) FROM chats c WHERE c.workspace_id=w.id) AS chat_count,
                  (SELECT COUNT(*) FROM drafts d WHERE d.workspace_id=w.id) AS draft_count
           FROM workspaces w ORDER BY w.updated_at DESC""")]


def create_workspace(name: str, description: str = ""):
    wid, now = _id(), time.time()
    _q("INSERT INTO workspaces VALUES(?,?,?,?,?)", (wid, name.strip()[:80], description.strip()[:1500], now, now))
    return get_workspace(wid)


def get_workspace(wid: str):
    return _d(_q("SELECT * FROM workspaces WHERE id=?", (wid,), one=True))


def update_workspace(wid: str, name=None, description=None):
    w = get_workspace(wid)
    if not w:
        return None
    _q("UPDATE workspaces SET name=?, description=?, updated_at=? WHERE id=?",
       ((name if name is not None else w["name"]).strip()[:80],
        (description if description is not None else w["description"]).strip()[:1500], time.time(), wid))
    return get_workspace(wid)


def delete_workspace(wid: str):
    _q("DELETE FROM workspaces WHERE id=?", (wid,))

def list_chats(wid: str):
    return [dict(r) for r in _q("SELECT * FROM chats WHERE workspace_id=? ORDER BY updated_at DESC", (wid,))]


def create_chat(wid: str, title: str = "New chat"):
    cid, now = _id(), time.time()
    _q("INSERT INTO chats VALUES(?,?,?,?,?)", (cid, wid, title[:80], now, now))
    return get_chat(cid)


def get_chat(cid: str):
    return _d(_q("SELECT * FROM chats WHERE id=?", (cid,), one=True))


def rename_chat(cid: str, title: str):
    _q("UPDATE chats SET title=? WHERE id=?", (title.strip()[:80] or "Untitled", cid))
    return get_chat(cid)


def delete_chat(cid: str):
    _q("DELETE FROM chats WHERE id=?", (cid,))


def get_messages(cid: str):
    out = []
    for r in _q("SELECT * FROM messages WHERE chat_id=? ORDER BY created_at, rowid", (cid,)):
        m = dict(r)
        m["sources"] = json.loads(m["sources"] or "[]")
        m["meta"] = json.loads(m["meta"] or "{}")
        out.append(m)
    return out


def recent_history(cid: str, n: int):
    rows = _q("SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?", (cid, n))
    return [dict(r) for r in reversed(rows)]


def add_message(cid: str, role: str, content: str, sources=None, meta=None):
    mid, now = _id(), time.time()
    _q("INSERT INTO messages VALUES(?,?,?,?,?,?,?)",
       (mid, cid, role, content, json.dumps(sources or []), json.dumps(meta or {}), now))
    _q("UPDATE chats SET updated_at=? WHERE id=?", (now, cid))
    return mid


def list_drafts(wid: str):
    return [dict(r) for r in _q(
        "SELECT id, workspace_id, title, doc_type, created_at, updated_at FROM drafts WHERE workspace_id=? ORDER BY updated_at DESC", (wid,))]


def create_draft(wid: str, title: str, doc_type: str = "", details: str = "", content: str = ""):
    did, now = _id(), time.time()
    _q("INSERT INTO drafts VALUES(?,?,?,?,?,?,?,?)", (did, wid, title[:120], doc_type, details, content, now, now))
    return get_draft(did)


def get_draft(did: str):
    return _d(_q("SELECT * FROM drafts WHERE id=?", (did,), one=True))


def update_draft(did: str, **fields):
    d = get_draft(did)
    if not d:
        return None
    for k in ("title", "doc_type", "details", "content"):
        if fields.get(k) is not None:
            d[k] = fields[k]
    _q("UPDATE drafts SET title=?, doc_type=?, details=?, content=?, updated_at=? WHERE id=?",
       (d["title"][:120], d["doc_type"], d["details"], d["content"], time.time(), did))
    return get_draft(did)


def delete_draft(did: str):
    _q("DELETE FROM drafts WHERE id=?", (did,))
