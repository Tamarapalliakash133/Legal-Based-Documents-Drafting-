import asyncio
import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag.config as config
from observability.monitor import setup_observability

setup_observability() 

from drafting.draft import DOC_TYPES, build_pdf, get_draft_engine
from guardrails.input_rules import check_input
from ocr.text_extract import extract_text, ocr_available, warm_ocr
from rag.cache_file import redis_available, setup_llm_cache
from rag.load_chunk import build_index
from rag.memory_chain import get_engine
from store_logs import app_store as store
from store_logs.db import log_event

FRONTEND_DIST = os.path.join(config.BASE_DIR, "frontend", "dist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_llm_cache()
    build_index(force_rebuild=False)
    engine = get_engine()
    get_draft_engine()
    threading.Thread(target=engine.warmup, daemon=True).start()
    if os.getenv("OCR_WARMUP", "true").lower() == "true":
        threading.Thread(target=warm_ocr, daemon=True).start()
    yield


app = FastAPI(title="Law Research & Drafting", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)



class WorkspaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=1500)


class WorkspacePatch(BaseModel):
    name: Optional[str] = Field(default=None, max_length=80)
    description: Optional[str] = Field(default=None, max_length=1500)


class ChatIn(BaseModel):
    title: str = "New chat"


class ChatPatch(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=20000)


class DraftGenerateIn(BaseModel):
    workspace_id: str
    doc_type: str = "custom"
    details: str = Field(default="", max_length=20000)
    ocr_text: str = Field(default="", max_length=40000)


class DraftIn(BaseModel):
    title: str = "Untitled draft"
    doc_type: str = ""
    details: str = ""
    content: str = ""


class DraftPatch(BaseModel):
    title: Optional[str] = None
    doc_type: Optional[str] = None
    details: Optional[str] = None
    content: Optional[str] = None


class PdfIn(BaseModel):
    title: str = "Draft"
    content: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def _need(obj, what="Item"):
    if not obj:
        raise HTTPException(404, f"{what} not found")
    return obj



@app.get("/api/health")
def health():
    from rag.retriever import get_retriever
    r = get_retriever()
    return {
        "ok": True,
        "redis": redis_available(),
        "tracing": os.environ.get("LANGSMITH_TRACING") == "true",
        "hyde": config.HYDE_ENABLED,
        "reranker": r._reranker is not None,
        "ocr": ocr_available(),
        "model": config.LLM_MODEL,
    }


@app.get("/api/meta")
def meta():
    return {"doc_types": DOC_TYPES, "ocr": ocr_available()}



@app.get("/api/workspaces")
def list_workspaces():
    return store.list_workspaces()


@app.post("/api/workspaces")
def create_workspace(body: WorkspaceIn):
    return store.create_workspace(body.name, body.description)


@app.patch("/api/workspaces/{wid}")
def patch_workspace(wid: str, body: WorkspacePatch):
    return _need(store.update_workspace(wid, body.name, body.description), "Workspace")


@app.delete("/api/workspaces/{wid}")
def delete_workspace(wid: str):
    store.delete_workspace(wid)
    return {"ok": True}



@app.get("/api/workspaces/{wid}/chats")
def list_chats(wid: str):
    _need(store.get_workspace(wid), "Workspace")
    return store.list_chats(wid)


@app.post("/api/workspaces/{wid}/chats")
def create_chat(wid: str, body: ChatIn):
    _need(store.get_workspace(wid), "Workspace")
    return store.create_chat(wid, body.title)


@app.get("/api/chats/{cid}")
def get_chat(cid: str):
    chat = _need(store.get_chat(cid), "Chat")
    return {**chat, "messages": store.get_messages(cid)}


@app.patch("/api/chats/{cid}")
def rename_chat(cid: str, body: ChatPatch):
    return _need(store.rename_chat(cid, body.title), "Chat")


@app.delete("/api/chats/{cid}")
def delete_chat(cid: str):
    store.delete_chat(cid)
    return {"ok": True}


@app.post("/api/chats/{cid}/messages")
async def send_message(cid: str, body: MessageIn):
    chat = _need(store.get_chat(cid), "Chat")
    history = store.recent_history(cid, config.HISTORY_TURNS)
    question = body.content.strip()

    store.add_message(cid, "user", question)
    if chat["title"] == "New chat":
        store.rename_chat(cid, question[:48] + ("…" if len(question) > 48 else ""))

    engine = get_engine()

    async def stream():
        parts, sources, meta, saved = [], [], {}, False
        try:
            async for ev in engine.answer_stream(question, history):
                if ev["type"] == "token":
                    parts.append(ev["text"])
                elif ev["type"] == "replace":
                    parts[:] = [ev["text"]]
                elif ev["type"] == "sources":
                    sources = ev["sources"]
                elif ev["type"] == "done":
                    meta = {k: v for k, v in ev.items() if k != "type"}
                yield _sse(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:  
            log_event("chat_error", error=str(e)[:300])
            parts.append("\n\nSorry - something went wrong while answering. Please try again.")
            yield _sse({"type": "replace", "text": "".join(parts)})
            yield _sse({"type": "done", "error": True})
        finally:
            text = "".join(parts).strip()
            if text and not saved:
                store.add_message(cid, "assistant", text, sources, meta)
                saved = True

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/workspaces/{wid}/drafts")
def list_drafts(wid: str):
    _need(store.get_workspace(wid), "Workspace")
    return store.list_drafts(wid)


@app.post("/api/workspaces/{wid}/drafts")
def create_draft(wid: str, body: DraftIn):
    _need(store.get_workspace(wid), "Workspace")
    return store.create_draft(wid, body.title, body.doc_type, body.details, body.content)


@app.get("/api/drafts/{did}")
def get_draft(did: str):
    return _need(store.get_draft(did), "Draft")


@app.patch("/api/drafts/{did}")
def patch_draft(did: str, body: DraftPatch):
    return _need(store.update_draft(did, **body.model_dump()), "Draft")


@app.delete("/api/drafts/{did}")
def delete_draft(did: str):
    store.delete_draft(did)
    return {"ok": True}


@app.post("/api/drafts/generate")
async def generate_draft(body: DraftGenerateIn):
    ws = _need(store.get_workspace(body.workspace_id), "Workspace")
    name = next((d["name"] for d in DOC_TYPES if d["id"] == body.doc_type), body.doc_type or "legal document")
    workspace_ctx = f"{ws['name']}. {ws['description']}".strip()
    engine = get_draft_engine()

    async def stream():
        try:
            async for ev in engine.generate_stream(name, body.details, body.ocr_text, workspace_ctx):
                yield _sse(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_event("draft_error", error=str(e)[:300])
            yield _sse({"type": "error", "message": "Drafting failed. Please try again."})

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/pdf")
async def make_pdf(body: PdfIn):
    pdf = await asyncio.to_thread(build_pdf, body.title, body.content)
    fname = "".join(c if c.isalnum() or c in "-_ " else "_" for c in body.title).strip().replace(" ", "_") or "draft"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{fname}.pdf"'})


ALLOWED_OCR = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}


@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_OCR:
        raise HTTPException(400, "Upload an image (png, jpg, webp, tiff) or a PDF.")
    if not ocr_available() and ext != ".pdf":
        raise HTTPException(503, "OCR engine (PaddleOCR) is not installed on the server.")
    data = await file.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 12 MB).")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        text = await asyncio.to_thread(extract_text, path)
    except Exception as e:
        log_event("ocr_error", error=str(e)[:300])
        raise HTTPException(500, "Could not read text from that file.")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    guard = check_input(text, mode="draft") if text.strip() else None
    return {"text": text.strip(), "chars": len(text.strip()),
            "warning": guard.message if guard and not guard.allowed and guard.category != "too_long" else None}



if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = os.path.join(FRONTEND_DIST, path)
        if path and os.path.isfile(f):
            return FileResponse(f)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
