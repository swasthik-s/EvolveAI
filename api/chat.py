"""Vercel Python function exposing EvolveAI's user and protected admin actions."""
import hashlib
import hmac
import json
import os
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from evolveai.activity import ActivityJournal
from evolveai.agent import EvolveAgent
from evolveai.memory import ExperienceMemory

app = FastAPI(title="EvolveAI API", docs_url=None, redoc_url=None)

# Vercel's local filesystem is ephemeral. Use durable storage before production.
DB_PATH = os.getenv("EVOLVEAI_DB_PATH", "/tmp/evolveai.db")
# Small pause between answer chunks so clients can paint the stream.
# Set EVOLVEAI_STREAM_DELAY_MS=0 in tests.
STREAM_DELAY_S = float(os.getenv("EVOLVEAI_STREAM_DELAY_MS", "12")) / 1000.0
_memory: ExperienceMemory | None = None
_agent: EvolveAgent | None = None
_activity: ActivityJournal | None = None

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".py", ".ts", ".tsx", ".js", ".jsx",
    ".css", ".html", ".xml", ".yml", ".yaml", ".toml", ".rs", ".go", ".java", ".rb",
    ".sh", ".mjs", ".c", ".h", ".cpp", ".sql",
}


def get_agent() -> EvolveAgent:
    global _memory, _agent
    if _agent is None or getattr(_memory, "path", None) != DB_PATH:
        _memory = ExperienceMemory(DB_PATH)
        _agent = EvolveAgent(_memory)
    return _agent


def get_journal() -> ActivityJournal:
    global _activity
    if _activity is None or getattr(_activity, "path", None) != DB_PATH:
        _activity = ActivityJournal(DB_PATH)
    return _activity


def require_admin(token: str | None) -> None:
    expected = os.getenv("EVOLVEAI_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API is not configured.")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Admin authentication failed.")


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class ChatAttachment(BaseModel):
    """A file the browser already read. Bytes are not written to disk."""
    name: str = Field(min_length=1, max_length=180)
    type: str = Field(default="application/octet-stream", max_length=160)
    size: int = Field(default=0, ge=0, le=2_000_000)
    encoding: str = Field(default="text", pattern="^(text|base64|omitted)$")
    content: str = Field(default="", max_length=80_000)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    action: str = Field(default="chat", pattern="^(chat|feedback|admin_overview|admin_tasks|admin_memories|admin_lessons|admin_ghost|admin_add_memory|admin_delete_memory|admin_test|admin_activity)$")
    task_id: str = Field(default="", max_length=64)
    score: float = Field(default=0.5, ge=0, le=1)
    feedback: str = Field(default="", max_length=300)
    key: str = Field(default="", max_length=160)
    value: str = Field(default="", max_length=4000)
    query: str = Field(default="", max_length=1000)
    category: str = Field(default="", max_length=60)
    status: str = Field(default="", max_length=30)
    limit: int = Field(default=50, ge=1, le=500)
    # Opt-in. Omitted or false keeps the original JSON response.
    stream: bool = False
    attachments: list[ChatAttachment] = Field(default_factory=list, max_length=4)


def _is_text_attachment(item: ChatAttachment) -> bool:
    if item.encoding != "text" or not item.content:
        return False
    mime = item.type.lower().split(";", 1)[0].strip()
    if mime.startswith("image/") or mime.startswith("audio/") or mime.startswith("video/"):
        return False
    if mime.startswith("text/") or mime in {"application/json", "application/xml", "application/x-yaml", "application/yaml", "application/javascript", "application/typescript"}:
        return True
    dot = item.name.rfind(".")
    return dot >= 0 and item.name[dot:].lower() in _TEXT_EXTENSIONS


def _ingest_attachments(items: list[ChatAttachment], journal: ActivityJournal) -> list[dict[str, Any]]:
    """Record attachment metadata and a text excerpt in the activity journal.

    The journal lives in the same SQLite file as the agent. On Vercel that file
    is ephemeral. File bytes are not saved to disk and are not executed.
    Images are acknowledged by name only; the engine does not view pixels.
    """
    stored: list[dict[str, Any]] = []
    for item in items:
        digest = hashlib.sha256(item.content.encode("utf-8", errors="replace")).hexdigest() if item.content else ""
        if item.encoding == "omitted" or not item.content:
            policy = "metadata_only"
            excerpt = ""
        elif _is_text_attachment(item):
            policy = "text_excerpt"
            excerpt = item.content[:4000]
        else:
            policy = "binary_metadata"
            excerpt = ""
        record = {
            "name": item.name,
            "type": item.type,
            "size": item.size,
            "encoding": item.encoding,
            "sha256": digest,
            "stored": True,
            "persistence": "activity_journal",
            "content_policy": policy,
            "excerpt": excerpt[:500],
        }
        journal.record(
            "user_task",
            "attachment_received",
            "completed",
            f"Stored attachment {item.name}",
            details={
                "name": item.name,
                "content_type": item.type,
                "bytes": item.size,
                "encoding": item.encoding,
                "sha256": digest,
                "excerpt": excerpt[:1500],
                "persistence": "Metadata and a text excerpt live in the activity journal. File bytes are not written to disk and are not executed.",
            },
        )
        stored.append(record)
    return stored


def _receipt(stored: list[dict[str, Any]]) -> str:
    if not stored:
        return ""
    lines = ["", "", "**Attachments received**"]
    notes = {
        "text_excerpt": "text excerpt saved in the activity journal",
        "binary_metadata": "metadata saved; the engine does not interpret this file",
        "metadata_only": "name and size saved; the body was omitted because it was too large",
    }
    for item in stored:
        lines.append(f"- **{item['name']}** ({item['type'] or 'file'}, {item['size']} bytes) — {notes.get(item['content_policy'], 'saved')}")
        excerpt = str(item.get("excerpt") or "").strip()
        if excerpt:
            preview = " ".join(excerpt.split())
            if len(preview) > 280:
                preview = preview[:277] + "..."
            lines.append(f"  > {preview}")
    lines.append("")
    lines.append("*Uploaded files are not executed. Images are acknowledged, not viewed.*")
    return "\n".join(lines)


def _execute_chat(payload: ChatRequest, progress=None) -> dict[str, Any]:
    agent = get_agent()
    journal = get_journal()
    stored = _ingest_attachments(payload.attachments, journal)
    message = payload.message.strip()
    if not message and stored:
        message = "Please review the attached files."
    if not message:
        raise HTTPException(status_code=422, detail="message cannot be empty.")
    history = [item.model_dump() for item in payload.history]
    journal.record(
        "user_task", "chat_request", "started", "User request accepted by the agent.",
        details={"mode": "user_chat", "attachments": len(stored)},
    )
    result = agent.run(message, history=history, progress=progress)
    answer = str(result.get("answer", "")) + _receipt(stored)
    journal.record(
        "user_task", "chat_request", "completed", "Agent returned a response.",
        task_id=str(result.get("task_id") or ""),
        details={"mode": result.get("mode", "rules"), "attachments": len(stored)},
    )
    body: dict[str, Any] = {
        "answer": answer,
        "task_id": result.get("task_id"),
        "mode": result.get("mode", "rules"),
        "reasoning_plan": result.get("reasoning_plan", {}),
    }
    if stored:
        body["attachments"] = stored
    return body


def _dispatch(payload: ChatRequest, x_admin_token: str | None) -> dict[str, Any]:
    # Admin actions are authenticated at the API boundary, not just in Streamlit UI.
    if payload.action.startswith("admin_"):
        require_admin(x_admin_token)
        agent = get_agent()
        memory = agent.memory
        activity = get_journal()
        if payload.action == "admin_activity":
            return {"events": activity.recent(payload.limit, payload.category, payload.status, payload.query), "activity_stats": activity.stats()}

        if payload.action == "admin_overview":
            return {
                "stats": memory.stats(),
                "strategies": memory.strategy_statistics(),
                "api": "ok",
                "storage": "sqlite_ephemeral" if DB_PATH.startswith("/tmp/") else "configured_path",
            }
        if payload.action == "admin_tasks":
            return {"tasks": memory.recent_tasks(payload.limit)}
        if payload.action == "admin_memories":
            return {"memories": memory.memories()}
        if payload.action == "admin_lessons":
            return {"lessons": memory.search_lessons(payload.query, payload.limit)}
        if payload.action == "admin_ghost":
            return {"knowledge": memory.search_ghost_knowledge(payload.query, payload.limit)}
        if payload.action == "admin_add_memory":
            if not memory.remember(payload.key, payload.value, source="admin"):
                raise HTTPException(status_code=422, detail="Both key and value are required.")
            return {"saved": True, "memories": memory.memories()}
        if payload.action == "admin_delete_memory":
            if not payload.key.strip():
                raise HTTPException(status_code=422, detail="key is required.")
            return {"deleted": memory.forget(payload.key)}
        if payload.action == "admin_test":
            if not payload.message.strip():
                raise HTTPException(status_code=422, detail="message cannot be empty.")
            result = agent.run(payload.message.strip(), history=[])
            return {
                "answer": result.get("answer", ""),
                "task_id": result.get("task_id"),
                "mode": result.get("mode", "rules"),
                "reasoning_plan": result.get("reasoning_plan", {}),
            }

    if payload.action == "feedback":
        if not payload.task_id:
            raise HTTPException(status_code=422, detail="task_id is required for feedback.")
        saved = get_agent().memory.rate_task(payload.task_id, payload.score, payload.feedback)
        return {"saved": bool(saved)}

    return _execute_chat(payload)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 48) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + size)
        if end < length:
            space = text.rfind(" ", start + 12, min(length, end + 16))
            if space > start:
                end = space + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def iter_chat_sse(payload: ChatRequest) -> Iterator[str]:
    """Stream stage events while the agent runs, then the answer in chunks.

    The engine still produces one complete answer. Chunks are slices of that
    answer so the UI can render them as they arrive. Clients that do not set
    stream=true keep receiving a single JSON object.
    """
    events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
    holder: dict[str, Any] = {}

    def progress(label: str) -> None:
        events.put(("status", {"stage": str(label)}))

    def worker() -> None:
        try:
            holder["result"] = _execute_chat(payload, progress=progress)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
            holder["error"] = detail
        except Exception as exc:
            print(f"EvolveAI API error: {type(exc).__name__}: {exc}")
            holder["error"] = "EvolveAI could not process this request. Check function logs and retry."
        finally:
            events.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        try:
            item = events.get(timeout=0.05)
        except queue.Empty:
            if not thread.is_alive() and events.empty():
                break
            continue
        if item is None:
            break
        kind, data = item
        yield _sse(kind, data)
    thread.join(timeout=1)
    if holder.get("error"):
        yield _sse("error", {"detail": str(holder["error"])})
        return
    result = holder.get("result")
    if not isinstance(result, dict):
        yield _sse("error", {"detail": "EvolveAI could not process this request. Check function logs and retry."})
        return
    answer = str(result.get("answer") or "")
    for piece in _chunk_text(answer):
        yield _sse("delta", {"text": piece})
        if STREAM_DELAY_S > 0:
            time.sleep(STREAM_DELAY_S)
    yield _sse("done", result)


@app.get("/")
@app.get("/api/chat")
@app.get("/chat")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "EvolveAI"}


@app.post("/")
@app.post("/api/chat")
@app.post("/chat")
def chat(
    payload: ChatRequest,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    # Streaming is opt-in and only for the user chat action, so admin and
    # feedback clients keep receiving JSON.
    if payload.stream and payload.action == "chat":
        if not payload.message.strip() and not payload.attachments:
            raise HTTPException(status_code=422, detail="message cannot be empty.")
        return StreamingResponse(
            iter_chat_sse(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    try:
        return _dispatch(payload, x_admin_token)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"EvolveAI API error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="EvolveAI could not process this request. Check function logs and retry.") from exc
