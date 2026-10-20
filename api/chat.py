"""Vercel Python function exposing EvolveAI's user and protected admin actions."""
import hmac
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from evolveai.agent import EvolveAgent
from evolveai.memory import ExperienceMemory
from evolveai.activity import ActivityJournal

app = FastAPI(title="EvolveAI API", docs_url=None, redoc_url=None)

# Vercel's local filesystem is ephemeral. Use durable storage before production.
DB_PATH = os.getenv("EVOLVEAI_DB_PATH", "/tmp/evolveai.db")
_memory: ExperienceMemory | None = None
_agent: EvolveAgent | None = None
_activity: ActivityJournal | None = None


def get_agent() -> EvolveAgent:
    global _memory, _agent
    if _agent is None:
        _memory = ExperienceMemory(DB_PATH)
        _agent = EvolveAgent(_memory)
    return _agent


def require_admin(token: str | None) -> None:
    expected = os.getenv("EVOLVEAI_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API is not configured.")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Admin authentication failed.")


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


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
) -> dict[str, Any]:
    try:
        # Admin actions are authenticated at the API boundary, not just in Streamlit UI.
        if payload.action.startswith("admin_"):
            require_admin(x_admin_token)
            agent = get_agent()
            memory = agent.memory
            activity = _activity or ActivityJournal(DB_PATH)
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

        agent = get_agent()
        if payload.action == "feedback":
            if not payload.task_id:
                raise HTTPException(status_code=422, detail="task_id is required for feedback.")
            saved = agent.memory.rate_task(payload.task_id, payload.score, payload.feedback)
            return {"saved": bool(saved)}

        if not payload.message.strip():
            raise HTTPException(status_code=422, detail="message cannot be empty.")
        history = [item.model_dump() for item in payload.history]
        journal = _activity or ActivityJournal(DB_PATH)
        journal.record("user_task", "chat_request", "started", "User request accepted by the agent.", details={"mode": "user_chat"})
        result = agent.run(payload.message.strip(), history=history)
        journal.record("user_task", "chat_request", "completed", "Agent returned a response.", task_id=str(result.get("task_id") or ""), details={"mode": result.get("mode", "rules")})
        return {
            "answer": result.get("answer", ""),
            "task_id": result.get("task_id"),
            "mode": result.get("mode", "rules"),
            "reasoning_plan": result.get("reasoning_plan", {}),
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"EvolveAI API error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="EvolveAI could not process this request. Check function logs and retry.") from exc
