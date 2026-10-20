"""Vercel Python function exposing the existing EvolveAI agent."""
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evolveai.agent import EvolveAgent
from evolveai.memory import ExperienceMemory

app = FastAPI(title="EvolveAI API", docs_url=None, redoc_url=None)

# Vercel functions are ephemeral. /tmp is writable, but this SQLite store is not
# durable across cold starts/redeploys. Configure durable storage before relying
# on production memory persistence.
DB_PATH = os.getenv("EVOLVEAI_DB_PATH", "/tmp/evolveai.db")
_memory: ExperienceMemory | None = None
_agent: EvolveAgent | None = None


def get_agent() -> EvolveAgent:
    global _memory, _agent
    if _agent is None:
        _memory = ExperienceMemory(DB_PATH)
        _agent = EvolveAgent(_memory)
    return _agent


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    action: str = Field(default="chat", pattern="^(chat|feedback)$")
    task_id: str = Field(default="", max_length=64)
    score: float = Field(default=0.5, ge=0, le=1)
    feedback: str = Field(default="", max_length=300)


@app.get("/")
@app.get("/api/chat")
@app.get("/chat")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "EvolveAI"}


@app.post("/")
@app.post("/api/chat")
@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        agent = get_agent()
        if payload.action == "feedback":
            if not payload.task_id:
                raise HTTPException(status_code=422, detail="task_id is required for feedback.")
            saved = agent.memory.rate_task(payload.task_id, payload.score, payload.feedback)
            return {"saved": bool(saved)}

        if not payload.message.strip():
            raise HTTPException(status_code=422, detail="message cannot be empty.")
        history = [item.model_dump() for item in payload.history]
        result = agent.run(payload.message.strip(), history=history)
        return {
            "answer": result.get("answer", ""),
            "task_id": result.get("task_id"),
            "mode": result.get("mode", "rules"),
            "reasoning_plan": result.get("reasoning_plan", {}),
        }
    except HTTPException:
        raise
    except Exception as exc:
        # Keep internals out of client responses; server logs retain traceback.
        print(f"EvolveAI API error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="EvolveAI could not process this request. Check function logs and retry.") from exc
