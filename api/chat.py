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
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


class FeedbackRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=64)
    score: float = Field(ge=0, le=1)
    feedback: str = Field(default="", max_length=300)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "EvolveAI"}


@app.post("/")
def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        history = [item.model_dump() for item in payload.history]
        result = get_agent().run(payload.message.strip(), history=history)
        return {
            "answer": result.get("answer", ""),
            "task_id": result.get("task_id"),
            "mode": result.get("mode", "rules"),
            "reasoning_plan": result.get("reasoning_plan", {}),
        }
    except Exception as exc:
        # Keep internals out of client responses; server logs retain traceback.
        print(f"EvolveAI chat error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="EvolveAI could not process this message. Check function logs and retry.") from exc


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict[str, bool]:
    try:
        saved = get_agent().memory.rate_task(payload.task_id, payload.score, payload.feedback)
        return {"saved": bool(saved)}
    except Exception as exc:
        print(f"EvolveAI feedback error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Feedback could not be saved.") from exc
