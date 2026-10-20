"""Feedback endpoint for the Vercel Python runtime."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.chat import get_agent

app = FastAPI(title="EvolveAI Feedback API", docs_url=None, redoc_url=None)


class FeedbackRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=64)
    score: float = Field(ge=0, le=1)
    feedback: str = Field(default="", max_length=300)


@app.post("/")
def save_feedback(payload: FeedbackRequest) -> dict[str, bool]:
    try:
        saved = get_agent().memory.rate_task(payload.task_id, payload.score, payload.feedback)
        return {"saved": bool(saved)}
    except Exception as exc:
        print(f"EvolveAI feedback error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Feedback could not be saved.") from exc
