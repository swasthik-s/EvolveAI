"""Small deterministic planning layer for EvolveAI.

It does not generate language like an LLM. It resolves lightweight conversational
references and produces a transparent plan that the agent can execute.
"""
import re
from dataclasses import dataclass


_FOLLOWUP_STARTS = (
    "what about", "how about", "why", "how so", "and ", "what then",
    "explain more", "tell me more", "what does that", "is it", "does it",
    "can you explain that", "why is that", "which one", "how much",
)


@dataclass(frozen=True)
class ReasoningPlan:
    original_message: str
    working_query: str
    is_follow_up: bool
    intent: str
    needs_fresh_research: bool
    steps: tuple[str, ...]


def _is_question(text):
    low = text.strip().lower()
    starts = (
        "what ", "who ", "when ", "where ", "why ", "how ", "which ",
        "is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ",
        "could ", "would ", "should ", "will ", "tell me about ",
        "explain ", "define ", "find out ", "look up ",
    )
    return low.endswith("?") or low.startswith(starts)


def _needs_freshness(text):
    terms = {
        "today", "latest", "current", "currently", "now", "recent", "news",
        "weather", "price", "prices", "stock", "stocks", "exchange", "rate",
        "schedule", "availability", "updated",
    }
    words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    return bool(words & terms)


def build_plan(message, history=None):
    """Interpret a message and create an explicit, inspectable execution plan."""
    original = str(message).strip()
    low = original.lower()
    history = history or []
    previous_user = next(
        (
            str(item.get("content", "")).strip()
            for item in reversed(history)
            if item.get("role") == "user" and str(item.get("content", "")).strip()
        ),
        "",
    )

    is_short = len(original.split()) <= 12
    is_follow_up = bool(
        previous_user
        and is_short
        and low != previous_user.lower()
        and (low.endswith("?") or low.startswith(_FOLLOWUP_STARTS))
    )
    working_query = (
        f"{previous_user}\nFollow-up question: {original}"
        if is_follow_up else original
    )

    if low in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        intent = "greeting"
    elif low.startswith(("remember that", "please remember", "my name is", "call me", "i prefer")):
        intent = "memory_update"
    elif re.fullmatch(r"[\d\s.+*/()%\-]+", re.sub(r"^(what is|calculate|compute)\s+", "", low).rstrip(" ?")):
        intent = "calculation"
    elif _is_question(working_query):
        intent = "question"
    else:
        intent = "open_request"

    fresh = _needs_freshness(working_query)
    research = intent == "question"
    steps = ["interpret_message", "retrieve_context", "check_saved_knowledge"]
    if research:
        steps.extend(["research_if_needed", "evaluate_evidence"])
    steps.append("compose_response")

    return ReasoningPlan(
        original_message=original,
        working_query=working_query,
        is_follow_up=is_follow_up,
        intent=intent,
        needs_fresh_research=fresh,
        steps=tuple(steps),
    )
