"""Conversation-aware response adaptation without generative models.

The layer keeps factual content intact but changes conversational strategy when
a user repeats a question. It uses recent turn history, not random template rotation.
"""
import re

from .language_engine import generate_from_meaning, identity_plan, user_identity_plan


def normalize_message(text):
    text = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return re.sub(r"[^a-z0-9]+$", "", text)


def repeated_question_count(message, history=None):
    target = normalize_message(message)
    if not target:
        return 0
    return sum(
        1 for item in (history or [])
        if item.get("role") == "user"
        and normalize_message(item.get("content", "")) == target
    )


def _personal_name(answer):
    match = re.search(r"You're\s+(.+?),\s+the person I'm talking to", answer, re.I)
    return match.group(1).strip() if match else None


def adapt_response(message, answer, history=None, intent="general"):
    """Adapt delivery to repetition while preserving the underlying answer.

    This deliberately does not rewrite researched facts or invent new knowledge.
    """
    count = repeated_question_count(message, history)
    if count == 0:
        return answer

    # Repeated greetings should acknowledge conversational context rather than
    # replay the exact same opener. History excludes the current user turn.
    greeting = normalize_message(message)
    if greeting in {"hi", "hey", "hello", "good morning", "good afternoon", "good evening",
                    "hi evolveai", "hello evolveai"}:
        return "Hey again 😄 I'm here. What's on your mind?"

    if intent == "assistant_identity":
        plan = identity_plan(repeated=True)
        return generate_from_meaning(plan) + (" 😄" if count == 1 else " I can explain the same idea another way.")

    if intent == "personal_context":
        name = _personal_name(answer)
        plan = user_identity_plan(name, repeated=True)
        return generate_from_meaning(plan) + (" 😄" if count == 1 else "")

    # Repetition alone is not a reason to add a canned disclaimer.
    # Preserve the answer unless there is a meaningful, intent-specific adaptation.
    return answer
