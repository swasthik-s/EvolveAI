"""Conversation-aware response adaptation without generative models.

The layer keeps factual content intact but changes conversational strategy when
a user repeats a question. It uses recent turn history, not random template rotation.
"""
import re


def normalize_message(text):
    text = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return re.sub(r"[?.!]+$", "", text)


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

    if intent == "assistant_identity":
        if count == 1:
            return (
                "Still me—EvolveAI 😄. I’m the same assistant, but I can explain myself "
                "from a different angle if you’re testing how I respond."
            )
        return (
            "Checking whether I’ll change my story? 😄 I’m EvolveAI. My identity stays the "
            "same; my explanation can change with the conversation."
        )

    if intent == "personal_context":
        name = _personal_name(answer)
        if name:
            if count == 1:
                return f"Still you, {name} 😄 I haven’t lost track."
            return f"Yep, {name—you’re still you 😄. I remember; no need to reintroduce yourself."
        if count == 1:
            return "You’re the person I’m chatting with 😄. I still don’t have your name saved, so I won’t guess."
        return "Same answer as before: you’re the person I’m chatting with. I’d need your name to identify you more specifically."

    # Keep facts and citations intact; acknowledge repetition instead of pretending
    # this is a new discovery.
    if count == 1:
        return "I’ve answered this earlier in our chat, so I’ll keep the answer consistent:\n\n" + answer
    return "We’ve come back to this a few times. I’ll keep the facts consistent rather than invent a new answer:\n\n" + answer
