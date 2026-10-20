"""Core identity policy for EvolveAI.

Identity is represented as meaning and passed to the symbolic language engine.
It must be resolved locally, before any memory search or web research.
"""
import re

from .language_engine import generate_from_meaning, identity_plan


_IDENTITY_QUESTIONS = {
    "who are you", "who are u", "what are you", "what are u",
    "who am i talking to", "who am i chatting with",
    "what is evolveai", "what's evolveai", "whats evolveai",
    "are you an ai", "are you an artificial intelligence",
}


def answer_identity_question(message):
    normalized = re.sub(r"\s+", " ", str(message or "").strip().lower())
    normalized = normalized.rstrip(" ?.!")
    if normalized not in _IDENTITY_QUESTIONS:
        return None
    return generate_from_meaning(identity_plan())
