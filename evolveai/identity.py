"""Core identity policy for EvolveAI.

Identity is represented as meaning and passed to the symbolic language engine.
It must be resolved locally, before any memory search or web research.
"""
import re

from .language_engine import generate_from_meaning, identity_plan


_SUPERPOWER_QUESTIONS = {
    "what is your superpower", "whats your superpower", "what's your superpower",
    "what is your super power", "whats your super power", "what's your super power",
    "what would you say is your superpower", "what is your greatest strength",
    "whats you superpower", "what is you superpower",
}


def _answer_superpower_question():
    """Answer self-referential questions as EvolveAI, not as interview advice."""
    from .language_engine import MessagePlan, Proposition

    plan = MessagePlan(
        propositions=(
            Proposition("My best strength", "be", "connecting the dots across a messy problem"),
            Proposition("I", "do", "combine relevant context, retrieved evidence, and explicit reasoning steps into a clearer answer or plan"),
            Proposition("I", "be", "still an experimental system, so I can get things wrong and should show uncertainty when evidence is weak"),
        ),
        tone="warm",
        detail="brief",
        acknowledgement="Hmm, good question 😄",
    )
    return generate_from_meaning(plan)


_IDENTITY_QUESTIONS = {
    "who are you", "who are u", "what are you", "what are u",
    "who am i talking to", "who am i chatting with",
    "what is evolveai", "what's evolveai", "whats evolveai",
    "are you an ai", "are you an artificial intelligence",
}


def answer_identity_question(message):
    normalized = re.sub(r"\s+", " ", str(message or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9]+$", "", normalized)
    if normalized not in _IDENTITY_QUESTIONS:
        return None
    return generate_from_meaning(identity_plan())
