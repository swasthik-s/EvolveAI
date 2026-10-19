"""Core identity policy for EvolveAI. Identity is not a web-research topic."""
import re


def answer_identity_question(message):
    normalized = re.sub(r"\s+", " ", str(message or "").strip().lower())
    normalized = normalized.rstrip(" ?.!")

    if normalized in {
        "who are you", "who are u", "what are you", "what are u",
        "who am i talking to", "who am i chatting with",
        "what is evolveai", "what's evolveai", "whats evolveai",
        "are you an ai", "are you an artificial intelligence",
    }:
        return (
            "I'm EvolveAI, your experimental AI assistant. I use explicit rules, memory, "
            "research, and feedback-based learning rather than a generative LLM. I'm still "
            "developing, so I can make mistakes, but I should check context before searching."
        )
    return None
