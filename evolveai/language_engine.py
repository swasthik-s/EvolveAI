"""A first symbolic English realisation engine for EvolveAI.

This module generates sentences from semantic structures, rather than selecting
whole canned replies. It is intentionally inspectable and model-free.

Scope: grammar and wording are composed from reusable lexical and grammatical
parts. This is a foundation, not unrestricted human-level language generation.
"""
from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class Proposition:
    subject: str
    relation: str
    complement: str
    tense: str = "present"
    modality: Optional[str] = None


@dataclass(frozen=True)
class MessagePlan:
    propositions: tuple[Proposition, ...]
    speech_act: str = "statement"
    tone: str = "neutral"
    detail: str = "brief"
    acknowledgement: Optional[str] = None


def _clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _capitalize(text):
    text = _clean(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def _punctuate(text, mark="."):
    text = _clean(text).rstrip(" .!?")
    return text + mark if text else ""


def _copula(subject, tense="present"):
    first = subject.strip().lower()
    if tense == "past":
        return "was" if first in {"i", "he", "she", "it", "evolveai"} else "were"
    if first == "i":
        return "am"
    if first in {"you", "we", "they"}:
        return "are"
    return "is"


def realize_proposition(proposition: Proposition) -> str:
    """Realise a simple semantic proposition as a grammatical English clause."""
    subject = _clean(proposition.subject)
    complement = _clean(proposition.complement)
    relation = proposition.relation.strip().lower()
    if not subject or not complement:
        return ""

    if relation in {"be", "identity", "description", "state"}:
        verb = _copula(subject, proposition.tense)
        if proposition.modality:
            verb = proposition.modality.strip() + " " + _base_be(subject)
        return _punctuate(f"{subject} {verb} {complement}")

    if relation == "have":
        verb = "have" if subject.lower() in {"i", "you", "we", "they"} else "has"
        if proposition.tense == "past":
            verb = "had"
        if proposition.modality:
            verb = proposition.modality.strip() + " have"
        return _punctuate(f"{subject} {verb} {complement}")

    if relation == "do":
        verb = _inflect_verb(complement, subject, proposition.tense)
        return _punctuate(f"{subject} {verb}")

    # Unknown relations are rendered conservatively as a copular statement.
    return _punctuate(f"{subject} {_copula(subject, proposition.tense)} {complement}")


def _base_be(subject):
    return "be"


def _inflect_verb(verb_phrase, subject, tense):
    phrase = _clean(verb_phrase)
    if tense == "past":
        irregular = {"learn": "learned", "build": "built", "go": "went", "do": "did"}
        first, *rest = phrase.split(" ", 1)
        inflected = irregular.get(first.lower(), first + ("d" if first.endswith("e") else "ed"))
        return " ".join([inflected] + rest)
    if subject.lower() in {"i", "you", "we", "they"}:
        return phrase
    first, *rest = phrase.split(" ", 1)
    lower = first.lower()
    if lower.endswith(("s", "sh", "ch", "x", "z", "o")):
        inflected = first + "es"
    elif lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        inflected = first[:-1] + "ies"
    else:
        inflected = first + "s"
    return " ".join([inflected] + rest)


def realize_plan(plan: MessagePlan) -> str:
    """Turn a content plan into an answer with basic discourse-level control."""
    clauses = [realize_proposition(p).rstrip(".!?") for p in plan.propositions]
    clauses = [c for c in clauses if c]
    if not clauses:
        return "I don't have enough information to form a reliable answer yet."

    if plan.speech_act == "question":
        text = " ".join(clauses)
        return _punctuate(text, "?")

    if plan.detail == "brief":
        clauses = clauses[:2]

    if plan.tone == "warm" and plan.acknowledgement:
        acknowledgement = _punctuate(plan.acknowledgement)
        clauses = [acknowledgement] + [_punctuate(_capitalize(clause)) for clause in clauses]
    else:
        clauses = [_punctuate(_capitalize(clause)) for clause in clauses]
    return " ".join(clauses)


def identity_plan(name="EvolveAI", repeated=False):
    """Return semantic content for the assistant's identity, not a canned sentence."""
    propositions = (
        Proposition("I", "be", f"{name}, an experimental AI assistant"),
        Proposition("I", "do", "use explicit rules, memory, research, and feedback to improve"),
    )
    if repeated:
        return MessagePlan(
            propositions=(
                Proposition("I", "be", name),
                Proposition("my identity", "be", "the same, even when I explain it differently"),
            ),
            tone="warm",
            detail="brief",
            acknowledgement="Still here",
        )
    return MessagePlan(propositions=propositions, detail="brief")


def user_identity_plan(name, repeated=False):
    subject_name = _clean(name)
    if not subject_name:
        return MessagePlan((Proposition("you", "be", "the person I'm talking with"),), detail="brief")
    if repeated:
        return MessagePlan(
            propositions=(
                Proposition("you", "be", subject_name),
                Proposition("I", "do", "still have that context"),
            ),
            tone="warm",
            detail="brief",
            acknowledgement="Yep",
        )
    return MessagePlan((Proposition("you", "be", subject_name),), detail="brief")


def generate_from_meaning(plan: MessagePlan) -> str:
    """Public entry point for future semantic planners."""
    result = realize_plan(plan)
    # Defensive cleanup: avoid doubled spaces and malformed punctuation.
    return re.sub(r"\s+([,.!?])", r"\1", result).strip()
