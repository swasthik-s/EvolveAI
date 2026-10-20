"""Language-independent conversation signals using transparent local rules.

This module identifies communicative function, not the user's full meaning.
Signals are evidence for a planner; unknown messages remain unknown.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ConversationFrame:
    text: str
    normalized: str
    speech_act: str
    expects_response: bool
    refers_to_context: bool
    correction_signal: bool
    uncertainty_signal: bool
    confidence: float
    evidence: tuple[str, ...]


_CORRECTIONS = (
    "no,", "no ", "that's wrong", "that is wrong", "not what i meant",
    "i meant", "actually,", "actually ", "you misunderstood",
    "incorrect", "that's not right", "that is not right",
)
_UNCERTAINTY = (
    "maybe", "i think", "not sure", "i'm not sure", "im not sure",
    "perhaps", "could be", "i guess",
)
_CONTEXT_STARTS = (
    "what about", "how about", "why?", "how so", "and ", "then ",
    "that one", "this one", "it?", "does it", "is it", "how much",
    "which one", "explain more", "tell me more",
)
_REQUEST_STARTS = (
    "please ", "can you ", "could you ", "would you ", "help me ",
    "show me ", "give me ", "make me ", "build ", "create ", "write ",
    "find ", "calculate ", "explain ", "tell me ",
)
_QUESTION_STARTS = (
    "what ", "who ", "when ", "where ", "why ", "how ", "which ",
    "is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ",
    "could ", "would ", "should ", "will ", "have ", "has ",
)


def understand_message(message: str, history=None) -> ConversationFrame:
    raw = str(message or "").strip()
    normalized = re.sub(r"\s+", " ", raw.lower()).strip()
    evidence = []
    if not normalized:
        return ConversationFrame(raw, normalized, "empty", False, False, False, False, 1.0, ("empty_input",))

    is_correction = any(normalized.startswith(s) or s in normalized for s in _CORRECTIONS)
    is_uncertain = any(s in normalized for s in _UNCERTAINTY)
    history = history or []
    previous_assistant = next(
        (str(item.get("content", "")).strip() for item in reversed(history)
         if item.get("role") == "assistant" and str(item.get("content", "")).strip()),
        "",
    )
    is_contextual = bool(previous_assistant and (
        normalized.endswith("?") and len(normalized.split()) <= 8
        or normalized.startswith(_CONTEXT_STARTS)
        or re.fullmatch(r"(why|how|what|which|and then)[?!., ]*", normalized)
    ))

    if is_correction:
        speech_act, confidence = "correction", 0.88
        evidence.append("correction_marker")
    elif normalized in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        speech_act, confidence = "greeting", 0.98
        evidence.append("greeting_match")
    elif is_contextual:
        speech_act, confidence = "follow_up", 0.68
        evidence.append("contextual_form_and_prior_turn")
    elif normalized.endswith("?") or normalized.startswith(_QUESTION_STARTS):
        speech_act, confidence = "question", 0.82
        evidence.append("question_form")
    elif normalized.startswith(_REQUEST_STARTS) or re.match(r"^(?:i need|i want|let's|lets)\b", normalized):
        speech_act, confidence = "request", 0.78
        evidence.append("request_form")
    elif is_uncertain:
        speech_act, confidence = "uncertain_statement", 0.72
        evidence.append("uncertainty_marker")
    else:
        speech_act, confidence = "statement", 0.52
        evidence.append("default_statement")

    if is_uncertain:
        evidence.append("uncertainty_marker")
    return ConversationFrame(
        text=raw,
        normalized=normalized,
        speech_act=speech_act,
        expects_response=speech_act in {"question", "request", "follow_up", "correction"},
        refers_to_context=is_contextual,
        correction_signal=is_correction,
        uncertainty_signal=is_uncertain,
        confidence=confidence,
        evidence=tuple(evidence),
    )
