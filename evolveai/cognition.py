"""A small, inspectable language-understanding layer built without a generative model.

It converts a message into a structured frame: intent, subject, aliases, and a
search query. Rules are deliberately explicit and testable; unknown cases fall
back to the user's original wording rather than inventing meaning.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Understanding:
    original: str
    normalized: str
    intent: str
    subject: str
    search_query: str
    entities: tuple[str, ...]
    confidence: float
    explanation: str


# A tiny seed ontology. It is data, not a claim that this list covers language.
# Grow it from curated knowledge and user corrections, not blind web scraping.
_ENTITY_ALIASES = {
    "iron man": ("iron man", "ironman", "tony stark", "marvel iron man"),
    "ultron": ("ultron", "marvel ultron", "avengers ultron"),
    "captain america": ("captain america", "steve rogers"),
    "spider-man": ("spider-man", "spiderman", "peter parker"),
}
_QUESTION_PATTERNS = (
    (re.compile(r"^(?:who is|who's|tell me who)\s+(.+?)\??$", re.I), "person_or_character"),
    (re.compile(r"^(?:what is|what's|define|explain)\s+(.+?)\??$", re.I), "definition"),
    (re.compile(r"^(?:tell me about|explain)\s+(.+?)\??$", re.I), "overview"),
    (re.compile(r"^(?:why is|why does|why do)\s+(.+?)\??$", re.I), "why"),
    (re.compile(r"^(?:how does|how do|how can)\s+(.+?)\??$", re.I), "how"),
)
_STOP_END = re.compile(r"[?!.]+$")


def _normalize(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _resolve_entity(subject: str):
    normalized = _normalize(subject)
    # Prefer the longest alias match to avoid partial-entity collisions.
    matches = []
    for canonical, aliases in _ENTITY_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize(alias)
            if re.search(r"(?<!\w)" + re.escape(alias_norm) + r"(?!\w)", normalized):
                matches.append((len(alias_norm), canonical, alias))
    if not matches:
        return subject.strip(), ()
    _, canonical, _ = max(matches)
    return canonical.title(), tuple(_ENTITY_ALIASES[canonical])


def understand(message: str, history=None) -> Understanding:
    """Parse common question forms and entity aliases into a transparent frame."""
    original = str(message or "").strip()
    normalized = _normalize(original)
    history = history or []
    subject = ""
    intent = "open_request"
    explanation = "No high-confidence pattern matched; preserve the original wording."

    for pattern, candidate_intent in _QUESTION_PATTERNS:
        match = pattern.match(original.strip())
        if match:
            subject = _STOP_END.sub("", match.group(1)).strip(" \"'“”")
            intent = candidate_intent
            explanation = f"Matched a {candidate_intent} question pattern."
            break

    # Also handle a bare subject question such as "Iron Man?" without guessing
    # that every short phrase is a factual question.
    if not subject and original.endswith("?") and len(original.split()) <= 6:
        candidate = _STOP_END.sub("", original).strip()
        canonical, aliases = _resolve_entity(candidate)
        if aliases:
            subject, intent = canonical, "overview"
            explanation = "Recognized a known entity alias in a short question."

    if subject:
        canonical, aliases = _resolve_entity(subject)
        if aliases:
            subject = canonical
            search_query = f"{subject} fictional character identity background powers story"
            confidence = 0.96
            entities = (subject,)
            explanation += " Canonicalized a known alias using the local entity lexicon."
        else:
            # Keep the user-provided subject. Do not silently map it to a guess.
            search_query = f"{subject} {intent} reliable sources"
            confidence = 0.68
            entities = (subject,)
    else:
        search_query = original
        confidence = 0.35
        entities = ()

    return Understanding(
        original=original,
        normalized=normalized,
        intent=intent,
        subject=subject,
        search_query=search_query,
        entities=entities,
        confidence=confidence,
        explanation=explanation,
    )
