"""Heuristic relevance gate for memory and research candidates."""
import re
from dataclasses import dataclass

STOP = set("what when where which who whom whose why how does did have has had will would could should can the and for from with that this about into your you are was were is not am me my it he she they them we us tell explain find look search know please".split())
FRESH = set("today latest current currently now recent news weather price prices schedule availability updated".split())


@dataclass(frozen=True)
class Relevance:
    accepted: bool
    score: float
    reason: str
    matched_terms: tuple[str, ...]


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def terms(text):
    return {w for w in normalize(text).split() if len(w) > 2 and w not in STOP}


def is_personal_context_query(query):
    low = normalize(query)
    return low in {
        "who am i", "what is my name", "whats my name",
        "what do you know about me", "what do you remember about me",
        "what have i told you", "what did i tell you",
    }


def assess_relevance(query, title, evidence, saved_question=""):
    """Reject personal questions and candidates with weak subject alignment."""
    if is_personal_context_query(query):
        return Relevance(False, 0.0, "Personal questions must use personal context, not web knowledge.", ())
    q = terms(query)
    if not q:
        return Relevance(False, 0.0, "No discriminative subject terms in the query.", ())
    key = terms(saved_question)
    body = terms(str(title or "") + " " + str(evidence or ""))
    key_matches = q & key
    body_matches = q & body
    key_coverage = len(key_matches) / len(q)
    body_coverage = len(body_matches) / len(q)
    score = 0.7 * key_coverage + 0.3 * body_coverage
    matched = tuple(sorted(key_matches | body_matches))
    if key and key_coverage < 0.5:
        return Relevance(False, round(score, 3), "Saved question is not sufficiently similar to the current question.", matched)
    if body_coverage < 0.35:
        return Relevance(False, round(score, 3), "Evidence does not cover enough of the query subject.", matched)
    if score < 0.42:
        return Relevance(False, round(score, 3), "Candidate score is below the acceptance threshold.", matched)
    return Relevance(True, round(score, 3), "Subject overlap passed the heuristic threshold.", matched)
