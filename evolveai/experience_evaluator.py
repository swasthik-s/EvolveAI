"""Transparent experience evaluation for EvolveAI's incoming data.

Scores are operational heuristics, not truth judgments. Lexical overlap plus
negation is only a possible contradiction signal and must trigger comparison,
never automatic belief revision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[\w'-]{3,}", re.UNICODE)
_NEGATIONS = {
    "not", "never", "no", "none", "cannot", "can't", "doesn't", "isn't",
    "aren't", "wasn't", "weren't", "without",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower().strip("'-")
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if len(token.strip("'-")) >= 3
    }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class Experience:
    content: str
    source_id: str
    source_uri: str = ""
    domain: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def experience_id(self) -> str:
        raw = "\x1f".join((self.source_id, self.source_uri, self.content))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class Evaluation:
    experience_id: str
    novelty: float
    relevance: float
    uncertainty: float
    contradiction_signal: float
    duplicate_similarity: float
    priority: float
    decision: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


class ExperienceEvaluator:
    """Bounded lexical evaluator with a pluggable relevance scorer."""

    def __init__(
        self,
        *,
        max_content_chars: int = 20000,
        max_known_texts: int = 2000,
        duplicate_threshold: float = 0.88,
        learn_threshold: float = 0.62,
        investigate_threshold: float = 0.42,
        relevance_scorer=None,
    ) -> None:
        self.max_content_chars = max(100, int(max_content_chars))
        self.max_known_texts = max(1, int(max_known_texts))
        self.duplicate_threshold = _clamp(duplicate_threshold)
        self.learn_threshold = _clamp(learn_threshold)
        self.investigate_threshold = _clamp(investigate_threshold)
        self.relevance_scorer = relevance_scorer

    @staticmethod
    def _similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def evaluate(
        self,
        experience: Experience,
        *,
        known_texts: Iterable[str] = (),
        known_terms: Iterable[str] = (),
        expected_domain: str = "",
    ) -> Evaluation:
        content = str(experience.content or "")[:self.max_content_chars].strip()
        terms = _tokens(content)
        bounded_known = [str(item)[:self.max_content_chars] for item in known_texts]
        bounded_known = bounded_known[-self.max_known_texts:]
        known_term_set = set(known_terms)

        similarities = [
            self._similarity(terms, _tokens(text))
            for text in bounded_known if text.strip()
        ]
        duplicate_similarity = max(similarities, default=0.0)
        novelty = (
            1.0 - len(terms & known_term_set) / len(terms)
            if terms else 0.0
        )

        if self.relevance_scorer is not None:
            relevance = _clamp(self.relevance_scorer(experience))
        elif expected_domain and experience.domain:
            relevance = 1.0 if experience.domain == expected_domain else 0.35
        else:
            relevance = 0.5

        uncertainty = 0.0 if not terms else _clamp(
            0.65 * novelty + 0.35 * (1.0 - relevance)
        )

        contradiction_signal = 0.0
        candidate_negation = bool(terms & _NEGATIONS)
        if candidate_negation:
            for previous in bounded_known:
                previous_terms = _tokens(previous)
                overlap = self._similarity(terms, previous_terms)
                previous_negation = bool(previous_terms & _NEGATIONS)
                if overlap >= 0.45 and candidate_negation != previous_negation:
                    contradiction_signal = max(contradiction_signal, overlap)

        priority = _clamp(
            0.30 * novelty + 0.25 * relevance
            + 0.20 * uncertainty + 0.25 * contradiction_signal
        )

        reasons: list[str] = []
        if not content:
            decision = "reject_empty"
            reasons.append("No usable text was supplied.")
        elif duplicate_similarity >= self.duplicate_threshold:
            decision = "reject_duplicate"
            reasons.append("High lexical similarity to a retained experience; preserve the existing record.")
        elif contradiction_signal >= 0.45:
            decision = "compare"
            reasons.append("High-overlap statements differ in negation; compare their sources before revising beliefs.")
        elif priority >= self.learn_threshold:
            decision = "learn_now"
            reasons.append("Heuristic priority exceeds the immediate-learning threshold.")
        elif priority >= self.investigate_threshold:
            decision = "investigate"
            reasons.append("The experience may fill a knowledge gap or merits additional evidence.")
        else:
            decision = "retain"
            reasons.append("Low immediate priority; retain provenance for later review.")

        if novelty >= 0.7:
            reasons.append("Many terms are not present in the supplied known-term set.")
        if relevance == 0.5 and self.relevance_scorer is None and not expected_domain:
            reasons.append("Relevance is neutral because no goal or learned relevance scorer is configured.")
        if contradiction_signal:
            reasons.append("Contradiction signal is lexical only and is not proof of factual conflict.")

        return Evaluation(
            experience_id=experience.experience_id,
            novelty=_clamp(novelty),
            relevance=relevance,
            uncertainty=uncertainty,
            contradiction_signal=contradiction_signal,
            duplicate_similarity=duplicate_similarity,
            priority=priority,
            decision=decision,
            reasons=tuple(reasons),
        )
