"""Experience-driven curiosity primitives for EvolveAI.

This module does not contain a subject list or a bank of questions. It derives
investigation candidates from observed text, novelty, weakly supported
co-occurrences, and prediction/evidence gaps. It is a transparent bootstrap
heuristic, not semantic understanding or a general reasoning model.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[\w'-]{3,}", re.UNICODE)
_DEFAULT_STOP = {
    "about", "after", "again", "against", "also", "because", "been", "before",
    "being", "between", "both", "could", "does", "doing", "during", "each",
    "from", "have", "having", "into", "just", "more", "most", "other", "over",
    "same", "some", "such", "than", "that", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "under", "very", "what",
    "when", "where", "which", "while", "with", "would", "your",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str, stop_words: set[str]) -> list[str]:
    return [
        token.lower().strip("'-")
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if token.lower() not in stop_words and len(token.strip("'-")) >= 3
    ]


def _stable_id(*parts: str) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class Observation:
    content: str
    source_id: str
    source_uri: str = ""
    observed_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def observation_id(self) -> str:
        return _stable_id(self.source_id, self.source_uri, self.content)


@dataclass(frozen=True)
class Investigation:
    investigation_id: str
    gap_type: str
    target: str
    rationale: str
    evidence_ids: tuple[str, ...]
    priority: float
    question: str
    status: str = "proposed"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence_ids"] = list(self.evidence_ids)
        return result


class CuriosityEngine:
    """Persistent, bounded curiosity queue derived from observed experiences.

    The initial algorithm discovers candidate terms and under-supported
    relationships. It deliberately does not assert that co-occurrence implies
    a semantic or causal relationship.
    """

    def __init__(
        self,
        db_path: str = "data/curiosity.db",
        *,
        stop_words: Iterable[str] | None = None,
        max_content_chars: int = 20000,
        max_terms_per_observation: int = 80,
        max_active: int = 500,
    ) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.stop_words = set(stop_words or _DEFAULT_STOP)
        self.max_content_chars = max(100, int(max_content_chars))
        self.max_terms_per_observation = max(5, int(max_terms_per_observation))
        self.max_active = max(1, int(max_active))
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_uri TEXT NOT NULL,
                content TEXT NOT NULL, observed_at TEXT NOT NULL, metadata_json TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS concepts (
                term TEXT PRIMARY KEY, observation_count INTEGER NOT NULL,
                first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS cooccurrences (
                left_term TEXT NOT NULL, right_term TEXT NOT NULL,
                observation_count INTEGER NOT NULL,
                PRIMARY KEY(left_term, right_term)
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY, gap_type TEXT NOT NULL, target TEXT NOT NULL,
                rationale TEXT NOT NULL, evidence_json TEXT NOT NULL,
                priority REAL NOT NULL, question TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(gap_type, target)
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_investigations_status_priority ON investigations(status, priority DESC)")

    def observe(self, observation: Observation) -> list[Investigation]:
        content = str(observation.content or "")[:self.max_content_chars].strip()
        if not content:
            return []
        obs_id = observation.observation_id
        terms = list(dict.fromkeys(_tokens(content, self.stop_words)))[:self.max_terms_per_observation]
        if not terms:
            return []

        with self._connect() as conn:
            existing = conn.execute("SELECT 1 FROM observations WHERE id=?", (obs_id,)).fetchone()
            if existing:
                return self.pending(limit=20)

            conn.execute(
                "INSERT INTO observations VALUES(?,?,?,?,?,?)",
                (obs_id, observation.source_id, observation.source_uri, content,
                 observation.observed_at, json.dumps(observation.metadata, ensure_ascii=False, default=str)),
            )

            new_terms: list[str] = []
            for term in terms:
                row = conn.execute("SELECT observation_count FROM concepts WHERE term=?", (term,)).fetchone()
                if row is None:
                    new_terms.append(term)
                    conn.execute(
                        "INSERT INTO concepts VALUES(?,?,?,?)",
                        (term, 1, observation.observed_at, observation.observed_at),
                    )
                else:
                    conn.execute(
                        "UPDATE concepts SET observation_count=?, last_seen=? WHERE term=?",
                        (int(row["observation_count"]) + 1, observation.observed_at, term),
                    )

            # Store co-occurrence as a weak observation, never as a verified relation.
            unique_terms = sorted(set(terms))
            for i, left in enumerate(unique_terms):
                for right in unique_terms[i + 1:]:
                    conn.execute(
                        """INSERT INTO cooccurrences VALUES(?,?,1)
                           ON CONFLICT(left_term,right_term)
                           DO UPDATE SET observation_count=observation_count+1""",
                        (left, right),
                    )

        generated: list[Investigation] = []
        for term in new_terms:
            generated.append(self._propose(
                gap_type="unfamiliar_concept",
                target=term,
                rationale="A new lexical concept candidate appeared in an observation; its meaning and properties are not established in memory.",
                evidence_ids=(obs_id,),
                priority=0.65,
                question=f"What does '{term}' mean in this observed context, and what evidence describes it?",
            ))

        # Repeated co-occurrence can motivate investigation, but cannot establish meaning.
        if len(terms) >= 2:
            pairs = list(zip(unique_terms, unique_terms[1:]))
            for left, right in pairs[:8]:
                generated.append(self._propose(
                    gap_type="under_supported_relationship",
                    target=f"{left}::{right}",
                    rationale="These concepts appeared together in one observation; the nature and reliability of their relationship remain unknown.",
                    evidence_ids=(obs_id,),
                    priority=0.35,
                    question=f"What, if any, supported relationship exists between '{left}' and '{right}'?",
                ))

        return generated

    def _propose(
        self, *, gap_type: str, target: str, rationale: str,
        evidence_ids: tuple[str, ...], priority: float, question: str,
    ) -> Investigation:
        investigation_id = _stable_id(gap_type, target)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO investigations
                   (id,gap_type,target,rationale,evidence_json,priority,question,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(gap_type,target) DO UPDATE SET
                     priority=MAX(priority, excluded.priority)""",
                (investigation_id, gap_type, target, rationale,
                 json.dumps(list(evidence_ids)), float(priority), question, "proposed", utc_now()),
            )
            row = conn.execute("SELECT * FROM investigations WHERE id=?", (investigation_id,)).fetchone()
        return self._row_to_investigation(row)

    @staticmethod
    def _row_to_investigation(row: sqlite3.Row) -> Investigation:
        return Investigation(
            investigation_id=row["id"], gap_type=row["gap_type"], target=row["target"],
            rationale=row["rationale"], evidence_ids=tuple(json.loads(row["evidence_json"])),
            priority=float(row["priority"]), question=row["question"], status=row["status"],
        )

    def pending(self, limit: int = 50) -> list[Investigation]:
        bounded = max(1, min(int(limit), self.max_active))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM investigations WHERE status='proposed' ORDER BY priority DESC, created_at ASC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [self._row_to_investigation(row) for row in rows]

    def set_status(self, investigation_id: str, status: str) -> bool:
        allowed = {"proposed", "investigating", "resolved", "deferred", "rejected"}
        if status not in allowed:
            raise ValueError(f"Unsupported investigation status: {status}")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE investigations SET status=? WHERE id=?",
                (status, investigation_id),
            )
            return cur.rowcount > 0

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "observations": conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
                "concept_candidates": conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
                "investigations": conn.execute("SELECT COUNT(*) FROM investigations").fetchone()[0],
                "pending": conn.execute("SELECT COUNT(*) FROM investigations WHERE status='proposed'").fetchone()[0],
            }
