"""Cognitive harness connecting observations, curiosity, investigation and revision.

The harness orchestrates registered read-only investigation tools. It does not
invent evidence, execute generated code, or claim that lexical signals are
semantic understanding. All external investigation is dependency-injected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable

from .curiosity import CuriosityEngine, Investigation, Observation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Evidence:
    content: str
    source_id: str
    source_uri: str = ""
    title: str = ""
    observed_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationResult:
    success: bool
    evidence: list[Evidence] = field(default_factory=list)
    explanation: str = ""
    unresolved: list[str] = field(default_factory=list)


Investigator = Callable[[Investigation], InvestigationResult]


class CognitiveHarness:
    """Bounded observe -> discover -> prioritize -> investigate -> revise loop.

    The default investigator is deliberately absent: the harness will queue
    investigations until a permitted connector is explicitly supplied.
    """

    def __init__(
        self,
        curiosity: CuriosityEngine,
        *,
        db_path: str = "data/cognitive_harness.db",
        investigator: Investigator | None = None,
        max_investigations_per_cycle: int = 5,
        max_evidence_per_investigation: int = 10,
    ) -> None:
        self.curiosity = curiosity
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.investigator = investigator
        self.max_investigations_per_cycle = max(1, min(int(max_investigations_per_cycle), 50))
        self.max_evidence_per_investigation = max(1, min(int(max_evidence_per_investigation), 50))
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS cognitive_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                investigation_id TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS investigation_outcomes (
                investigation_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                explanation TEXT NOT NULL,
                unresolved_json TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_events_created ON cognitive_events(created_at)")

    def _event(self, event_type: str, detail: str, investigation_id: str = "", **payload: Any):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cognitive_events(event_type,investigation_id,detail,payload_json,created_at) VALUES(?,?,?,?,?)",
                (event_type, investigation_id, detail, json.dumps(payload, ensure_ascii=False, default=str), utc_now()),
            )

    def observe(self, observation: Observation) -> list[Investigation]:
        """Store an observation and derive new investigation candidates."""
        candidates = self.curiosity.observe(observation)
        self._event(
            "observation_received",
            "Observation processed; candidate generation is heuristic.",
            source_id=observation.source_id,
            source_uri=observation.source_uri,
            observation_id=observation.observation_id,
            candidate_count=len(candidates),
        )
        for candidate in candidates:
            self._event(
                "investigation_proposed",
                candidate.rationale,
                candidate.investigation_id,
                gap_type=candidate.gap_type,
                target=candidate.target,
                question=candidate.question,
                evidence_ids=list(candidate.evidence_ids),
                priority=candidate.priority,
            )
        return candidates

    def prioritize(self, limit: int | None = None) -> list[Investigation]:
        """Return bounded proposed investigations in priority order."""
        requested = self.max_investigations_per_cycle if limit is None else int(limit)
        return self.curiosity.pending(max(1, min(requested, self.max_investigations_per_cycle)))

    def investigate(self, investigation: Investigation) -> dict[str, Any]:
        """Run one investigation using an explicitly registered read-only callback."""
        if self.investigator is None:
            self._event(
                "investigation_deferred",
                "No investigator connector is configured; no external lookup was attempted.",
                investigation.investigation_id,
            )
            return {
                "investigation_id": investigation.investigation_id,
                "status": "deferred",
                "reason": "No investigator connector configured.",
                "evidence": [],
            }

        self.curiosity.set_status(investigation.investigation_id, "investigating")
        self._event("investigation_started", investigation.question, investigation.investigation_id)
        try:
            result = self.investigator(investigation)
            if not isinstance(result, InvestigationResult):
                raise TypeError("Investigator must return InvestigationResult.")
            evidence = result.evidence[:self.max_evidence_per_investigation]
            saved = self._save_evidence(investigation.investigation_id, evidence)
            status = "resolved" if result.success and saved else "deferred"
            self.curiosity.set_status(investigation.investigation_id, status)
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO investigation_outcomes VALUES(?,?,?,?,?,?)
                       ON CONFLICT(investigation_id) DO UPDATE SET
                       status=excluded.status, explanation=excluded.explanation,
                       unresolved_json=excluded.unresolved_json,
                       evidence_count=excluded.evidence_count, updated_at=excluded.updated_at""",
                    (investigation.investigation_id, status, result.explanation,
                     json.dumps(result.unresolved, ensure_ascii=False), len(saved), utc_now()),
                )
            self._event(
                "investigation_completed" if status == "resolved" else "investigation_unresolved",
                result.explanation or "Investigation did not establish a resolution.",
                investigation.investigation_id,
                evidence_count=len(saved),
                unresolved=result.unresolved,
                success=result.success,
            )
            # New evidence is retained and may itself create further curiosity.
            for item in saved:
                self.observe(Observation(
                    content=item.content,
                    source_id=item.source_id,
                    source_uri=item.source_uri,
                    observed_at=item.observed_at,
                    metadata={**item.metadata, "investigation_id": investigation.investigation_id,
                              "title": item.title},
                ))
            return {
                "investigation_id": investigation.investigation_id,
                "status": status,
                "explanation": result.explanation,
                "unresolved": result.unresolved,
                "evidence": [item.to_dict() for item in saved],
            }
        except Exception as exc:
            self.curiosity.set_status(investigation.investigation_id, "deferred")
            self._event(
                "investigation_error",
                f"{type(exc).__name__}: {exc}",
                investigation.investigation_id,
            )
            return {
                "investigation_id": investigation.investigation_id,
                "status": "deferred",
                "reason": f"{type(exc).__name__}: {exc}",
                "evidence": [],
            }

    def _save_evidence(self, investigation_id: str, items: Iterable[Evidence]) -> list[Evidence]:
        import hashlib
        saved: list[Evidence] = []
        with self._connect() as conn:
            for item in items:
                content = str(item.content or "").strip()[:20000]
                if not content or not item.source_id:
                    continue
                key = "\x1f".join((investigation_id, item.source_id, item.source_uri, content))
                evidence_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
                conn.execute(
                    """INSERT OR IGNORE INTO evidence VALUES(?,?,?,?,?,?,?,?,?)""",
                    (evidence_id, investigation_id, item.source_id, item.source_uri,
                     str(item.title or "")[:300], content,
                     json.dumps(item.metadata, ensure_ascii=False, default=str),
                     item.observed_at, utc_now()),
                )
                row = conn.execute("SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
                if row:
                    saved.append(Evidence(
                        content=row["content"], source_id=row["source_id"],
                        source_uri=row["source_uri"], title=row["title"],
                        observed_at=row["observed_at"],
                        metadata=json.loads(row["metadata_json"]),
                    ))
        return saved

    def run_cycle(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Process a bounded number of investigations; never runs indefinitely."""
        selected = self.prioritize(limit)
        self._event("cycle_started", "Bounded cognitive cycle started.", selected_count=len(selected))
        results = [self.investigate(item) for item in selected]
        self._event(
            "cycle_completed",
            "Bounded cognitive cycle completed.",
            resolved=sum(item["status"] == "resolved" for item in results),
            deferred=sum(item["status"] == "deferred" for item in results),
        )
        return results

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 1000))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cognitive_events ORDER BY id DESC LIMIT ?", (bounded,)
            ).fetchall()
        return [{
            "id": row["id"], "event_type": row["event_type"],
            "investigation_id": row["investigation_id"], "detail": row["detail"],
            "payload": json.loads(row["payload_json"]), "created_at": row["created_at"],
        } for row in rows]

    def evidence_for(self, investigation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence WHERE investigation_id=? ORDER BY created_at",
                (investigation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "events": conn.execute("SELECT COUNT(*) FROM cognitive_events").fetchone()[0],
                "evidence_records": conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0],
                "investigations": conn.execute("SELECT COUNT(*) FROM investigation_outcomes").fetchone()[0],
            }
