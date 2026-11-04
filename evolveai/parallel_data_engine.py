"""EvolveAI Parallel Data Engine (PDE).

Runs independent async source connectors behind a bounded queue. It normalizes
records, evaluates them, deduplicates them in a bounded LRU window, persists
provenance, and hands accepted experiences to one downstream learner callback.

This is an ingestion/orchestration layer. It does not create truth labels or
claim that lexical evaluation is semantic understanding.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, AsyncIterable, Awaitable, Callable, Iterable, Protocol

from .experience_evaluator import Experience, ExperienceEvaluator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SourceRecord:
    content: str
    source_id: str
    source_uri: str = ""
    title: str = ""
    domain: str = "unknown"
    license: str = "unspecified"
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_at: str = field(default_factory=utc_now)


class SourceConnector(Protocol):
    source_id: str

    def stream(self) -> AsyncIterable[SourceRecord]:
        """Yield records incrementally; do not download the full corpus."""
        ...


class ConnectorRegistry:
    """Runtime registry; adding a source does not require editing the engine."""

    def __init__(self) -> None:
        self._connectors: dict[str, SourceConnector] = {}

    def register(self, connector: SourceConnector) -> None:
        source_id = str(connector.source_id).strip()
        if not source_id:
            raise ValueError("Connector source_id cannot be empty.")
        if source_id in self._connectors:
            raise ValueError(f"Connector already registered: {source_id}")
        self._connectors[source_id] = connector

    def get(self, source_id: str) -> SourceConnector:
        return self._connectors[source_id]

    def ids(self) -> list[str]:
        return sorted(self._connectors)

    def __len__(self) -> int:
        return len(self._connectors)


@dataclass
class EngineStats:
    sources_started: int = 0
    sources_completed: int = 0
    records_received: int = 0
    records_normalized: int = 0
    records_duplicate: int = 0
    records_rejected: int = 0
    records_persisted: int = 0
    learner_handoffs: int = 0
    source_errors: int = 0
    max_queue_depth: int = 0
    decisions: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class ExperienceStore:
    """SQLite provenance and evaluation log; bounded by optional row retention."""

    def __init__(self, db_path: str = "data/pde_experiences.sqlite3",
                 max_records: int = 250000) -> None:
        self.db_path = str(db_path)
        self.max_records = max(100, int(max_records))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=20)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS experiences (
                experience_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                title TEXT NOT NULL,
                domain TEXT NOT NULL,
                license TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_source ON experiences(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_domain ON experiences(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_decision ON experiences(created_at)")

    def save(self, record: SourceRecord, evaluation: dict[str, Any]) -> bool:
        content = str(record.content or "").strip()
        if not content:
            return False
        digest = hashlib.sha256(
            "\x1f".join((record.source_id, record.source_uri, content)).encode("utf-8")
        ).hexdigest()[:24]
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO experiences VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (digest, record.source_id, record.source_uri, record.title[:500],
                 record.domain[:120], record.license[:300], content,
                 json.dumps(record.metadata, ensure_ascii=False, default=str),
                 record.observed_at,
                 json.dumps(evaluation, ensure_ascii=False, default=str), utc_now()),
            )
            conn.execute(
                """DELETE FROM experiences WHERE experience_id IN (
                   SELECT experience_id FROM experiences ORDER BY created_at DESC
                   LIMIT -1 OFFSET ?)""",
                (self.max_records,),
            )
            return cur.rowcount == 1

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            sources = conn.execute("SELECT COUNT(DISTINCT source_id) FROM experiences").fetchone()[0]
            return {"records": total, "sources": sources}


LearnerCallback = Callable[[SourceRecord, dict[str, Any]], Awaitable[None] | None]


class ParallelDataEngine:
    """Concurrent ingestion with bounded buffering and one learner handoff path."""

    def __init__(
        self,
        registry: ConnectorRegistry,
        *,
        evaluator: ExperienceEvaluator | None = None,
        store: ExperienceStore | None = None,
        learner: LearnerCallback | None = None,
        queue_size: int = 256,
        workers: int = 2,
        max_record_chars: int = 20000,
        dedupe_window: int = 10000,
        learn_decisions: Iterable[str] = ("learn_now",),
    ) -> None:
        self.registry = registry
        self.evaluator = evaluator or ExperienceEvaluator()
        self.store = store or ExperienceStore()
        self.learner = learner
        self.queue_size = max(1, int(queue_size))
        self.workers = max(1, int(workers))
        self.max_record_chars = max(100, int(max_record_chars))
        self.dedupe_window = max(100, int(dedupe_window))
        self.learn_decisions = set(learn_decisions)
        self.stats = EngineStats()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._known_terms: set[str] = set()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._stop = object()

    def _normalize(self, record: SourceRecord) -> SourceRecord | None:
        content = " ".join(str(record.content or "").split())
        content = content[:self.max_record_chars]
        if len(content) < 20:
            return None
        return SourceRecord(
            content=content,
            source_id=str(record.source_id).strip(),
            source_uri=str(record.source_uri or "")[:2000],
            title=str(record.title or "")[:500],
            domain=str(record.domain or "unknown")[:120],
            license=str(record.license or "unspecified")[:300],
            metadata=dict(record.metadata or {}),
            observed_at=record.observed_at or utc_now(),
        )

    def _is_duplicate(self, record: SourceRecord) -> bool:
        key = hashlib.sha256(
            (record.source_id + "\x1f" + record.source_uri + "\x1f" + record.content)
            .encode("utf-8")
        ).hexdigest()
        if key in self._seen:
            self._seen.move_to_end(key)
            return True
        self._seen[key] = None
        if len(self._seen) > self.dedupe_window:
            self._seen.popitem(last=False)
        return False

    async def _produce(self, connector: SourceConnector) -> None:
        self.stats.sources_started += 1
        try:
            async for item in connector.stream():
                self.stats.records_received += 1
                if not isinstance(item, SourceRecord):
                    self.stats.records_rejected += 1
                    continue
                normalized = self._normalize(item)
                if normalized is None or not normalized.source_id:
                    self.stats.records_rejected += 1
                    continue
                self.stats.records_normalized += 1
                if self._is_duplicate(normalized):
                    self.stats.records_duplicate += 1
                    continue
                # Awaiting put applies backpressure when the learner is slower.
                await self._queue.put(normalized)
                self.stats.max_queue_depth = max(
                    self.stats.max_queue_depth, self._queue.qsize()
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.stats.source_errors += 1
        finally:
            self.stats.sources_completed += 1

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is self._stop:
                    return
                record: SourceRecord = item
                experience = Experience(
                    content=record.content, source_id=record.source_id,
                    source_uri=record.source_uri, domain=record.domain,
                    metadata=record.metadata,
                )
                evaluation = self.evaluator.evaluate(
                    experience, known_terms=self._known_terms
                )
                result = evaluation.to_dict()
                decision = evaluation.decision
                self.stats.decisions[decision] = self.stats.decisions.get(decision, 0) + 1

                saved = self.store.save(record, result)
                if saved:
                    self.stats.records_persisted += 1

                # Update lexical familiarity only after evaluation and persistence.
                if decision not in {"reject_empty", "reject_duplicate"}:
                    self._known_terms.update(
                        token.lower() for token in record.content.split()
                        if len(token) >= 4
                    )

                if self.learner is not None and decision in self.learn_decisions:
                    outcome = self.learner(record, result)
                    if asyncio.iscoroutine(outcome):
                        await outcome
                    self.stats.learner_handoffs += 1
            finally:
                self._queue.task_done()

    async def run(self, *, source_ids: Iterable[str] | None = None,
                  max_records: int | None = None) -> dict[str, Any]:
        """Run selected connectors until exhausted or max_records is received."""
        ids = list(source_ids) if source_ids is not None else self.registry.ids()
        missing = [sid for sid in ids if sid not in self.registry.ids()]
        if missing:
            raise KeyError(f"Unregistered source IDs: {missing}")
        if not ids:
            return self.status()

        producers = [
            asyncio.create_task(self._produce(self.registry.get(sid)), name=f"source:{sid}")
            for sid in ids
        ]
        consumers = [
            asyncio.create_task(self._consume(), name=f"pde-consumer:{index}")
            for index in range(self.workers)
        ]
        try:
            if max_records is None:
                await asyncio.gather(*producers)
            else:
                target = max(1, int(max_records))
                while self.stats.records_received < target and any(not p.done() for p in producers):
                    await asyncio.sleep(0.02)
                for task in producers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
            await self._queue.join()
        finally:
            for _ in consumers:
                await self._queue.put(self._stop)
            await asyncio.gather(*consumers)
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            **self.stats.snapshot(),
            "registered_sources": self.registry.ids(),
            "queue_size": self._queue.qsize(),
            "queue_capacity": self.queue_size,
            "worker_count": self.workers,
            "dedupe_window_entries": len(self._seen),
            "store": self.store.stats(),
            "backpressure": "bounded asyncio queue; producers await capacity",
        }
