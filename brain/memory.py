"""Persistent episodic memory for EvolveAI training decisions."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

class EpisodicMemory:
    """Stores compact training events, not raw corpora or executable content."""
    def __init__(self, path, max_events=50000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events = max(100, int(max_events))
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL, domain TEXT NOT NULL, source_id TEXT NOT NULL,
                loss REAL NOT NULL, preview TEXT NOT NULL, metadata TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS events_domain_idx ON events(domain)")

    def _connect(self):
        return sqlite3.connect(str(self.path), timeout=30)

    def record_batch(self, step, loss, records, metadata=None):
        meta = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        rows = [(int(step), r.domain, r.source_id, float(loss), r.text[:240], meta) for r in records]
        if not rows:
            return
        with self._connect() as db:
            db.executemany("INSERT INTO events(step,domain,source_id,loss,preview,metadata) VALUES(?,?,?,?,?,?)", rows)
            db.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT ?)", (self.max_events,))

    def domain_summary(self):
        with self._connect() as db:
            rows = db.execute("SELECT domain, COUNT(*), AVG(loss), MIN(loss), MAX(step) FROM events GROUP BY domain").fetchall()
        return {d: {"events": int(n), "mean_loss": round(float(avg), 6),
                    "best_observed_loss": round(float(best), 6), "last_step": int(last)}
                for d, n, avg, best, last in rows}
