"""Bounded, local SQLite memory for training experiences (not raw corpus storage)."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path

class ExperienceMemory:
    def __init__(self, path, max_events=50000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events = int(max_events)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
            step INTEGER NOT NULL, domain TEXT NOT NULL, source_id TEXT NOT NULL,
            loss REAL NOT NULL, sample TEXT NOT NULL, metadata TEXT NOT NULL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_experiences_domain ON experiences(domain, id)")
        self.db.commit()

    def record_batch(self, step, loss, records, metadata=None):
        if not records:
            return
        now = time.time()
        rows = [(now, int(step), r.domain, r.source_id, float(loss), r.text[:500],
                 json.dumps(metadata or {}, ensure_ascii=False)) for r in records]
        with self.db:
            self.db.executemany("INSERT INTO experiences(created,step,domain,source_id,loss,sample,metadata) VALUES(?,?,?,?,?,?,?)", rows)
            self.db.execute("DELETE FROM experiences WHERE id NOT IN (SELECT id FROM experiences ORDER BY id DESC LIMIT ?)", (self.max_events,))

    def domain_summary(self):
        rows = self.db.execute("SELECT domain, COUNT(*), AVG(loss), MAX(id) FROM experiences GROUP BY domain ORDER BY domain").fetchall()
        return {d: {"experiences": n, "mean_loss": round(avg, 6), "last_id": last} for d,n,avg,last in rows}

    def close(self):
        self.db.close()
