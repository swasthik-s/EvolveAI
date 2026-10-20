"""Auditable activity journal for EvolveAI agent and background work.

This records observable actions and outcomes, not hidden model chain-of-thought.
Uses the same configured SQLite file as ExperienceMemory; production persistence
still requires a shared durable database backend.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


class ActivityJournal:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS brain_activity (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL,
                event TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_brain_activity_created ON brain_activity(created_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_brain_activity_task ON brain_activity(task_id)")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def record(self, category, event, status, summary, task_id="", details=None):
        allowed_status = {"started", "running", "completed", "failed", "waiting", "proposed", "rejected"}
        status = status if status in allowed_status else "proposed"
        event_id = uuid.uuid4().hex[:16]
        safe_details = details if isinstance(details, dict) else {}
        with self._connect() as db:
            db.execute(
                "INSERT INTO brain_activity(id,task_id,category,event,status,summary,details_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (event_id, str(task_id or "")[:64], str(category)[:60], str(event)[:100],
                 status, str(summary)[:1000], json.dumps(safe_details, ensure_ascii=False)[:12000], _now()),
            )
        return event_id

    def recent(self, limit=100, category="", status="", query=""):
        clauses, args = [], []
        if category:
            clauses.append("category = ?")
            args.append(category)
        if status:
            clauses.append("status = ?")
            args.append(status)
        if query.strip():
            clauses.append("(summary LIKE ? OR event LIKE ? OR task_id LIKE ?)")
            q = "%" + query.strip()[:100] + "%"
            args.extend([q, q, q])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(max(1, min(int(limit), 500)))
        with self._connect() as db:
            rows = db.execute(
                "SELECT id,task_id,category,event,status,summary,details_json,created_at "
                "FROM brain_activity" + where + " ORDER BY created_at DESC LIMIT ?", args
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["details"] = json.loads(item.pop("details_json"))
            except (TypeError, json.JSONDecodeError):
                item["details"] = {}
            result.append(item)
        return result

    def stats(self):
        with self._connect() as db:
            rows = db.execute("SELECT status, COUNT(*) AS n FROM brain_activity GROUP BY status").fetchall()
            total = db.execute("SELECT COUNT(*) FROM brain_activity").fetchone()[0]
            latest = db.execute("SELECT created_at FROM brain_activity ORDER BY created_at DESC LIMIT 1").fetchone()
        return {"events": total, "by_status": {r["status"]: r["n"] for r in rows},
                "last_event_at": latest["created_at"] if latest else None}
