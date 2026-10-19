import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class ExperienceMemory:
    """SQLite storage for tasks, ratings, and lessons."""
    def __init__(self, path="data/evolveai.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, prompt TEXT NOT NULL, answer TEXT NOT NULL, mode TEXT NOT NULL, score REAL, feedback TEXT, created_at TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS lessons (id TEXT PRIMARY KEY, task TEXT NOT NULL, lesson TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at)")

    def save_task(self, prompt, answer, mode):
        task_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute("INSERT INTO tasks(id,prompt,answer,mode,created_at) VALUES(?,?,?,?,?)", (task_id, prompt, answer, mode, now_iso()))
        return task_id

    def rate_task(self, task_id, score, feedback=""):
        score = max(0.0, min(1.0, float(score)))
        with self._connect() as conn:
            row = conn.execute("SELECT prompt FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                return False
            conn.execute("UPDATE tasks SET score=?, feedback=? WHERE id=?", (score, feedback, task_id))
            if score >= 0.8:
                self._insert_lesson(conn, row["prompt"], "This task received positive feedback: " + row["prompt"], score, "user_feedback")
        return True

    def _insert_lesson(self, conn, task, lesson, confidence, source):
        conn.execute("INSERT INTO lessons(id,task,lesson,confidence,source,created_at) VALUES(?,?,?,?,?,?)", (uuid.uuid4().hex[:12], task, lesson, float(confidence), source, now_iso()))

    def add_lesson(self, task, lesson, confidence=0.7, source="agent"):
        confidence = max(0.0, min(1.0, float(confidence)))
        with self._connect() as conn:
            self._insert_lesson(conn, task, lesson, confidence, source)

    def search_lessons(self, query="", limit=10):
        words = {w.lower() for w in query.split() if len(w) > 2}
        with self._connect() as conn:
            rows = conn.execute("SELECT task, lesson, confidence, source, created_at FROM lessons ORDER BY confidence DESC, created_at DESC LIMIT 500").fetchall()
        ranked = []
        for row in rows:
            item = dict(row)
            text = (item["task"] + " " + item["lesson"]).lower()
            overlap = sum(1 for word in words if word in text)
            if not words or overlap:
                item["_rank"] = overlap
                ranked.append(item)
        ranked.sort(key=lambda x: (x["_rank"], x["confidence"]), reverse=True)
        for item in ranked:
            item.pop("_rank", None)
        return ranked[:max(1, min(int(limit), 100))]

    def recent_tasks(self, limit=50):
        with self._connect() as conn:
            rows = conn.execute("SELECT id, prompt, answer, mode, score, feedback, created_at FROM tasks ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
        return [dict(row) for row in rows]

    def stats(self):
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            rated = conn.execute("SELECT COUNT(*) FROM tasks WHERE score IS NOT NULL").fetchone()[0]
            good = conn.execute("SELECT COUNT(*) FROM tasks WHERE score >= 0.8").fetchone()[0]
        return {"tasks": total, "rated_tasks": rated, "success_rate": good / rated if rated else 0.0}
