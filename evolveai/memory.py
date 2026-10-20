import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class ExperienceMemory:
    """SQLite store for chat history, personal memories, lessons, and Ghost Knowledge."""

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
            conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, prompt TEXT NOT NULL, answer TEXT NOT NULL,
                mode TEXT NOT NULL, score REAL, feedback TEXT, created_at TEXT NOT NULL,
                strategy TEXT NOT NULL DEFAULT ''
            )""")
            task_columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
            if "strategy" not in task_columns:
                conn.execute("ALTER TABLE tasks ADD COLUMN strategy TEXT NOT NULL DEFAULT ''")
            conn.execute("""CREATE TABLE IF NOT EXISTS strategy_outcomes (
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, strategy TEXT NOT NULL,
                score REAL NOT NULL, feedback TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_outcomes_strategy ON strategy_outcomes(strategy)")
            conn.execute("""CREATE TABLE IF NOT EXISTS lessons (
                id TEXT PRIMARY KEY, task TEXT NOT NULL, lesson TEXT NOT NULL,
                confidence REAL NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS ghost_knowledge (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, fact TEXT NOT NULL,
                title TEXT NOT NULL, source_url TEXT NOT NULL, confidence REAL NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL, last_checked_at TEXT NOT NULL,
                UNIQUE(question, source_url)
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ghost_created ON ghost_knowledge(created_at)")

    def save_task(self, prompt, answer, mode="rules", strategy=""):
        task_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks(id,prompt,answer,mode,created_at,strategy) VALUES(?,?,?,?,?,?)",
                (task_id, prompt, answer, mode, now_iso(), strategy),
            )
        return task_id

    def rate_task(self, task_id, score, feedback=""):
        score = max(0.0, min(1.0, float(score)))
        with self._connect() as conn:
            row = conn.execute("SELECT prompt, answer, strategy FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                return False
            conn.execute("UPDATE tasks SET score=?, feedback=? WHERE id=?", (score, feedback, task_id))
            if row["strategy"]:
                conn.execute(
                    "INSERT INTO strategy_outcomes(id,task_id,strategy,score,feedback,created_at) VALUES(?,?,?,?,?,?)",
                    (uuid.uuid4().hex[:12], task_id, row["strategy"], score, str(feedback or ""), now_iso()),
                )
            if score >= 0.8:
                lesson = "A response to this prompt was marked helpful. Example response: " + row["answer"][:1200]
                self._insert_lesson(conn, row["prompt"], lesson, score, "user_feedback")
            elif score <= 0.2:
                self._insert_lesson(
                    conn, row["prompt"], "Avoid repeating the previous response pattern; user marked it unhelpful.",
                    0.7, "user_feedback",
                )
        return True

    def _insert_lesson(self, conn, task, lesson, confidence, source):
        conn.execute(
            "INSERT INTO lessons(id,task,lesson,confidence,source,created_at) VALUES(?,?,?,?,?,?)",
            (uuid.uuid4().hex[:12], task, lesson, float(confidence), source, now_iso()),
        )

    def record_strategy_outcome(self, task_id, strategy, score, feedback=""):
        """Record explicit feedback for a chosen communication strategy."""
        score = max(0.0, min(1.0, float(score)))
        strategy = str(strategy or "").strip()
        if not strategy:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                return False
            conn.execute("UPDATE tasks SET strategy=? WHERE id=?", (strategy, task_id))
            conn.execute(
                "INSERT INTO strategy_outcomes(id,task_id,strategy,score,feedback,created_at) VALUES(?,?,?,?,?,?)",
                (uuid.uuid4().hex[:12], task_id, strategy, score, str(feedback or ""), now_iso()),
            )
        return True

    def strategy_statistics(self):
        """Aggregate feedback by strategy without exposing individual conversations."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT strategy, COUNT(*) AS count, AVG(score) AS mean_score "
                "FROM strategy_outcomes GROUP BY strategy"
            ).fetchall()
        return {
            row["strategy"]: {"count": int(row["count"]), "mean_score": float(row["mean_score"])}
            for row in rows
        }

    def add_lesson(self, task, lesson, confidence=0.7, source="user"):
        confidence = max(0.0, min(1.0, float(confidence)))
        with self._connect() as conn:
            self._insert_lesson(conn, task, lesson, confidence, source)

    def search_lessons(self, query="", limit=5):
        words = {w.lower() for w in query.split() if len(w) > 2}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task, lesson, confidence, source, created_at FROM lessons "
                "ORDER BY created_at DESC LIMIT 500"
            ).fetchall()
        ranked = []
        for row in rows:
            item = dict(row)
            haystack = (item["task"] + " " + item["lesson"]).lower()
            overlap = sum(1 for word in words if word in haystack)
            if not words or overlap:
                item["_rank"] = overlap
                ranked.append(item)
        ranked.sort(key=lambda item: (item["_rank"], item["confidence"], item["created_at"]), reverse=True)
        for item in ranked:
            item.pop("_rank", None)
        return ranked[:max(1, min(int(limit), 100))]

    def remember(self, key, value, source="user"):
        key, value = key.strip().lower(), value.strip()
        if not key or not value:
            return False
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO memories(key,value,source,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                source=excluded.source, updated_at=excluded.updated_at""",
                (key, value, source, now_iso()),
            )
        return True

    def memories(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value, source, updated_at FROM memories ORDER BY key").fetchall()
        return [dict(row) for row in rows]

    def forget(self, key):
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE key=?", (key.strip().lower(),))
        return cursor.rowcount > 0

    def add_ghost_knowledge(self, question, fact, title, source_url, confidence=0.4, status="uncertain"):
        question, fact = question.strip(), fact.strip()
        title, source_url = title.strip() or "Web source", source_url.strip()
        if not question or not fact or not source_url:
            return False
        confidence = max(0.0, min(1.0, float(confidence)))
        now = now_iso()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO ghost_knowledge
                (id, question, fact, title, source_url, confidence, status, created_at, last_checked_at)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(question, source_url) DO UPDATE SET
                fact=excluded.fact, title=excluded.title, confidence=excluded.confidence,
                status=excluded.status, last_checked_at=excluded.last_checked_at""",
                (uuid.uuid4().hex[:12], question, fact, title, source_url, confidence, status, now, now),
            )
        return True

    def search_ghost_knowledge(self, query="", limit=5):
        words = {w.lower() for w in query.split() if len(w) > 2}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, question, fact, title, source_url, confidence, status, created_at, last_checked_at "
                "FROM ghost_knowledge ORDER BY last_checked_at DESC LIMIT 1000"
            ).fetchall()
        ranked = []
        for row in rows:
            item = dict(row)
            haystack = (item["question"] + " " + item["fact"] + " " + item["title"]).lower()
            overlap = sum(1 for word in words if word in haystack)
            if not words or overlap:
                item["_rank"] = overlap / max(1, len(words))
                ranked.append(item)
        ranked.sort(key=lambda item: (item["_rank"], item["confidence"], item["last_checked_at"]), reverse=True)
        for item in ranked:
            item.pop("_rank", None)
        return ranked[:max(1, min(int(limit), 100))]

    def recent_tasks(self, limit=50):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, prompt, answer, mode, score, feedback, created_at FROM tasks "
                "ORDER BY created_at DESC LIMIT ?", (max(1, min(int(limit), 500)),)
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self):
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            rated = conn.execute("SELECT COUNT(*) FROM tasks WHERE score IS NOT NULL").fetchone()[0]
            good = conn.execute("SELECT COUNT(*) FROM tasks WHERE score >= 0.8").fetchone()[0]
            memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            lessons = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
            ghost = conn.execute("SELECT COUNT(*) FROM ghost_knowledge").fetchone()[0]
        return {
            "tasks": total, "rated_tasks": rated,
            "success_rate": good / rated if rated else 0.0,
            "memories": memories, "lessons": lessons, "ghost_knowledge": ghost,
        }
