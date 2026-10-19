import json
import re
import uuid
from pathlib import Path
from .memory import now_iso


class SkillLibrary:
    """Stores candidate skills as reviewable JSON files; never executes them."""
    def __init__(self, path="data/skills"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def create_draft(self, name, description, content, source="agent"):
        skill_id = uuid.uuid4().hex[:12]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-")[:60] or "skill"
        item = {"id": skill_id, "name": safe_name, "description": description.strip(), "content": content, "source": source, "status": "draft", "created_at": now_iso()}
        (self.path / f"{skill_id}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        return item

    def list_all(self):
        items = []
        for file in self.path.glob("*.json"):
            try:
                items.append(json.loads(file.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

    def set_status(self, skill_id, status):
        if status not in {"draft", "approved", "rejected"}:
            raise ValueError("Unsupported skill status")
        path = self.path / f"{skill_id}.json"
        if not path.exists():
            return False
        item = json.loads(path.read_text(encoding="utf-8"))
        item["status"] = status
        item["reviewed_at"] = now_iso()
        path.write_text(json.dumps(item, indent=2), encoding="utf-8")
        return True

    def count_approved(self):
        return sum(1 for item in self.list_all() if item.get("status") == "approved")
