"""Bounded persistent episodic learning log for EvolveAI.

Stores provenance and learning outcomes, not executable content. The log is
append-only JSONL with a hard record cap; raw stream buffers remain in memory.
"""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


class EpisodicMemory:
    def __init__(self, path, max_events=20000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events = int(max_events)
        self.events = []
        self.domain_events = Counter()
        if self.path.exists():
            self._load()

    def _load(self):
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-self.max_events:]
            for line in lines:
                try:
                    item = json.loads(line)
                    self.events.append(item)
                    if item.get("domain"):
                        self.domain_events[item["domain"]] += 1
                except (json.JSONDecodeError, TypeError):
                    continue
        except OSError as exc:
            print(f"[MEMORY] Could not load prior memory: {exc}")
        self.events = self.events[-self.max_events:]

    def record_batch(self, step, loss, records, context=None):
        grouped = {}
        for record in records:
            grouped.setdefault(record.domain, []).append(record)
        for domain, items in grouped.items():
            event = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "step": int(step), "domain": domain, "source_ids": sorted({r.source_id for r in items}),
                "loss": float(loss), "count": len(items),
                "examples": [r.text[:240] for r in items[:2]],
                "context": context or {},
            }
            self.events.append(event)
            self.domain_events[domain] += len(items)
            self._append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
            self._rewrite()

    def _append(self, event):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _rewrite(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def domain_summary(self):
        summary = {}
        for event in self.events:
            domain = event.get("domain", "unknown")
            item = summary.setdefault(domain, {"events": 0, "examples_seen": 0, "mean_loss": 0.0})
            item["events"] += 1
            item["examples_seen"] += int(event.get("count", 0))
            item["mean_loss"] += float(event.get("loss", 0.0))
        for item in summary.values():
            item["mean_loss"] = item["mean_loss"] / max(1, item["events"])
        return summary
