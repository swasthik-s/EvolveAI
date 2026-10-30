"""Bounded multi-source streaming gatekeeper for EvolveAI.

Reads approved Hugging Face datasets in streaming mode. It never downloads or
concatenates a complete corpus. A bounded in-memory replay buffer feeds the SNN.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import deque, Counter
from dataclasses import dataclass
from typing import Iterator

import torch
from datasets import load_dataset


# Dataset cards and terms can change. Review each upstream card before enabling it.
SOURCE_REGISTRY = {
    "tinystories": {
        "path": "roneneldan/TinyStories", "config": None, "split": "train",
        "fields": ["text"], "domain": "language",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "license_note": "CDLA-Sharing-1.0; verify current card and obligations.",
    },
    "wikitext": {
        "path": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "train",
        "fields": ["text"], "domain": "encyclopedic",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext",
        "license_note": "Review dataset card and source-text attribution terms before use.",
    },
    "ag_news": {
        "path": "fancyzhx/ag_news", "config": None, "split": "train",
        "fields": ["text"], "domain": "news",
        "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
        "license_note": "Review dataset card and underlying news-content rights before use.",
    },
    "gsm8k": {
        "path": "openai/gsm8k", "config": "main", "split": "train",
        "fields": ["question", "answer"], "domain": "mathematics",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "license_note": "Review current dataset card and terms before use.",
    },
}


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class StreamRecord:
    text: str
    source_id: str
    domain: str


class StreamingGatekeeper:
    """Round-robin source reader with quality, deduplication and memory gates."""

    def __init__(
        self,
        source_ids: list[str],
        *,
        max_record_chars: int = 4000,
        train_buffer_size: int = 512,
        validation_buffer_size: int = 128,
        min_chars: int = 40,
        seed: int = 42,
        progress=None,
    ):
        if not source_ids:
            raise ValueError("Select at least one approved source.")
        unknown = sorted(set(source_ids) - set(SOURCE_REGISTRY))
        if unknown:
            raise ValueError(f"Unapproved source IDs: {unknown}")
        self.rng = random.Random(seed)
        self.max_record_chars = int(max_record_chars)
        self.min_chars = int(min_chars)
        self.train_buffer = deque(maxlen=int(train_buffer_size))
        self.validation_buffer = deque(maxlen=int(validation_buffer_size))
        self.iterators: dict[str, Iterator] = {}
        self.source_order = list(dict.fromkeys(source_ids))
        self.source_cursor = 0
        self.seen = set()
        self.seen_order = deque(maxlen=50000)
        self.stats = {sid: Counter() for sid in self.source_order}
        self.total_admitted = 0
        self.total_rejected = 0
        self.last_admitted: StreamRecord | None = None
        self.progress = progress
        self._load_streams()

    def _load_streams(self):
        failures = []
        for sid in self.source_order:
            spec = SOURCE_REGISTRY[sid]
            try:
                kwargs = {"path": spec["path"], "split": spec["split"], "streaming": True}
                if spec["config"]:
                    kwargs["name"] = spec["config"]
                stream = load_dataset(**kwargs)
                self.iterators[sid] = iter(stream)
                print(f"[GATEKEEPER] CONNECTED | {sid} | domain={spec['domain']} | {spec['url']}")
                print(f"[GATEKEEPER] TERMS     | {sid} | {spec['license_note']}")
            except Exception as exc:
                failures.append((sid, str(exc)))
                print(f"[GATEKEEPER] SOURCE_ERROR | {sid} | {type(exc).__name__}: {exc}")
        self.source_order = [sid for sid in self.source_order if sid in self.iterators]
        if not self.source_order:
            raise RuntimeError(f"No data streams could be opened: {failures}")

    def _next_raw(self):
        if not self.source_order:
            raise StopIteration
        while self.source_order:
            sid = self.source_order[self.source_cursor % len(self.source_order)]
            self.source_cursor = (self.source_cursor + 1) % len(self.source_order)
            try:
                row = next(self.iterators[sid])
                return sid, row
            except StopIteration:
                self.iterators.pop(sid, None)
                self.source_order = [x for x in self.source_order if x in self.iterators]
                if self.source_order:
                    self.source_cursor %= len(self.source_order)
        raise StopIteration

    def _extract(self, sid, row):
        spec = SOURCE_REGISTRY[sid]
        pieces = []
        for field in spec["fields"]:
            value = row.get(field, "")
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
        text = _clean("\n".join(pieces))
        if len(text) < self.min_chars:
            self.stats[sid]["rejected_short"] += 1
            self.total_rejected += 1
            return None
        if len(text) > self.max_record_chars:
            text = text[:self.max_record_chars]
            self.stats[sid]["truncated"] += 1
        alnum = sum(ch.isalnum() for ch in text)
        if alnum / max(1, len(text)) < 0.20:
            self.stats[sid]["rejected_low_signal"] += 1
            self.total_rejected += 1
            return None
        digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
        if digest in self.seen:
            self.stats[sid]["rejected_duplicate"] += 1
            self.total_rejected += 1
            return None
        if len(self.seen_order) == self.seen_order.maxlen:
            expired = self.seen_order.popleft()
            self.seen.discard(expired)
        self.seen.add(digest)
        self.seen_order.append(digest)
        return StreamRecord(text=text, source_id=sid, domain=spec["domain"])

    def ingest(self, records: int = 8):
        admitted = 0
        for _ in range(max(1, int(records))):
            try:
                sid, row = self._next_raw()
            except StopIteration:
                break
            self.stats[sid]["rows_seen"] += 1
            record = self._extract(sid, row)
            if record is None:
                continue
            # Deterministic online split: 1 in 20 admitted records goes to validation.
            bucket = self.validation_buffer if self.total_admitted % 20 == 0 else self.train_buffer
            bucket.append(record)
            self.total_admitted += 1
            self.stats[sid]["admitted"] += 1
            self.last_admitted = record
            admitted += 1
            if self.progress:
                self.progress(record, self.total_admitted, self.stats)
        return admitted

    def warm_buffers(self, min_train=128, min_validation=16, max_reads=10000):
        reads = 0
        while (len(self.train_buffer) < min_train or len(self.validation_buffer) < min_validation):
            if reads >= max_reads:
                break
            got = self.ingest(records=16)
            reads += 16
            if got == 0 and not self.iterators:
                break
        if len(self.train_buffer) < 8:
            raise RuntimeError("Gatekeeper could not collect enough usable training records.")
        if not self.validation_buffer:
            # Very small streams: reserve a few tail records as validation.
            for _ in range(min(8, len(self.train_buffer) // 4)):
                self.validation_buffer.append(self.train_buffer.pop())
        print(f"[GATEKEEPER] READY | train_buffer={len(self.train_buffer)} | validation_buffer={len(self.validation_buffer)} | records_admitted={self.total_admitted}")

    @staticmethod
    def _tokens(record: StreamRecord) -> torch.Tensor:
        return torch.tensor(list(record.text.encode("utf-8")), dtype=torch.long)

    def make_batch(self, batch_size: int, block_size: int, device: str, validation=False):
        pool = self.validation_buffer if validation else self.train_buffer
        if not pool:
            raise RuntimeError("Gatekeeper buffer is empty.")
        chosen = [self.rng.choice(list(pool)) for _ in range(batch_size)]
        xs, ys = [], []
        for record in chosen:
            tokens = self._tokens(record)
            needed = block_size + 1
            if tokens.numel() < needed:
                repeats = (needed + tokens.numel() - 1) // tokens.numel()
                tokens = tokens.repeat(repeats)
            start = self.rng.randint(0, max(0, tokens.numel() - needed))
            window = tokens[start:start + needed]
            xs.append(window[:-1])
            ys.append(window[1:])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)

    def source_summary(self):
        return {sid: dict(counts) for sid, counts in self.stats.items()}

    def status(self):
        return {
            "records_admitted": self.total_admitted,
            "records_rejected": self.total_rejected,
            "train_buffer_records": len(self.train_buffer),
            "validation_buffer_records": len(self.validation_buffer),
            "active_sources": list(self.source_order),
            "sources": self.source_summary(),
        }
