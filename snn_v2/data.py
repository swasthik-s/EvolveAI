"""Multi-source public dataset streaming for SNN V2.

Sources are allowlisted and read in streaming mode. Review each upstream card,
license, and underlying content rights before enabling commercial use.
"""
from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import re
from typing import Any


SOURCE_REGISTRY = {
    "tinystories": {
        "path": "roneneldan/TinyStories", "config": None, "split": "train",
        "fields": ["text"], "domain": "stories",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "license_note": "Review the current dataset card and redistribution terms.",
    },
    "wikitext": {
        "path": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "train",
        "fields": ["text"], "domain": "encyclopedic",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext",
        "license_note": "Review source-text attribution and dataset terms.",
    },
    "ag_news": {
        "path": "fancyzhx/ag_news", "config": None, "split": "train",
        "fields": ["text"], "domain": "news",
        "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
        "license_note": "Review underlying publisher content rights before use.",
    },
    "gsm8k": {
        "path": "openai/gsm8k", "config": "main", "split": "train",
        "fields": ["question", "answer"], "domain": "mathematics",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "license_note": "Review the current dataset card and terms.",
    },
}


@dataclass(frozen=True)
class PublicRecord:
    text: str
    source_id: str
    domain: str


def clean_text(value: str) -> str:
    value = str(value or "").replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def iter_public_records(source_ids: list[str], *, max_records: int = 10000,
                        max_record_chars: int = 4096, min_chars: int = 40) -> Iterator[PublicRecord]:
    """Round-robin approved Hugging Face streams with per-run deduplication."""
    if not source_ids:
        raise ValueError("Select at least one public source")
    unknown = sorted(set(source_ids) - set(SOURCE_REGISTRY))
    if unknown:
        raise ValueError(f"Unapproved sources: {unknown}")
    if max_records < 1 or max_record_chars < min_chars or min_chars < 1:
        raise ValueError("Invalid record limits")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to stream public corpora: pip install datasets") from exc

    streams: dict[str, Any] = {}
    for source_id in dict.fromkeys(source_ids):
        spec = SOURCE_REGISTRY[source_id]
        kwargs = {"path": spec["path"], "split": spec["split"], "streaming": True}
        if spec["config"]:
            kwargs["name"] = spec["config"]
        streams[source_id] = iter(load_dataset(**kwargs))

    seen = set()
    emitted = 0
    active = list(streams)
    while active and emitted < max_records:
        for source_id in list(active):
            try:
                row = next(streams[source_id])
            except StopIteration:
                active.remove(source_id)
                continue
            spec = SOURCE_REGISTRY[source_id]
            parts = [clean_text(row.get(field, "")) for field in spec["fields"]]
            text = "\n".join(part for part in parts if part)[:max_record_chars]
            if len(text) < min_chars:
                continue
            digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            emitted += 1
            yield PublicRecord(text=text, source_id=source_id, domain=spec["domain"])
            if emitted >= max_records:
                break
