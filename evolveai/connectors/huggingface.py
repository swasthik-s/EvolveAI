"""Hugging Face streaming connector; reads incrementally, never materializes a corpus."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterable, Iterable

from ..parallel_data_engine import SourceRecord


class HuggingFaceStreamConnector:
    def __init__(
        self,
        source_id: str,
        dataset_path: str,
        *,
        config: str | None = None,
        split: str = "train",
        fields: Iterable[str] = ("text",),
        domain: str = "unknown",
        license_note: str = "verify dataset card and applicable terms before use",
        revision: str | None = None,
        max_records: int | None = None,
    ) -> None:
        self.source_id = source_id
        self.dataset_path = dataset_path
        self.config = config
        self.split = split
        self.fields = tuple(fields)
        self.domain = domain
        self.license_note = license_note
        self.revision = revision
        self.max_records = max_records

    @staticmethod
    def _next(iterator):
        try:
            return True, next(iterator)
        except StopIteration:
            return False, None

    async def stream(self) -> AsyncIterable[SourceRecord]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional 'datasets' package to use Hugging Face connectors."
            ) from exc

        kwargs: dict[str, Any] = {
            "path": self.dataset_path,
            "split": self.split,
            "streaming": True,
        }
        if self.config:
            kwargs["name"] = self.config
        if self.revision:
            kwargs["revision"] = self.revision

        dataset = await asyncio.to_thread(load_dataset, **kwargs)
        iterator = iter(dataset)
        emitted = 0
        while self.max_records is None or emitted < self.max_records:
            available, row = await asyncio.to_thread(self._next, iterator)
            if not available:
                break
            pieces = []
            for field in self.fields:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
            content = "\n".join(pieces).strip()
            if not content:
                continue
            uri = str(row.get("url") or row.get("source") or "")
            title = str(row.get("title") or "")
            yield SourceRecord(
                content=content,
                source_id=self.source_id,
                source_uri=uri,
                title=title,
                domain=self.domain,
                license=self.license_note,
                metadata={
                    "connector": "huggingface_stream",
                    "dataset_path": self.dataset_path,
                    "config": self.config,
                    "split": self.split,
                    "revision": self.revision,
                },
            )
            emitted += 1
