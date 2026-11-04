"""Dependency-free RSS/Atom connector using bounded HTTP reads."""
from __future__ import annotations

import asyncio
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from typing import AsyncIterable

from ..parallel_data_engine import SourceRecord


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ET.Element, names: set[str]) -> str:
    for child in element.iter():
        if child is element:
            continue
        if _tag(child) in names and child.text:
            value = child.text.strip()
            if value:
                return value
    return ""


class RSSFeedConnector:
    def __init__(
        self,
        source_id: str,
        feed_url: str,
        *,
        domain: str = "news",
        license_note: str = "publisher terms apply; verify before reuse",
        timeout: int = 20,
        max_bytes: int = 5_000_000,
    ) -> None:
        self.source_id = source_id
        self.feed_url = feed_url
        self.domain = domain
        self.license_note = license_note
        self.timeout = max(3, int(timeout))
        self.max_bytes = max(1024, int(max_bytes))

    def _fetch(self) -> bytes:
        request = Request(
            self.feed_url,
            headers={"User-Agent": "EvolveAI-ResearchBot/0.1 (+source attribution)"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = response.read(self.max_bytes + 1)
        if len(payload) > self.max_bytes:
            raise ValueError("RSS feed exceeds configured maximum response size.")
        return payload

    async def stream(self) -> AsyncIterable[SourceRecord]:
        payload = await asyncio.to_thread(self._fetch)
        root = ET.fromstring(payload)
        for item in root.iter():
            if _tag(item) not in {"item", "entry"}:
                continue
            title = _child_text(item, {"title"})
            summary = _child_text(item, {"description", "summary", "content", "encoded"})
            link = _child_text(item, {"link", "id"})
            if not summary:
                summary = title
            content = (title + "\n" + summary).strip()
            if len(content) < 20:
                continue
            yield SourceRecord(
                content=content,
                source_id=self.source_id,
                source_uri=link or self.feed_url,
                title=title,
                domain=self.domain,
                license=self.license_note,
                metadata={
                    "connector": "rss_atom",
                    "feed_url": self.feed_url,
                    "published": _child_text(item, {"pubdate", "published", "updated"}),
                },
            )
