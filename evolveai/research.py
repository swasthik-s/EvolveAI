"""Free, keyless web research for EvolveAI.

This module uses public search results and ordinary HTTP page retrieval. It does
not call an LLM. Extracted text is evidence, not trusted instructions.
"""
import ipaddress
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


USER_AGENT = "EvolveAIResearch/0.1 (public-page text extraction)"
MAX_PAGE_BYTES = 1_000_000
STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "did", "doing", "have", "has", "had", "will", "would", "could",
    "should", "can", "the", "and", "for", "from", "with", "that", "this",
    "about", "into", "your", "you", "are", "was", "were", "is", "not",
}


def _terms(text):
    return {w for w in re.findall(r"[a-zA-Z0-9]{3,}", text.lower()) if w not in STOP_WORDS}


def _public_http_url(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        host = parsed.hostname.lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return False
        try:
            address = ipaddress.ip_address(host)
            if not address.is_global:
                return False
        except ValueError:
            pass
        return True
    except (TypeError, ValueError):
        return False


def _page_text(url):
    if not _public_http_url(url):
        return ""
    try:
        with requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=(3, 6),
            stream=True, allow_redirects=False,
        ) as response:
            response.raise_for_status()
            if not _public_http_url(response.url):
                return ""
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ""
            chunks = []
            size = 0
            for chunk in response.iter_content(16_384):
                size += len(chunk)
                if size > MAX_PAGE_BYTES:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
        soup = BeautifulSoup(raw, "html.parser")
        for node in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            node.decompose()
        return " ".join(soup.stripped_strings)[:30_000]
    except (requests.RequestException, ValueError, UnicodeError):
        return ""


def _best_excerpt(text, query, limit=650):
    terms = _terms(query)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    ranked = []
    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(sentence) < 35:
            continue
        words = _terms(sentence)
        overlap = len(terms & words)
        if overlap:
            ranked.append((overlap / max(1, len(terms)), overlap, sentence))
    ranked.sort(key=lambda item: (item[0], item[1], min(len(item[2]), 500)), reverse=True)
    if ranked:
        excerpt = " ".join(item[2] for item in ranked[:3])
    else:
        excerpt = text[:limit]
    return excerpt[:limit].strip()


def research(query, max_results=5):
    """Search public web results, read a few pages, and return extractive evidence."""
    query = str(query).strip()
    if not query:
        return []
    try:
        results = list(DDGS().text(query, max_results=max(1, min(int(max_results), 5))))
    except Exception:
        return []

    evidence = []
    seen_urls = set()
    for item in results:
        url = str(item.get("href") or item.get("url") or "").strip()
        if not _public_http_url(url) or url in seen_urls:
            continue
        seen_urls.add(url)
        title = re.sub(r"\s+", " ", str(item.get("title") or "Untitled page")).strip()[:240]
        snippet = re.sub(r"\s+", " ", str(item.get("body") or "")).strip()
        page = _page_text(url)
        excerpt = _best_excerpt(page, query) if page else ""
        if not excerpt:
            excerpt = snippet[:650]
        if not excerpt:
            continue
        terms = _terms(query)
        overlap = len(terms & _terms(title + " " + excerpt))
        confidence = min(0.85, 0.25 + 0.12 * overlap)
        evidence.append({
            "title": title,
            "url": url,
            "excerpt": excerpt,
            "confidence": round(confidence, 2),
            "source_domain": urlparse(url).hostname or "",
        })
    evidence.sort(key=lambda item: (item["confidence"], len(item["excerpt"])), reverse=True)
    return evidence[:5]
