"""Deterministic evidence processing for EvolveAI.

This is not an LLM. It ranks, filters, deduplicates, and synthesizes retrieved
sentences into a concise answer instead of dumping raw page excerpts.
"""
import re
from urllib.parse import urlparse


BOILERPLATE = (
    "jump to content", "jump to navigation", "from the free encyclopedia",
    "for other uses", "cite this entry", "log in", "save word",
    "show more", "contents", "related articles", "edit source",
)


def _words(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower()))


def _clean(text):
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for phrase in BOILERPLATE:
        text = re.sub(re.escape(phrase), " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text):
    text = _clean(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|(?<=:)\s+", text)
    output = []
    for part in parts:
        sentence = part.strip(" -•|")
        low = sentence.lower()
        if len(sentence) < 35 or len(sentence) > 500:
            continue
        if any(phrase in low for phrase in ("redirects here", "disambiguation", "synonyms of")):
            continue
        if len(_words(sentence)) < 6:
            continue
        output.append(sentence)
    return output


def _question_focus(question):
    text = question.lower().strip()
    text = re.sub(r"^(what is|what are|who is|who are|define|explain|tell me about|can you explain)\s+", "", text)
    return re.sub(r"[?!.]+$", "", text).strip()


def process_evidence(question, evidence, max_sentences=3):
    """Return a short answer and source list from retrieved public evidence."""
    focus = _question_focus(question)
    query_words = _words(focus) or _words(question)
    candidates = []

    for item in evidence:
        source_text = str(item.get("excerpt", ""))
        for sentence in _sentences(source_text):
            words = _words(sentence)
            overlap = len(query_words & words)
            if not overlap:
                continue
            definition_bonus = 1.5 if re.search(
                r"\b(is|are|means|refers to|defined as|known as)\b", sentence.lower()
            ) else 0.0
            if re.search(r"\b(article|redirects|disambiguation|synonyms of)\b", sentence.lower()):
                definition_bonus -= 2.0
            score = overlap / max(1, len(query_words)) * 4 + definition_bonus
            score -= max(0, len(sentence) - 240) / 180
            candidates.append({
                "sentence": sentence, "score": score,
                "url": item.get("url", ""), "title": item.get("title", "Source"),
                "domain": item.get("source_domain") or urlparse(item.get("url", "")).hostname or "",
            })

    candidates.sort(key=lambda row: (row["score"], -len(row["sentence"])), reverse=True)
    selected, seen, used_domains = [], set(), set()
    for candidate in candidates:
        normalized = re.sub(r"[^a-z0-9]+", " ", candidate["sentence"].lower()).strip()
        if not normalized or normalized in seen:
            continue
        if len(selected) >= max_sentences:
            break
        selected.append(candidate)
        seen.add(normalized)
        used_domains.add(candidate["domain"])

    if not selected:
        return {
            "answer": "I found web pages, but couldn't extract enough relevant evidence to answer confidently. Could you narrow the question or give me a little more context?",
            "sources": [], "evidence_count": 0, "confidence": 0.0,
        }

    answer = " ".join(item["sentence"] for item in selected)
    if len(answer) > 1100:
        answer = answer[:1097].rsplit(" ", 1)[0] + "..."

    sources, seen_urls = [], set()
    for item in selected:
        if item["url"] and item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            sources.append({"title": item["title"], "url": item["url"], "domain": item["domain"]})

    domains = {source["domain"] for source in sources if source["domain"]}
    confidence = min(0.9, 0.25 + 0.12 * len(selected) + 0.08 * max(0, len(domains) - 1))
    if len(domains) < 2:
        confidence = min(confidence, 0.55)

    return {
        "answer": answer, "sources": sources,
        "evidence_count": len(selected), "confidence": round(confidence, 2),
    }
