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
    text = re.sub(r"\\[[0-9]+\\]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _sentences(text):
    text = _clean(text)
    parts = re.split(r"(?<=[.!?])\\s+(?=[A-Z0-9])|(?<=:)\\s+", text)
    output = []
    for part in parts:
        sentence = part.strip(" -•|")
        low = sentence.lower()
        if len(sentence) < 35 or len(sentence) > 500:
            continue
        if any(phrase in low for phrase in BOILERPLATE):
            continue
        if len(_words(sentence)) < 6:
            continue
        output.append(sentence)
    return output


def _question_focus(question):
    text = question.lower().strip()
    text = re.sub(r"^(what is|what are|who is|who are|define|explain|tell me about|can you explain)\\s+", "", text)
    text = re.sub(r"[?!.]+$", "", text)
    return text.strip()


def process_evidence(question, evidence, max_sentences=3):
    """Return a short natural answer, source list, and transparent evidence notes."""
    focus = _question_focus(question)
    query_words = _words(focus)
    if not query_words:
        query_words = _words(question)

    candidates = []
    for source_index, item in enumerate(evidence):
        source_text = " ".join([
            str(item.get("title", "")),
            str(item.get("excerpt", "")),
        ])
        for sentence in _sentences(source_text):
            words = _words(sentence)
            overlap = len(query_words & words)
            definition_bonus = 0.0
            if re.search(r"\\b(is|are|means|refers to|defined as|known as)\\b", sentence.lower()):
                definition_bonus = 1.5
            if re.search(r"\\b(article|redirects|disambiguation|synonyms of)\\b", sentence.lower()):
                definition_bonus -= 2.0
            length_penalty = max(0, len(sentence) - 240) / 180
            score = overlap / max(1, len(query_words)) * 4 + definition_bonus - length_penalty
            if overlap:
                candidates.append({
                    "sentence": sentence,
                    "score": score,
                    "source_index": source_index,
                    "url": item.get("url", ""),
                    "title": item.get("title", "Source"),
                    "domain": item.get("source_domain") or urlparse(item.get("url", "")).hostname or "",
                })

    # Prefer a clean definition, then distinct supporting details. Avoid duplicate
    # sentences copied by multiple sites.
    candidates.sort(key=lambda row: (row["score"], -len(row["sentence"])), reverse=True)
    selected = []
    seen = set()
    used_domains = set()
    for candidate in candidates:
        normalized = re.sub(r"[^a-z0-9]+", " ", candidate["sentence"].lower()).strip()
        if not normalized or normalized in seen:
            continue
        if len(selected) >= max_sentences:
            break
        # Prefer source diversity where possible, but allow another useful sentence
        # from the same source if no distinct domain has useful evidence.
        if candidate["domain"] in used_domains and len(selected) == 0:
            continue
        selected.append(candidate)
        seen.add(normalized)
        used_domains.add(candidate["domain"])

    if not selected:
        return {
            "answer": "I found web pages, but I couldn't extract enough relevant evidence to answer confidently. Could you narrow the question or give me a little more context?",
            "sources": [],
            "evidence_count": 0,
            "confidence": 0.0,
        }

    # Build a readable response rather than returning page snippets verbatim.
    first = selected[0]["sentence"]
    answer_parts = [first]
    for item in selected[1:]:
        sentence = item["sentence"]
        if sentence.lower() not in " ".join(answer_parts).lower():
            answer_parts.append(sentence)

    answer = " ".join(answer_parts)
    if len(answer) > 1100:
        answer = answer[:1097].rsplit(" ", 1)[0] + "..."

    sources = []
    seen_urls = set()
    for item in selected:
        if item["url"] and item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            sources.append({"title": item["title"], "url": item["url"], "domain": item["domain"]})

    distinct_domains = {source["domain"] for source in sources if source["domain"]}
    confidence = min(0.9, 0.25 + 0.12 * len(selected) + 0.08 * max(0, len(distinct_domains) - 1))
    if len(distinct_domains) < 2:
        confidence = min(confidence, 0.55)

    return {
        "answer": answer,
        "sources": sources,
        "evidence_count": len(selected),
        "confidence": round(confidence, 2),
    }
