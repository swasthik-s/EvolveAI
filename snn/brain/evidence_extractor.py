"""Evidence-backed extraction from Brain v3.1 documents.

This module emits reviewable candidates, not verified facts. Every item retains
its exact source sentence, document ID, source IDs, and extraction rule.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN = r"[A-Za-z][A-Za-z0-9'-]*"
CAPITALIZED_PHRASE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9'-]*)(?:\s+(?:[A-Z][a-zA-Z0-9'-]*|of|the|and|for)){1,5}\b")
ACRONYM = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b")
GENERIC = {
    "once", "time", "little", "boy", "girl", "she", "he", "they", "her",
    "his", "mom", "dad", "went", "loved", "liked", "asked", "wanted",
    "play", "park", "story", "said", "saw", "big", "small", "new", "old",
    "first", "last", "one", "two", "three", "today", "yesterday",
}

# Only explicit linguistic patterns produce relation candidates.
RELATION_PATTERNS = [
    ("is_a", re.compile(rf"\b({TOKEN}(?:\s+{TOKEN}){{0,3}})\s+(?:is|are)\s+(?:a|an|the)\s+({TOKEN}(?:\s+{TOKEN}){{0,3}})", re.I)),
    ("part_of", re.compile(rf"\b({TOKEN}(?:\s+{TOKEN}){{0,3}})\s+(?:is|are)\s+part\s+of\s+({TOKEN}(?:\s+{TOKEN}){{0,3}})", re.I)),
    ("causes", re.compile(rf"\b({TOKEN}(?:\s+{TOKEN}){{0,3}})\s+(?:causes?|leads\s+to)\s+({TOKEN}(?:\s+{TOKEN}){{0,3}})", re.I)),
    ("located_in", re.compile(rf"\b({TOKEN}(?:\s+{TOKEN}){{0,3}})\s+(?:is|are)\s+(?:located|found)\s+in\s+({TOKEN}(?:\s+{TOKEN}){{0,3}})", re.I)),
    ("used_for", re.compile(rf"\b({TOKEN}(?:\s+{TOKEN}){{0,3}})\s+(?:is|are)\s+used\s+for\s+({TOKEN}(?:\s+{TOKEN}){{0,3}})", re.I)),
    ("consists_of", re.compile(rf"\b({TOKEN}(?:\s+{TOKEN}){{0,3}})\s+(?:consists?\s+of|comprises?)\s+({TOKEN}(?:\s+{TOKEN}){{0,3}})", re.I)),
]


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _clean_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t,.;:!?()[]{}")


def extract_from_sentence(sentence: str):
    # Preserve terminal punctuation because this string is stored as source evidence.
    sentence = re.sub(r"\s+", " ", sentence).strip()
    entities = set()
    for pattern in (CAPITALIZED_PHRASE, ACRONYM):
        for match in pattern.findall(sentence):
            phrase = _clean_phrase(match)
            words = phrase.lower().split()
            if phrase and not all(w in GENERIC for w in words) and not (len(words) == 1 and words[0] in GENERIC):
                entities.add(phrase)

    relations = []
    for relation, pattern in RELATION_PATTERNS:
        for match in pattern.finditer(sentence):
            subject, obj = (_clean_phrase(x) for x in match.groups())
            if subject.lower() == obj.lower():
                continue
            relations.append({
                "subject": subject,
                "relation_candidate": relation,
                "object": obj,
                "status": "unverified_candidate",
                "evidence_sentence": sentence,
                "extraction_rule": relation,
            })
    return sorted(entities), relations


def run_extraction(memory_directory, output_directory=None):
    source_dir = Path(memory_directory)
    documents_path = source_dir / "semantic_documents.jsonl"
    if not documents_path.is_file():
        raise FileNotFoundError(f"Missing v3.1 document store: {documents_path}")
    output_dir = Path(output_directory) if output_directory else source_dir / "v3_2_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_records = {}
    relation_records = {}
    domain_counts = Counter()
    document_count = sentence_count = 0

    with documents_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            doc = json.loads(line)
            document_count += 1
            domain = doc.get("domain", "unknown")
            domain_counts[domain] += 1
            text = str(doc.get("text", "")).strip()
            sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
            if text and not sentences:
                sentences = [text]
            for sentence in sentences:
                sentence_count += 1
                entities, relations = extract_from_sentence(sentence)
                common = {
                    "document_id": doc.get("document_id"),
                    "source_ids": doc.get("source_ids", []),
                    "domain": domain,
                    "evidence_sentence": sentence,
                    "status": "unverified_candidate",
                }
                for entity in entities:
                    eid = _key(entity.lower(), str(common["document_id"]), sentence)
                    entity_records.setdefault(eid, {
                        "candidate": entity,
                        **common,
                        "extraction_rule": "capitalized_phrase_or_acronym",
                    })
                for relation in relations:
                    rid = _key(relation["subject"].lower(), relation["relation_candidate"],
                               relation["object"].lower(), str(common["document_id"]), sentence)
                    relation_records.setdefault(rid, {**relation, **common})

    entities_path = output_dir / "candidate_entities.jsonl"
    relations_path = output_dir / "candidate_relations.jsonl"
    for path, records in ((entities_path, entity_records), (relations_path, relation_records)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            for record in records.values():
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(path)

    report = {
        "schema_version": 1,
        "status": "candidate_extraction_complete_unverified",
        "source_documents": document_count,
        "sentences_examined": sentence_count,
        "candidate_entities": len(entity_records),
        "candidate_relations": len(relation_records),
        "documents_by_domain": dict(domain_counts),
        "entity_file": str(entities_path),
        "relation_file": str(relations_path),
        "policy": "Every candidate retains source evidence; extraction is heuristic and does not establish truth.",
    }
    report_path = output_dir / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract evidence-backed candidate concepts from Brain v3.1.")
    parser.add_argument("memory_directory")
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    print(json.dumps(run_extraction(args.memory_directory, args.output_directory), indent=2))
