"""Evidence-backed candidate extraction from Brain v3.1 documents.

Heuristic only: outputs are candidates, never verified facts. Every record keeps
its source document, source IDs, domain, and exact evidence sentence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SENTENCE_RE = re.compile(r"(?<=[.!?])\\s+")
TOKEN = r"[A-Za-z][A-Za-z0-9'-]{1,}"
CAPITALIZED_PHRASE = re.compile(
    r"\\b[A-Z][a-zA-Z0-9'-]*(?:\\s+(?:[A-Z][a-zA-Z0-9'-]*|of|the|and|for)){0,4}\\b"
)
ACRONYM = re.compile(r"\\b[A-Z]{2,}(?:-[A-Z0-9]+)*\\b")
GENERIC = {
    "once", "time", "little", "boy", "girl", "she", "he", "they", "her",
    "his", "mom", "dad", "went", "loved", "liked", "asked", "wanted",
    "play", "park", "story", "said", "saw", "big", "small", "new", "old",
    "first", "last", "one", "two", "three", "today", "yesterday", "it",
    "the", "and", "or", "but", "of", "for", "to", "in", "on", "at", "by",
    "there", "what", "when", "where", "who", "why", "how", "if", "then",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}
BAD_PHRASE_ENDINGS = {"and", "or", "of", "the", "for", "to", "in", "on", "at", "by"}
BAD_SUBJECT_STARTS = {"there", "what", "when", "where", "who", "why", "how", "if", "then", "and", "but", "or"}
BAD_OBJECT_ENDINGS = {"a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "at", "by", "with"}

# Conservative, bounded patterns. Specific relations precede the generic is_a rule.
NP = rf"{TOKEN}(?:\\s+{TOKEN}){{0,2}}"
SUBJECT = rf"(?P<subject>[A-Z][A-Za-z0-9'-]*(?:\\s+{TOKEN}){{0,2}})"
RELATION_PATTERNS = [
    ("part_of", re.compile(rf"\\b{SUBJECT}\\s+(?:is|are)\\s+part\\s+of\\s+(?P<object>{NP})")),
    ("causes", re.compile(rf"\\b{SUBJECT}\\s+(?:causes?|leads\\s+to)\\s+(?P<object>{NP})")),
    ("located_in", re.compile(rf"\\b{SUBJECT}\\s+(?:is|are)\\s+(?:located|found)\\s+in\\s+(?P<object>{NP})")),
    ("used_for", re.compile(rf"\\b{SUBJECT}\\s+(?:is|are)\\s+used\\s+for\\s+(?P<object>{NP})")),
    ("consists_of", re.compile(rf"\\b{SUBJECT}\\s+(?:consists?\\s+of|comprises?)\\s+(?P<object>{NP})")),
    ("is_a", re.compile(rf"\\b{SUBJECT}\\s+(?:is|are)\\s+(?:a|an)\\s+(?P<object>{NP})")),
]


def _key(*parts: str) -> str:
    return hashlib.sha256("\\0".join(parts).encode("utf-8")).hexdigest()


def _clean_phrase(value: str) -> str:
    return re.sub(r"\\s+", " ", value).strip(" \\t,.;:!?()[]{}")


def _valid_entity_phrase(phrase: str) -> bool:
    words = phrase.lower().split()
    return bool(words) and words[0] not in GENERIC and words[-1] not in BAD_PHRASE_ENDINGS


def _valid_relation(subject: str, obj: str, sentence: str) -> bool:
    s_words, o_words = subject.lower().split(), obj.lower().split()
    # Do not turn existential/question fragments into subject-predicate facts.
    if not s_words or s_words[0] in BAD_SUBJECT_STARTS:
        return False
    if not o_words or o_words[-1] in BAD_OBJECT_ENDINGS:
        return False
    if len(s_words) > 3 or len(o_words) > 3:
        return False
    # The episodic samples are often truncated at a fixed character limit.
    # Only accept complete sentences for relation candidates.
    if not sentence.rstrip().endswith((".", "?", "!")):
        return False
    return True


def extract_from_sentence(sentence: str):
    # Preserve punctuation: evidence must be byte-for-byte faithful apart from whitespace.
    sentence = re.sub(r"\\s+", " ", sentence).strip()
    entities = set()
    for pattern in (CAPITALIZED_PHRASE, ACRONYM):
        for match in pattern.findall(sentence):
            phrase = _clean_phrase(match)
            if _valid_entity_phrase(phrase) and phrase.lower() not in GENERIC:
                entities.add(phrase)

    relations = []
    for relation, pattern in RELATION_PATTERNS:
        for match in pattern.finditer(sentence):
            subject = _clean_phrase(match.group("subject"))
            obj = _clean_phrase(match.group("object"))
            if subject.lower() == obj.lower() or not _valid_relation(subject, obj, sentence):
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

    entity_records, relation_records = {}, {}
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
                        "candidate": entity, **common,
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
                stream.write(json.dumps(record, ensure_ascii=False) + "\\n")
        tmp.replace(path)

    report = {
        "schema_version": 2,
        "status": "candidate_extraction_complete_unverified",
        "source_documents": document_count,
        "sentences_examined": sentence_count,
        "candidate_entities": len(entity_records),
        "candidate_relations": len(relation_records),
        "documents_by_domain": dict(domain_counts),
        "entity_file": str(entities_path),
        "relation_file": str(relations_path),
        "policy": "Conservative regex candidates only; complete sentence required for relations; all candidates retain evidence and remain unverified.",
    }
    (output_dir / "extraction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract evidence-backed candidates from Brain v3.1.")
    parser.add_argument("memory_directory")
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    print(json.dumps(run_extraction(args.memory_directory, args.output_directory), indent=2))
