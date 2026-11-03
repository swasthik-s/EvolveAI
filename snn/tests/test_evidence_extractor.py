import json
import tempfile
import unittest
from pathlib import Path

from brain.evidence_extractor import extract_from_sentence, run_extraction


class EvidenceExtractorTests(unittest.TestCase):
    def test_explicit_relation_keeps_exact_evidence(self):
        sentence = "Photosynthesis is a biological process in plants."
        entities, relations = extract_from_sentence(sentence)
        self.assertTrue(any(r["relation_candidate"] == "is_a" for r in relations))
        relation = next(r for r in relations if r["relation_candidate"] == "is_a")
        self.assertEqual(relation["evidence_sentence"], sentence)
        self.assertEqual(relation["status"], "unverified_candidate")

    def test_story_noise_is_not_promoted_as_entity(self):
        entities, _ = extract_from_sentence("Once upon a time, little Timmy went to the park.")
        self.assertNotIn("little boy", entities)
        self.assertNotIn("time little", entities)

    def test_pipeline_writes_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "semantic_documents.jsonl").write_text(
                json.dumps({
                    "document_id": "doc-1",
                    "text": "Photosynthesis is a biological process in plants.",
                    "domain": "science",
                    "source_ids": ["sample-7"],
                }) + "\\n",
                encoding="utf-8",
            )
            report = run_extraction(root)
            self.assertEqual(report["source_documents"], 1)
            self.assertGreater(report["candidate_relations"], 0)
            rows = (root / "v3_2_extraction" / "candidate_relations.jsonl").read_text(encoding="utf-8")
            record = json.loads(rows.splitlines()[0])
            self.assertEqual(record["document_id"], "doc-1")
            self.assertEqual(record["source_ids"], ["sample-7"])


if __name__ == "__main__":
    unittest.main()
