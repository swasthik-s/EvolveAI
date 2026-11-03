import json
import tempfile
import unittest
from pathlib import Path

from brain.evidence_extractor import extract_from_sentence, run_extraction


class EvidenceExtractorTests(unittest.TestCase):
    def test_explicit_relation_keeps_exact_evidence(self):
        sentence = "Photosynthesis is a biological process in plants."
        _, relations = extract_from_sentence(sentence)
        self.assertEqual(relations, [])

    def test_rejects_existential_and_question_fragments(self):
        samples = [
            "There is a t",
            "If there are a total of 20 gnomes on the street, how many gnomes does the fifth house have?",
            "What is the averag",
            "France Surrenders Blogs are the hottest thing on the Net, but are they messing with traditional publishing principles?",
            "My current NTP blog entry is a good example.",
            "Liberal bias versus right-wing conspiracy is a wonderful distraction.",
        ]
        for sentence in samples:
            with self.subTest(sentence=sentence):
                self.assertEqual(extract_from_sentence(sentence)[1], [])

    def test_rejects_truncated_sentence(self):
        self.assertEqual(extract_from_sentence("Cary is an engineer in charge of designing")[1], [])

    def test_story_noise_is_not_promoted_as_entity(self):
        entities, _ = extract_from_sentence("Once upon a time, little Timmy went to the park.")
        self.assertNotIn("little boy", entities)
        self.assertNotIn("time little", entities)

    def test_entity_phrases_do_not_end_in_connectors(self):
        entities, _ = extract_from_sentence(
            "It met with positive sales in Japan, and was praised by both Japanese and western critics."
        )
        self.assertNotIn("Japanese and", entities)
        self.assertNotIn("November of", entities)

    def test_pipeline_writes_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "semantic_documents.jsonl").write_text(
                json.dumps({
                    "document_id": "doc-1",
                    "text": "Photosynthesis is a biological process in plants.",
                    "domain": "science",
                    "source_ids": ["sample-7"],
                }) + "\n",
                encoding="utf-8",
            )
            report = run_extraction(root)
            self.assertEqual(report["source_documents"], 1)
            self.assertEqual(report["candidate_relations"], 0)


if __name__ == "__main__":
    unittest.main()
