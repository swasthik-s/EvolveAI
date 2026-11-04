import tempfile
import unittest
from pathlib import Path

from evolveai.curiosity import CuriosityEngine, Observation
from evolveai.cognitive_harness import CognitiveHarness, Evidence, InvestigationResult


class CognitiveHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.curiosity = CuriosityEngine(str(root / "curiosity.db"))
        self.harness = CognitiveHarness(
            self.curiosity,
            db_path=str(root / "harness.db"),
            max_investigations_per_cycle=3,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_observation_creates_auditable_investigation_candidates(self):
        candidates = self.harness.observe(Observation(
            content="A basalt formation contains olivine crystals.",
            source_id="geology-feed",
            source_uri="https://example.test/geology",
        ))
        self.assertTrue(candidates)
        events = self.harness.events()
        self.assertTrue(any(event["event_type"] == "observation_received" for event in events))
        self.assertTrue(any(event["event_type"] == "investigation_proposed" for event in events))

    def test_no_connector_defers_without_fabricating_evidence(self):
        self.harness.observe(Observation(
            content="A new instrument measures atmospheric pressure.",
            source_id="sensor-doc",
        ))
        selected = self.harness.prioritize(1)
        result = self.harness.investigate(selected[0])
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(self.harness.stats()["evidence_records"], 0)

    def test_investigator_evidence_is_persisted_with_provenance(self):
        def investigator(candidate):
            return InvestigationResult(
                success=True,
                explanation="A permitted reference supplied supporting context.",
                evidence=[Evidence(
                    content="Olivine is a mineral group commonly found in mafic igneous rocks.",
                    source_id="reference-a",
                    source_uri="https://example.test/olivine",
                    title="Olivine reference",
                )],
            )

        self.harness.investigator = investigator
        self.harness.observe(Observation(
            content="Olivine occurs in basaltic rocks.",
            source_id="field-note",
        ))
        candidate = next(x for x in self.harness.prioritize(3) if x.target == "olivine")
        result = self.harness.investigate(candidate)
        self.assertEqual(result["status"], "resolved")
        stored = self.harness.evidence_for(candidate.investigation_id)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["source_id"], "reference-a")
        self.assertEqual(stored[0]["source_uri"], "https://example.test/olivine")

    def test_cycle_is_bounded(self):
        self.harness.observe(Observation(
            content="A new material exhibits unusual thermal expansion.",
            source_id="materials-feed",
        ))
        results = self.harness.run_cycle(limit=2)
        self.assertLessEqual(len(results), 2)
        self.assertTrue(all(item["status"] == "deferred" for item in results))


if __name__ == "__main__":
    unittest.main()
