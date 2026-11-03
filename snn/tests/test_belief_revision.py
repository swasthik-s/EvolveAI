import tempfile
import unittest
from pathlib import Path

from brain.belief_revision import BeliefRevisionEngine


class BeliefRevisionTests(unittest.TestCase):
    def test_single_source_cannot_confirm_claim(self):
        engine = BeliefRevisionEngine()
        first = engine.observe("Earth", "orbits", "Sun", "Earth orbits the Sun.",
                               "source-a", reliability=0.9)
        engine.observe("Earth", "orbits", "Sun", "Repeated copy.", "source-a",
                       reliability=0.9)
        self.assertEqual(first["status"], "insufficient_evidence")
        self.assertEqual(engine.report()["observations"], 2)
        self.assertEqual(engine.get(first["claim_id"])["support_weight"], 0.9)

    def test_independent_support_changes_provisional_status(self):
        engine = BeliefRevisionEngine()
        engine.observe("Earth", "orbits", "Sun", "Evidence one.", "source-a", reliability=0.8)
        result = engine.observe("Earth", "orbits", "Sun", "Evidence two.", "source-b",
                                reliability=0.8)
        self.assertEqual(result["status"], "provisionally_supported")
        self.assertEqual(len(result["supporting_sources"]), 2)
        self.assertTrue(result["revisions"])

    def test_conflicting_independent_evidence_is_preserved(self):
        engine = BeliefRevisionEngine()
        for sid in ("s1", "s2"):
            engine.observe("Water", "boils_at", "100 C", "Supports claim.", sid,
                           polarity=1, reliability=0.8)
        for sid in ("s3", "s4"):
            result = engine.observe("Water", "boils_at", "100 C", "Contrary context.", sid,
                                    polarity=-1, reliability=0.8)
        self.assertEqual(result["status"], "conflicted")
        self.assertEqual(len(result["supporting_sources"]), 2)
        self.assertEqual(len(result["contradicting_sources"]), 2)

    def test_invalid_provenance_and_reliability_rejected(self):
        engine = BeliefRevisionEngine()
        with self.assertRaises(ValueError):
            engine.observe("A", "rel", "B", "Evidence", "", reliability=0.8)
        with self.assertRaises(ValueError):
            engine.observe("A", "rel", "B", "Evidence", "source", reliability=1.2)

    def test_persistence_preserves_revision_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "beliefs.json"
            engine = BeliefRevisionEngine(path)
            engine.observe("A", "rel", "B", "Evidence one.", "s1", reliability=0.8)
            result = engine.observe("A", "rel", "B", "Evidence two.", "s2",
                                    reliability=0.8)
            loaded = BeliefRevisionEngine(path).get(result["claim_id"])
            self.assertEqual(loaded["status"], "provisionally_supported")
            self.assertEqual(len(loaded["observations"]), 2)
            self.assertEqual(len(loaded["revisions"]), 1)


if __name__ == "__main__":
    unittest.main()
