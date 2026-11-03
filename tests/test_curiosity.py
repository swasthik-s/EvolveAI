import tempfile
import unittest
from pathlib import Path

from evolveai.curiosity import CuriosityEngine, Observation


class CuriosityEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = CuriosityEngine(str(Path(self.tmp.name) / "curiosity.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_discovers_concept_candidates_from_unseen_observations(self):
        items = self.engine.observe(Observation(
            content="A copper instrument produces vibration in a chamber.",
            source_id="source-a",
            source_uri="https://example.test/a",
        ))
        self.assertTrue(items)
        concepts = {item.target for item in items if item.gap_type == "unfamiliar_concept"}
        self.assertIn("copper", concepts)
        self.assertIn("instrument", concepts)
        self.assertTrue(all(item.evidence_ids for item in items))

    def test_does_not_duplicate_same_observation(self):
        observation = Observation(content="Quartz forms crystals under certain conditions.", source_id="s1")
        first = self.engine.observe(observation)
        second = self.engine.observe(observation)
        self.assertGreater(len(first), 0)
        self.assertEqual(len(second), len(self.engine.pending()))

    def test_question_targets_are_derived_from_input_not_subject_catalog(self):
        items = self.engine.observe(Observation(
            content="A fictional android studies human behaviour.",
            source_id="s2",
        ))
        questions = [item.question for item in items]
        self.assertTrue(any("android" in q for q in questions))
        self.assertTrue(any("human" in q for q in questions))
        self.assertFalse(any("Iron Man" in q for q in questions))

    def test_investigation_status_is_persistent(self):
        item = self.engine.observe(Observation(content="Basalt cools after volcanic activity.", source_id="s3"))[0]
        self.assertTrue(self.engine.set_status(item.investigation_id, "investigating"))
        self.assertNotIn(item.investigation_id, {x.investigation_id for x in self.engine.pending()})

    def test_bounds_content_and_pending_results(self):
        engine = CuriosityEngine(
            str(Path(self.tmp.name) / "bounded.db"),
            max_content_chars=120,
            max_active=2,
        )
        engine.observe(Observation(content=" ".join(["unusualterm"] * 100), source_id="s4"))
        self.assertLessEqual(len(engine.pending(limit=50)), 2)
        self.assertEqual(engine.stats()["observations"], 1)


if __name__ == "__main__":
    unittest.main()
