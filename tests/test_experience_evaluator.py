import unittest

from evolveai.experience_evaluator import Experience, ExperienceEvaluator


class ExperienceEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = ExperienceEvaluator()

    def test_empty_experience_is_rejected(self):
        result = self.evaluator.evaluate(Experience(content=" ", source_id="s"))
        self.assertEqual(result.decision, "reject_empty")

    def test_exact_or_near_duplicate_is_rejected(self):
        text = "Copper conducts electricity through mobile electrons in its structure."
        result = self.evaluator.evaluate(
            Experience(content=text, source_id="new"),
            known_texts=[text],
        )
        self.assertEqual(result.decision, "reject_duplicate")
        self.assertGreaterEqual(result.duplicate_similarity, 0.88)

    def test_possible_contradiction_is_compared_not_believed(self):
        known = "The alloy conducts electricity at room temperature."
        incoming = "The alloy does not conduct electricity at room temperature."
        result = self.evaluator.evaluate(
            Experience(content=incoming, source_id="s2"),
            known_texts=[known],
        )
        self.assertEqual(result.decision, "compare")
        self.assertGreater(result.contradiction_signal, 0.0)
        self.assertTrue(any("not proof" in reason for reason in result.reasons))

    def test_novelty_uses_supplied_known_terms(self):
        result = self.evaluator.evaluate(
            Experience(content="Quartz crystal oscillation frequency", source_id="s3"),
            known_terms={"quartz", "crystal"},
        )
        self.assertGreater(result.novelty, 0.0)
        self.assertLess(result.novelty, 1.0)

    def test_learned_relevance_scorer_is_bounded_and_used(self):
        evaluator = ExperienceEvaluator(relevance_scorer=lambda _: 0.95)
        result = evaluator.evaluate(
            Experience(content="A novel observation about a material", source_id="s4")
        )
        self.assertEqual(result.relevance, 0.95)

    def test_scores_stay_in_unit_interval(self):
        result = self.evaluator.evaluate(
            Experience(content="An observation contains several new concepts.", source_id="s5")
        )
        for value in (
            result.novelty, result.relevance, result.uncertainty,
            result.contradiction_signal, result.duplicate_similarity, result.priority,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
