"""Tests for history-driven learning agenda selection."""
import unittest

from self_directed import create_learning_plan


class SelfDirectedPlannerTests(unittest.TestCase):
    def test_initial_plan_compares_both_architectures(self):
        plan = create_learning_plan({"cycles": [], "best_validation_loss": None}, max_trials=4, seed=7)
        architectures = {item["architecture"] for item in plan["proposed_experiments"]}
        self.assertEqual(plan["focus"], "establish_baseline")
        self.assertEqual(architectures, {"baseline", "evolve_context"})
        self.assertEqual(len(plan["proposed_experiments"]), 4)

    def test_improving_history_refines_historical_champion(self):
        state = {
            "cycles": [
                {"best_validation_loss": 3.5, "validation_protocol": "deterministic evenly-spaced validation windows v3"},
                {"best_validation_loss": 3.2, "validation_protocol": "deterministic evenly-spaced validation windows v3"},
            ],
            "best_validation_loss": 3.2,
            "best_trial": {
                "architecture": "baseline",
                "hidden_size": 64,
                "learning_rate": 0.001,
            },
        }
        plan = create_learning_plan(state, max_trials=4, seed=9)
        self.assertEqual(plan["focus"], "refine_champion")
        self.assertEqual(plan["proposed_experiments"][0]["architecture"], "baseline")
        self.assertTrue(any(x["architecture"] == "evolve_context" for x in plan["proposed_experiments"]))

    def test_plateau_triggers_exploration(self):
        state = {
            "cycles": [
                {"best_validation_loss": 3.2, "validation_protocol": "deterministic evenly-spaced validation windows v3"},
                {"best_validation_loss": 3.2, "validation_protocol": "deterministic evenly-spaced validation windows v3"},
            ],
            "best_validation_loss": 3.2,
            "best_trial": {
                "architecture": "baseline",
                "hidden_size": 64,
                "learning_rate": 0.001,
            },
        }
        plan = create_learning_plan(state, max_trials=4, seed=11)
        self.assertEqual(plan["focus"], "escape_plateau")
        self.assertGreaterEqual(plan["stale_cycles"], 1)
        self.assertTrue(any(x["architecture"] == "evolve_context" for x in plan["proposed_experiments"]))

    def test_all_proposals_stay_inside_reviewed_limits(self):
        plan = create_learning_plan({"cycles": []}, max_trials=6, seed=13)
        for item in plan["proposed_experiments"]:
            self.assertIn(item["architecture"], {"baseline", "evolve_context"})
            self.assertIn(item["hidden_size"], {32, 64, 96, 128})
            self.assertGreaterEqual(item["learning_rate"], 0.0001)
            self.assertLessEqual(item["learning_rate"], 0.005)


if __name__ == "__main__":
    unittest.main()
