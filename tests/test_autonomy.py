import unittest

from evolveai.autonomy import (
    ActionRegistry,
    ActionResult,
    ActionSpec,
    AutonomousEngine,
    ExecutionPolicy,
    Goal,
    GoalStatus,
    state_to_dict,
)


class AutonomousEngineTests(unittest.TestCase):
    def test_runs_prerequisite_actions_and_verifies_goal(self):
        registry = ActionRegistry()
        registry.register(
            ActionSpec(
                name="collect",
                description="Collect input data",
                effects=("data_collected",),
                cost=1,
            ),
            lambda goal, state: ActionResult(True, observations=["data_collected"]),
        )
        registry.register(
            ActionSpec(
                name="analyze",
                description="Analyze collected data",
                preconditions=("data_collected",),
                effects=("report_verified",),
                cost=1,
            ),
            lambda goal, state: ActionResult(True, observations=["report_verified"]),
        )
        engine = AutonomousEngine(registry)
        goal = Goal(
            description="Create a verified report",
            success_criteria=("report_verified",),
            max_steps=4,
            max_cost=5,
        )

        state = engine.run(goal)

        self.assertEqual(state.status, GoalStatus.SUCCEEDED)
        self.assertEqual(state.completed_actions, ["collect", "analyze"])
        self.assertTrue(any(e.event == "goal_verified" for e in state.events))
        self.assertEqual(state_to_dict(state)["status"], "succeeded")

    def test_blocks_when_prerequisite_cannot_be_met(self):
        registry = ActionRegistry()
        registry.register(
            ActionSpec(
                name="analyze",
                description="Analyze unavailable data",
                preconditions=("data_collected",),
                effects=("report_verified",),
            ),
            lambda goal, state: ActionResult(True, observations=["report_verified"]),
        )
        state = AutonomousEngine(registry).run(
            Goal("Create report", success_criteria=("report_verified",)),
        )
        self.assertEqual(state.status, GoalStatus.BLOCKED)

    def test_retries_failed_action_only_with_bounded_attempts(self):
        registry = ActionRegistry()
        registry.register(
            ActionSpec(name="flaky", description="Failing tool", effects=("done",)),
            lambda goal, state: ActionResult(False, error="temporary error"),
        )
        state = AutonomousEngine(registry).run(
            Goal("Finish", success_criteria=("done",), max_steps=10),
        )
        self.assertEqual(state.status, GoalStatus.BLOCKED)
        self.assertEqual(state.failed_actions["flaky"], 2)

    def test_requires_approval_for_irreversible_actions(self):
        registry = ActionRegistry()
        registry.register(
            ActionSpec(
                name="delete",
                description="Delete external resource",
                effects=("done",),
                reversible=False,
            ),
            lambda goal, state: ActionResult(True, observations=["done"]),
        )
        state = AutonomousEngine(registry, policy=ExecutionPolicy()).run(
            Goal("Finish", success_criteria=("done",)),
        )
        self.assertEqual(state.status, GoalStatus.BLOCKED)
        self.assertTrue(any(e.event == "action_denied" for e in state.events))

    def test_stops_when_step_budget_is_exhausted(self):
        registry = ActionRegistry()
        registry.register(
            ActionSpec(name="step", description="One step", effects=("not_done",)),
            lambda goal, state: ActionResult(True, observations=["not_done"]),
        )
        state = AutonomousEngine(registry).run(
            Goal("Finish", success_criteria=("done",), max_steps=1),
        )
        self.assertEqual(state.status, GoalStatus.BUDGET_EXHAUSTED)
        self.assertEqual(state.steps, 1)


if __name__ == "__main__":
    unittest.main()
