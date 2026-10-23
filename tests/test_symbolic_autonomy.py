import pytest

from evolveai.autonomy import ActionSpec, Goal, Observation, plan, run_goal


def action(action_id, pre=(), add=(), delete=(), risk="low", approval=False):
    return ActionSpec(
        action_id=action_id,
        description=action_id,
        preconditions=frozenset(pre),
        add_effects=frozenset(add),
        delete_effects=frozenset(delete),
        risk=risk,
        requires_approval=approval,
    )


def test_planner_finds_valid_multistep_plan():
    actions = [
        action("load", add={"loaded"}),
        action("reconcile", pre={"loaded"}, add={"reconciled"}),
        action("verify", pre={"reconciled"}, add={"verified"}),
    ]
    result = plan(set(), Goal("g1", "verify report", frozenset({"verified"})), actions)
    assert result.found
    assert [a.action_id for a in result.actions] == ["load", "reconcile", "verify"]


def test_already_satisfied_goal_returns_empty_plan():
    result = plan({"done"}, Goal("g", "done", frozenset({"done"})), [])
    assert result.found
    assert result.reason == "goal_already_satisfied"
    assert result.actions == ()


def test_impossible_goal_is_not_claimed_complete():
    result = plan(set(), Goal("g", "impossible", frozenset({"missing"})), [])
    assert not result.found
    assert result.reason == "no_plan_found"


def test_controller_executes_and_verifies_goal():
    actions = [action("prepare", add={"prepared"}), action("finish", pre={"prepared"}, add={"done"})]
    events = []

    def execute(spec, state):
        return Observation(spec.action_id, True, facts_added=spec.add_effects, facts_removed=spec.delete_effects)

    result = run_goal(
        Goal("g", "finish task", frozenset({"done"})), set(), actions,
        execute, lambda spec: False, event_sink=events.append,
    )
    assert result.completed
    assert result.executed == ["prepare", "finish"]
    assert any(event["event"] == "goal_finished" for event in events)


def test_high_risk_action_requires_authorization():
    actions = [action("delete_data", add={"deleted"}, risk="critical")]
    result = run_goal(
        Goal("g", "delete data", frozenset({"deleted"})), set(), actions,
        lambda spec, state: Observation(spec.action_id, True, facts_added=spec.add_effects),
        lambda spec: False,
    )
    assert result.status == "approval_required"
    assert result.executed == []


def test_action_failure_stops_run():
    actions = [action("unstable", add={"done"})]
    result = run_goal(
        Goal("g", "finish", frozenset({"done"})), set(), actions,
        lambda spec, state: Observation(spec.action_id, False, detail="failed"),
        lambda spec: True,
    )
    assert result.status == "action_failed"
    assert not result.completed


def test_search_budget_is_bounded():
    actions = [action(f"a{i}", add={f"fact{i}"}) for i in range(5)]
    result = plan(set(), Goal("g", "unreachable", frozenset({"never"})), actions, max_expanded_states=1)
    assert result.reason == "search_budget_exhausted"
