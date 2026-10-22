"""A bounded, deterministic planner for explicitly modelled actions."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .models import ActionSpec, Goal


@dataclass(frozen=True)
class PlanResult:
    actions: tuple[ActionSpec, ...]
    expanded_states: int
    reason: str

    @property
    def found(self) -> bool:
        return bool(self.actions) or self.reason == "goal_already_satisfied"


def apply_action(state: frozenset[str], action: ActionSpec) -> frozenset[str]:
    if not action.preconditions.issubset(state):
        raise ValueError(f"Action {action.action_id!r} preconditions are not satisfied")
    return frozenset((set(state) - set(action.delete_effects)) | set(action.add_effects))


def plan(
    initial_state: set[str] | frozenset[str],
    goal: Goal,
    actions: list[ActionSpec] | tuple[ActionSpec, ...],
    *,
    max_expanded_states: int = 10_000,
    max_plan_length: int = 40,
) -> PlanResult:
    """Breadth-first search; optimal in number of actions for unit-cost planning.

    This first version deliberately bounds search and uses explicit state predicates.
    It does not infer preconditions or effects from natural-language descriptions.
    """
    if max_expanded_states < 1 or max_plan_length < 0:
        raise ValueError("search limits must be positive (plan length may be zero)")
    start = frozenset(initial_state)
    if goal.is_satisfied(start):
        return PlanResult((), 0, "goal_already_satisfied")

    queue = deque([(start, tuple())])
    visited = {start}
    expanded = 0

    while queue:
        state, path = queue.popleft()
        if len(path) >= max_plan_length:
            continue
        for action in actions:
            if not action.preconditions.issubset(state):
                continue
            next_state = apply_action(state, action)
            if next_state in visited:
                continue
            next_path = path + (action,)
            if goal.is_satisfied(next_state):
                # Independently replay the returned plan before returning it.
                check = start
                for step in next_path:
                    check = apply_action(check, step)
                if not goal.is_satisfied(check):
                    raise AssertionError("Planner produced an invalid plan")
                return PlanResult(next_path, expanded, "plan_found")
            visited.add(next_state)
            queue.append((next_state, next_path))
            expanded += 1
            if expanded >= max_expanded_states:
                return PlanResult((), expanded, "search_budget_exhausted")

    return PlanResult((), expanded, "no_plan_found")
