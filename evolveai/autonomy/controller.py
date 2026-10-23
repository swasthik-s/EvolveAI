"""Bounded execution controller for registered symbolic actions."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from .models import ActionSpec, Goal, Observation, RunResult
from .planner import plan


class ActionExecutor(Protocol):
    def __call__(self, action: ActionSpec, state: set[str]) -> Observation: ...


class AuthorizationPolicy(Protocol):
    def __call__(self, action: ActionSpec) -> bool: ...


def run_goal(
    goal: Goal,
    initial_state: set[str],
    actions: list[ActionSpec],
    executor: ActionExecutor,
    authorize: AuthorizationPolicy,
    *,
    max_actions: int = 20,
    max_replans: int = 3,
    max_expanded_states: int = 10_000,
    event_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> RunResult:
    """Plan, execute, verify, and replan within strict action limits.

    Executor implementations must enforce their own timeouts and sandboxing.
    An observation only changes state when the executor reports success.
    """
    result = RunResult(goal_id=goal.goal_id, status="running", final_state=set(initial_state))
    action_map = {a.action_id: a for a in actions}
    if len(action_map) != len(actions):
        return _finish(result, "invalid_action_registry", "Duplicate action IDs", event_sink)

    def emit(event: str, **fields: object) -> None:
        if event_sink:
            event_sink({"event": event, "goal_id": goal.goal_id, **fields})

    replans = 0
    while len(result.executed) < max_actions:
        if goal.is_satisfied(result.final_state):
            return _finish(result, "completed", "Goal predicates verified", event_sink)

        planned = plan(
            result.final_state, goal, actions,
            max_expanded_states=max_expanded_states,
            max_plan_length=max_actions - len(result.executed),
        )
        if not planned.found:
            status = "budget_exhausted" if planned.reason == "search_budget_exhausted" else "blocked"
            return _finish(result, status, planned.reason, event_sink)

        result.plan = [a.action_id for a in planned.actions]
        emit("plan_selected", plan=result.plan, expanded_states=planned.expanded_states)

        needs_replan = False
        for action in planned.actions:
            if len(result.executed) >= max_actions:
                return _finish(result, "budget_exhausted", "Maximum action count reached", event_sink)
            if not action.preconditions.issubset(result.final_state):
                needs_replan = True
                break
            if action.requires_approval or action.risk in {"high", "critical"}:
                if not authorize(action):
                    emit("action_denied", action_id=action.action_id, risk=action.risk)
                    return _finish(result, "approval_required", f"Action denied: {action.action_id}", event_sink)
            try:
                observation = executor(action, set(result.final_state))
            except Exception as exc:  # adapters must return sanitized details in production
                emit("action_error", action_id=action.action_id, error_type=type(exc).__name__)
                return _finish(result, "action_failed", f"Executor failed for {action.action_id}", event_sink)
            result.observations.append(observation)
            result.executed.append(action.action_id)
            emit("action_observed", action_id=action.action_id, succeeded=observation.succeeded)

            if not observation.succeeded:
                return _finish(result, "action_failed", observation.detail or action.action_id, event_sink)

            # Apply observed effects, not merely the planned effects. The adapter is
            # responsible for truthful observation and independent postcondition checks.
            result.final_state.difference_update(observation.facts_removed)
            result.final_state.update(observation.facts_added)
            if not observation.facts_added and not observation.facts_removed:
                # A simple deterministic adapter may return no explicit delta. In that
                # case, use its declared effects as the observed transition.
                result.final_state.difference_update(action.delete_effects)
                result.final_state.update(action.add_effects)

            if goal.is_satisfied(result.final_state):
                return _finish(result, "completed", "Goal predicates verified", event_sink)
            if not action.add_effects.issubset(result.final_state):
                needs_replan = True
                break

        if needs_replan:
            replans += 1
            if replans > max_replans:
                return _finish(result, "replan_limit_reached", "Repeated plan invalidation", event_sink)

    return _finish(result, "budget_exhausted", "Maximum action count reached", event_sink)


def _finish(
    result: RunResult,
    status: str,
    reason: str,
    event_sink: Callable[[Mapping[str, object]], None] | None,
) -> RunResult:
    result.status = status
    result.reason = reason
    if event_sink:
        event_sink({"event": "goal_finished", "goal_id": result.goal_id, "status": status, "reason": reason})
    return result
