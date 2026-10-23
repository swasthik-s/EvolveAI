"""Symbolic, model-free autonomy primitives for EvolveAI.

This module deliberately contains no LLM calls. It provides a small, inspectable
goal/action loop. Domain-specific tools are injected by the application.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol
import json
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GoalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class Goal:
    description: str
    success_criteria: tuple[str, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)
    max_steps: int = 12
    max_cost: float = 10.0
    goal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class ActionSpec:
    """An explicitly registered action with declarative prerequisites and effects."""
    name: str
    description: str
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    cost: float = 1.0
    risk: float = 0.0
    reversible: bool = True
    requires_approval: bool = False
    input_schema: Mapping[str, str] = field(default_factory=dict)


@dataclass
class ActionResult:
    success: bool
    observations: list[str] = field(default_factory=list)
    output: Any = None
    error: str = ""
    cost: float = 1.0
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TraceEvent:
    timestamp: str
    event: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalState:
    goal: Goal
    facts: set[str] = field(default_factory=set)
    artifacts: dict[str, Any] = field(default_factory=dict)
    completed_actions: list[str] = field(default_factory=list)
    failed_actions: dict[str, int] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    total_cost: float = 0.0
    steps: int = 0
    last_error: str = ""

    def record(self, event: str, detail: str, **data: Any) -> None:
        self.events.append(TraceEvent(utc_now(), event, detail, data))


class ActionExecutor(Protocol):
    def __call__(self, goal: Goal, state: GoalState) -> ActionResult: ...


class ActionRegistry:
    """Registry of approved, explicitly implemented actions; never executes arbitrary text."""

    def __init__(self) -> None:
        self._specs: dict[str, ActionSpec] = {}
        self._executors: dict[str, ActionExecutor] = {}

    def register(self, spec: ActionSpec, executor: ActionExecutor) -> None:
        if not spec.name or spec.name in self._specs:
            raise ValueError(f"Action name is empty or already registered: {spec.name!r}")
        if spec.cost < 0 or spec.risk < 0:
            raise ValueError("Action cost and risk must be non-negative")
        self._specs[spec.name] = spec
        self._executors[spec.name] = executor

    def specs(self) -> tuple[ActionSpec, ...]:
        return tuple(self._specs.values())

    def execute(self, name: str, goal: Goal, state: GoalState) -> ActionResult:
        if name not in self._executors:
            raise KeyError(f"Action is not registered: {name}")
        return self._executors[name](goal, state)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


class ExecutionPolicy:
    """Fail-closed policy gate. Approval is an injected callback, not inferred."""

    def __init__(
        self,
        approval: Callable[[ActionSpec, Goal, GoalState], bool] | None = None,
        max_action_risk: float = 0.25,
    ) -> None:
        self.approval = approval or (lambda _spec, _goal, _state: False)
        self.max_action_risk = max_action_risk

    def authorize(self, spec: ActionSpec, goal: Goal, state: GoalState) -> PolicyDecision:
        if spec.risk > self.max_action_risk:
            return PolicyDecision(False, "action exceeds configured risk threshold")
        if spec.requires_approval or not spec.reversible:
            if not self.approval(spec, goal, state):
                return PolicyDecision(False, "explicit approval required")
        return PolicyDecision(True, "permitted by execution policy")


class SymbolicPlanner:
    """Greedy forward planner over explicit facts, preconditions, and effects.

    This is intentionally a bounded baseline, not a complete STRIPS/HTN planner.
    Actions with unmet prerequisites are not eligible.
    """

    def rank(
        self,
        specs: Iterable[ActionSpec],
        state: GoalState,
        goal: Goal,
    ) -> list[ActionSpec]:
        missing = set(goal.success_criteria) - state.facts
        eligible = [
            spec for spec in specs
            if spec.name not in state.completed_actions
            and state.failed_actions.get(spec.name, 0) < 2
            and set(spec.preconditions).issubset(state.facts)
        ]

        def score(spec: ActionSpec) -> tuple[float, float, str]:
            progress = len(set(spec.effects) & missing)
            information = 0.2 if not spec.effects else 0.0
            # Prefer actions that make goal progress, then low-cost/low-risk actions.
            utility = progress + information - (0.15 * spec.cost) - (0.5 * spec.risk)
            return (utility, -spec.cost, spec.name)

        return sorted(eligible, key=score, reverse=True)


class AutonomousEngine:
    """Run a goal through a bounded observe-plan-act-verify loop."""

    def __init__(
        self,
        registry: ActionRegistry,
        planner: SymbolicPlanner | None = None,
        policy: ExecutionPolicy | None = None,
        verifier: Callable[[Goal, GoalState], bool] | None = None,
        observer: Callable[[ActionSpec, ActionResult, GoalState], None] | None = None,
        event_sink: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner or SymbolicPlanner()
        self.policy = policy or ExecutionPolicy()
        self.verifier = verifier or (
            lambda goal, state: set(goal.success_criteria).issubset(state.facts)
        )
        self.observer = observer
        self.event_sink = event_sink

    def _record(self, state: GoalState, event: str, detail: str, **data: Any) -> None:
        state.record(event, detail, **data)
        if self.event_sink:
            self.event_sink(state.events[-1])

    def run(self, goal: Goal, initial_facts: Iterable[str] = ()) -> GoalState:
        state = GoalState(goal=goal, facts=set(initial_facts), status=GoalStatus.RUNNING)
        self._record(state, "goal_started", goal.description, goal_id=goal.goal_id)

        while not self.verifier(goal, state):
            if state.steps >= max(0, goal.max_steps):
                state.status = GoalStatus.BUDGET_EXHAUSTED
                self._record(state, "stopped", "maximum step count reached")
                return state
            if state.total_cost >= max(0.0, goal.max_cost):
                state.status = GoalStatus.BUDGET_EXHAUSTED
                self._record(state, "stopped", "maximum execution cost reached")
                return state

            candidates = self.planner.rank(self.registry.specs(), state, goal)
            if not candidates:
                state.status = GoalStatus.BLOCKED
                missing = sorted(set(goal.success_criteria) - state.facts)
                self._record(state, "blocked", "no eligible action can advance the goal",
                             missing_success_criteria=missing)
                return state

            selected = None
            for candidate in candidates:
                decision = self.policy.authorize(candidate, goal, state)
                if decision.allowed:
                    selected = candidate
                    break
                self._record(state, "action_denied", decision.reason, action=candidate.name)

            if selected is None:
                state.status = GoalStatus.BLOCKED
                state.last_error = "All candidate actions were denied by policy."
                self._record(state, "blocked", state.last_error)
                return state

            self._record(state, "action_started", selected.description, action=selected.name)
            try:
                result = self.registry.execute(selected.name, goal, state)
            except Exception as exc:  # Tool boundary: record error and re-plan.
                result = ActionResult(success=False, error=f"{type(exc).__name__}: {exc}",
                                      cost=selected.cost)

            state.steps += 1
            state.total_cost += max(0.0, result.cost)
            if result.success:
                state.completed_actions.append(selected.name)
                # Only observations returned by the executor become facts. Declared effects
                # are planning metadata, not proof that an effect actually occurred.
                state.facts.update(result.observations)
                if result.output is not None:
                    state.artifacts[selected.name] = result.output
                state.last_error = ""
                self._record(state, "action_succeeded", selected.name,
                             observations=result.observations, evidence=result.evidence)
            else:
                state.failed_actions[selected.name] = state.failed_actions.get(selected.name, 0) + 1
                state.last_error = result.error or "Action failed without an error description."
                self._record(state, "action_failed", state.last_error, action=selected.name)

            if self.observer:
                self.observer(selected, result, state)

        state.status = GoalStatus.SUCCEEDED
        self._record(state, "goal_verified", "success criteria independently satisfied",
                     success_criteria=list(goal.success_criteria))
        return state


def state_to_dict(state: GoalState) -> dict[str, Any]:
    """JSON-serializable execution snapshot suitable for persistence or admin inspection."""
    payload = {
        "goal": asdict(state.goal),
        "facts": sorted(state.facts),
        "artifacts": state.artifacts,
        "completed_actions": list(state.completed_actions),
        "failed_actions": dict(state.failed_actions),
        "status": state.status.value,
        "total_cost": state.total_cost,
        "steps": state.steps,
        "last_error": state.last_error,
        "events": [asdict(event) for event in state.events],
    }
    # Validate that snapshots never contain unserializable objects.
    json.dumps(payload, default=str)
    return payload
