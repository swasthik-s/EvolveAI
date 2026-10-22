"""Typed data structures for the symbolic autonomy subsystem.

This module contains no model calls. It represents goals and actions explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Goal:
    """A goal is satisfied when every predicate is present in the state."""

    goal_id: str
    description: str
    desired: frozenset[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_satisfied(self, state: set[str] | frozenset[str]) -> bool:
        return self.desired.issubset(state)


@dataclass(frozen=True)
class ActionSpec:
    """A deterministic action model using STRIPS-style state transitions."""

    action_id: str
    description: str
    preconditions: frozenset[str]
    add_effects: frozenset[str]
    delete_effects: frozenset[str] = frozenset()
    cost: float = 1.0
    risk: str = "low"  # low, medium, high, critical
    requires_approval: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("action_id cannot be empty")
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        if self.cost <= 0:
            raise ValueError("cost must be greater than zero")
        if self.risk not in {"low", "medium", "high", "critical"}:
            raise ValueError("risk must be low, medium, high, or critical")
        if self.add_effects & self.delete_effects:
            raise ValueError("a predicate cannot be both added and deleted")


@dataclass(frozen=True)
class Observation:
    action_id: str
    succeeded: bool
    facts_added: frozenset[str] = frozenset()
    facts_removed: frozenset[str] = frozenset()
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    goal_id: str
    status: str
    plan: list[str] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    final_state: set[str] = field(default_factory=set)
    reason: str = ""

    @property
    def completed(self) -> bool:
        return self.status == "completed"
