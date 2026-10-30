"""Self-directed learning planner for EvolveAI.

The planner turns measured experiment history into bounded research objectives.
It chooses among reviewed configurations; it does not invent executable code,
fetch arbitrary URLs, or alter its own safety policy.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any


ALLOWED_ARCHITECTURES = ("baseline", "evolve_context")
ALLOWED_HIDDEN_SIZES = (32, 64, 96, 128)


def _best_config(state: dict[str, Any]) -> dict[str, Any]:
    best = state.get("best_trial") or {}
    return {
        "architecture": best.get("architecture", "baseline"),
        "hidden_size": int(best.get("hidden_size", 64)),
        "learning_rate": float(best.get("learning_rate", 0.001)),
    }


def create_learning_plan(state: dict[str, Any], max_trials: int = 4,
                         seed: int = 42) -> dict[str, Any]:
    """Diagnose progress and select a bounded experiment agenda."""
    cycles = state.get("cycles", [])
    previous = _best_config(state)
    losses = [
        float(item["best_validation_loss"])
        for item in cycles
        if item.get("best_validation_loss") is not None
    ]
    current_best = state.get("best_validation_loss")
    stale_cycles = 0
    if len(losses) >= 2:
        for older, newer in zip(reversed(losses[:-1]), reversed(losses[1:])):
            if newer < older - 1e-6:
                break
            stale_cycles += 1

    alternate = "baseline" if previous["architecture"] == "evolve_context" else "evolve_context"
    if not cycles:
        focus = "establish_baseline"
        objective = "Find a reliable starting architecture and learning-rate region."
        hypothesis = "The simpler baseline and context-augmented SNN may have different learning dynamics."
        rationale = "No prior cycle exists, so compare both reviewed architectures before committing to a direction."
        candidates = [
            ("baseline", 64, 0.001),
            ("evolve_context", 64, 0.001),
            ("baseline", 32, 0.001),
            ("evolve_context", 32, 0.001),
            ("baseline", 64, 0.0005),
            ("evolve_context", 64, 0.0005),
        ]
    elif stale_cycles >= 1:
        focus = "escape_plateau"
        objective = "Test whether architecture or capacity changes improve on the current plateau."
        hypothesis = "Recent cycles have not improved the historical best; a controlled architecture/capacity change may escape the plateau."
        rationale = f"The last {stale_cycles + 1} recorded results show no strict improvement."
        candidates = [
            (previous["architecture"], previous["hidden_size"], previous["learning_rate"]),
            (alternate, previous["hidden_size"], previous["learning_rate"]),
            (alternate, 32 if previous["hidden_size"] >= 64 else 64, previous["learning_rate"]),
            (previous["architecture"], previous["hidden_size"], max(0.0001, previous["learning_rate"] * 0.5)),
            (previous["architecture"], previous["hidden_size"], min(0.005, previous["learning_rate"] * 2.0)),
            (alternate, previous["hidden_size"], max(0.0001, previous["learning_rate"] * 0.5)),
        ]
    else:
        focus = "refine_champion"
        objective = "Continue the best model while testing nearby settings and retaining an architecture control."
        hypothesis = "A modest continuation or learning-rate adjustment may reduce held-out prediction error without losing the current capability."
        rationale = "The current champion is improving or there is not enough history to diagnose a plateau."
        candidates = [
            (previous["architecture"], previous["hidden_size"], previous["learning_rate"]),
            (previous["architecture"], previous["hidden_size"], max(0.0001, previous["learning_rate"] * 0.5)),
            (previous["architecture"], 32 if previous["hidden_size"] >= 64 else 64, previous["learning_rate"]),
            (alternate, previous["hidden_size"], previous["learning_rate"]),
            (previous["architecture"], previous["hidden_size"], min(0.005, previous["learning_rate"] * 2.0)),
            (alternate, 32 if previous["hidden_size"] >= 64 else 64, previous["learning_rate"]),
        ]

    configs = []
    seen = set()
    for architecture, hidden_size, learning_rate in candidates:
        if architecture not in ALLOWED_ARCHITECTURES or hidden_size not in ALLOWED_HIDDEN_SIZES:
            continue
        learning_rate = float(learning_rate)
        if not 0.0001 <= learning_rate <= 0.005:
            continue
        key = (architecture, int(hidden_size), round(learning_rate, 7))
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "architecture": architecture,
            "hidden_size": int(hidden_size),
            "learning_rate": learning_rate,
            "seed": int(seed),
        })
        if len(configs) >= max(1, min(int(max_trials), 6)):
            break

    return {
        "planner_version": 1,
        "cycle_number": len(cycles) + 1,
        "focus": focus,
        "objective": objective,
        "hypothesis": hypothesis,
        "rationale": rationale,
        "previous_historical_best": current_best,
        "recent_validation_losses": losses[-5:],
        "stale_cycles": stale_cycles,
        "proposed_experiments": configs,
        "decision_policy": "History-driven selection among reviewed architectures, hidden sizes, and learning rates; all runs remain bounded by the governor.",
    }
