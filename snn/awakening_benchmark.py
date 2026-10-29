"""Paired benchmark for Awakening's online readout learning.

Compares online updates with a frozen-readout control on the same seeded XOR
stream. Reports greedy accuracy separately from exploratory actions.

Run from repository root:
    python snn/awakening_benchmark.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNN_DIR = ROOT / "snn"
if str(SNN_DIR) not in sys.path:
    sys.path.insert(0, str(SNN_DIR))

from awakening import AwakeningAgent


def run_condition(seed: int, steps: int, learning: bool) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    agent = AwakeningAgent(seed=seed)
    if not learning:
        agent.learning_rate = 0.0

    records = []
    remaining = steps
    while remaining:
        batch = min(100, remaining)
        records.extend(agent.run(batch))
        remaining -= batch

    tail_size = min(200, len(records))
    tail = records[-tail_size:]
    return {
        "seed": seed,
        "condition": "online_readout_update" if learning else "frozen_readout_control",
        "steps": len(records),
        "overall_action_accuracy": sum(r["correct"] for r in records) / len(records),
        "overall_greedy_accuracy": sum(r["greedy_correct"] for r in records) / len(records),
        "final_window_action_accuracy": sum(r["correct"] for r in tail) / len(tail),
        "final_window_greedy_accuracy": sum(r["greedy_correct"] for r in tail) / len(tail),
        "exploration_rate_observed": sum(r["explored"] for r in records) / len(records),
        "mean_spikes_per_step": sum(r["spike_count"] for r in records) / len(records),
        "non_finite_resets": agent.fault_count,
        "all_states_finite": all(r["finite_state"] for r in records),
    }


def paired_stats(differences: list[float]) -> dict[str, float | int]:
    n = len(differences)
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {
        "paired_seeds": n,
        "mean_difference": round(mean, 4),
        "standard_deviation_of_difference": round(sd, 4),
        "standard_error": round(se, 4),
        "approx_95pct_ci_low": round(mean - 1.96 * se, 4),
        "approx_95pct_ci_high": round(mean + 1.96 * se, 4),
        "seeds_online_better": sum(d > 0 for d in differences),
        "seeds_tied": sum(d == 0 for d in differences),
        "seeds_online_worse": sum(d < 0 for d in differences),
    }


def run_benchmark(seeds: list[int] | None = None, steps: int = 2000) -> dict[str, Any]:
    seeds = list(range(1, 31)) if seeds is None else seeds
    results = [
        run_condition(seed, steps, learning)
        for seed in seeds
        for learning in (True, False)
    ]
    by_condition = {
        condition: [r for r in results if r["condition"] == condition]
        for condition in ("online_readout_update", "frozen_readout_control")
    }
    summary = {}
    for condition, rows in by_condition.items():
        summary[condition] = {
            "mean_overall_action_accuracy": round(statistics.mean(r["overall_action_accuracy"] for r in rows), 4),
            "mean_overall_greedy_accuracy": round(statistics.mean(r["overall_greedy_accuracy"] for r in rows), 4),
            "mean_final_window_action_accuracy": round(statistics.mean(r["final_window_action_accuracy"] for r in rows), 4),
            "mean_final_window_greedy_accuracy": round(statistics.mean(r["final_window_greedy_accuracy"] for r in rows), 4),
            "mean_observed_exploration_rate": round(statistics.mean(r["exploration_rate_observed"] for r in rows), 4),
            "mean_spikes_per_step": round(statistics.mean(r["mean_spikes_per_step"] for r in rows), 4),
            "finite_runs": sum(r["all_states_finite"] for r in rows),
            "runs": len(rows),
        }

    online = {r["seed"]: r for r in by_condition["online_readout_update"]}
    frozen = {r["seed"]: r for r in by_condition["frozen_readout_control"]}
    overall_diffs = [
        online[s]["overall_greedy_accuracy"] - frozen[s]["overall_greedy_accuracy"]
        for s in seeds
    ]
    final_diffs = [
        online[s]["final_window_greedy_accuracy"] - frozen[s]["final_window_greedy_accuracy"]
        for s in seeds
    ]
    return {
        "experiment": "awakening_paired_online_learning_vs_frozen_control",
        "task": "online XOR; target withheld from observation features",
        "steps_per_run": steps,
        "seeds": seeds,
        "chance_accuracy_reference": 0.5,
        "primary_metric": "greedy accuracy (argmax prediction, excluding random exploratory actions)",
        "summary": summary,
        "paired_overall_greedy_accuracy_difference": paired_stats(overall_diffs),
        "paired_final_window_greedy_accuracy_difference": paired_stats(final_diffs),
        "interpretation_guidance": (
            "A positive paired difference with a confidence interval above zero and a repeatable "
            "advantage across seeds supports a learning effect on this toy task. This does not "
            "demonstrate English acquisition, general intelligence, autonomy, or consciousness. "
            "The confidence interval is a normal-approximation interval, not an exact small-sample test."
        ),
        "runs": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
