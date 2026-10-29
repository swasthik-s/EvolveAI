"""Controlled benchmark for the Awakening SNN experiment.

Compares online readout updates against a frozen-readout control across seeds.
This is a diagnostic toy-task benchmark, not a test of general intelligence.
Run from repository root: python snn/awakening_benchmark.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNN_DIR = ROOT / "snn"
if str(SNN_DIR) not in sys.path:
    sys.path.insert(0, str(SNN_DIR))

import torch
from awakening import AwakeningAgent


def run_condition(seed: int, steps: int, learning: bool) -> dict[str, Any]:
    agent = AwakeningAgent(seed=seed)
    if not learning:
        agent.learning_rate = 0.0

    records = agent.run(steps)
    window = max(1, min(100, steps // 4))
    windows = []
    for start in range(0, len(records), window):
        chunk = records[start:start + window]
        if chunk:
            windows.append({
                "from_step": chunk[0]["step"],
                "to_step": chunk[-1]["step"],
                "accuracy": round(sum(r["correct"] for r in chunk) / len(chunk), 4),
                "mean_spikes": round(sum(r["spike_count"] for r in chunk) / len(chunk), 4),
            })

    return {
        "seed": seed,
        "condition": "online_readout_update" if learning else "frozen_readout_control",
        "steps": steps,
        "overall_accuracy": round(sum(r["correct"] for r in records) / len(records), 4),
        "last_window_accuracy": windows[-1]["accuracy"],
        "mean_spikes_per_step": round(sum(r["spike_count"] for r in records) / len(records), 4),
        "non_finite_resets": agent.fault_count,
        "all_states_finite": all(r["finite_state"] for r in records),
        "prediction_counts": {
            "predict_0": sum(r["prediction"] == 0 for r in records),
            "predict_1": sum(r["prediction"] == 1 for r in records),
        },
        "windows": windows,
    }


def run_benchmark(seeds: list[int] | None = None, steps: int = 1000) -> dict[str, Any]:
    seeds = seeds or [7, 21, 42, 73, 101]
    results = [
        run_condition(seed, steps, learning)
        for seed in seeds
        for learning in (True, False)
    ]
    grouped = {}
    for condition in ("online_readout_update", "frozen_readout_control"):
        rows = [r for r in results if r["condition"] == condition]
        grouped[condition] = {
            "mean_overall_accuracy": round(statistics.mean(r["overall_accuracy"] for r in rows), 4),
            "mean_last_window_accuracy": round(statistics.mean(r["last_window_accuracy"] for r in rows), 4),
            "mean_spikes_per_step": round(statistics.mean(r["mean_spikes_per_step"] for r in rows), 4),
            "finite_runs": sum(r["all_states_finite"] for r in rows),
            "runs": len(rows),
        }
    difference = grouped["online_readout_update"]["mean_last_window_accuracy"] - grouped["frozen_readout_control"]["mean_last_window_accuracy"]
    return {
        "experiment": "awakening_online_learning_vs_frozen_control",
        "task": "online XOR with target withheld from observation features",
        "steps_per_run": steps,
        "seeds": seeds,
        "chance_accuracy_reference": 0.5,
        "summary": grouped,
        "last_window_accuracy_difference_online_minus_frozen": round(difference, 4),
        "interpretation_guidance": (
            "Evidence for learning requires a repeatable advantage over the frozen control across seeds, "
            "finite state, and non-degenerate spiking. This task alone says nothing about English, "
            "general intelligence, autonomy, or consciousness."
        ),
        "runs": results,
    }


if __name__ == "__main__":
    report = run_benchmark()
    print(json.dumps(report, indent=2))
