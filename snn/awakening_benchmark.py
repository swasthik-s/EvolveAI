"""Reset-state generalization benchmark for Awakening's XOR readout.

Each seed trains online on random XOR examples, then evaluates on a balanced
set with learning/exploration disabled. Before EVERY evaluation trial, recurrent
state and reward-derived observation fields are reset, so each evaluation input
is tested from a fresh internal state. A frozen-readout control receives the
same training stream but never updates its readout.

Run from repository root:
    python snn/awakening_benchmark.py
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNN_DIR = ROOT / "snn"
if str(SNN_DIR) not in sys.path:
    sys.path.insert(0, str(SNN_DIR))

from awakening import AwakeningAgent


def mean_ci(differences: list[float]) -> dict[str, Any]:
    n = len(differences)
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {
        "paired_seeds": n,
        "mean_difference": round(mean, 4),
        "standard_deviation": round(sd, 4),
        "standard_error": round(se, 4),
        "approx_95pct_ci_low": round(mean - 1.96 * se, 4),
        "approx_95pct_ci_high": round(mean + 1.96 * se, 4),
        "online_better": sum(d > 0 for d in differences),
        "tied": sum(d == 0 for d in differences),
        "online_worse": sum(d < 0 for d in differences),
    }


def reset_for_independent_trial(agent: AwakeningAgent) -> None:
    """Clear recurrent/reward context without touching learned parameters."""
    agent.fast.zero_()
    agent.slow.zero_()
    agent.previous_spikes.zero_()
    agent.adaptation.zero_()
    agent.last_reward = 0.0
    agent.last_surprise = 0.0
    # The time/parity observation features must not reveal trial position.
    agent.step_count = 0


def run_condition(
    seed: int,
    train_steps: int,
    eval_repeats_per_pair: int,
    learning: bool,
) -> dict[str, Any]:
    agent = AwakeningAgent(seed=seed)
    if not learning:
        agent.learning_rate = 0.0

    train_records = []
    remaining = train_steps
    while remaining:
        batch = min(100, remaining)
        train_records.extend(agent.run(batch))
        remaining -= batch

    # Balanced evaluation set: each XOR combination occurs equally often.
    eval_pairs = [
        (a, b)
        for a in (0, 1)
        for b in (0, 1)
        for _ in range(eval_repeats_per_pair)
    ]
    random.Random(100_000 + seed).shuffle(eval_pairs)

    eval_records = []
    for pair in eval_pairs:
        reset_for_independent_trial(agent)
        eval_records.append(agent.step(input_pair=pair, learn=False, explore=False))

    combo_metrics = {}
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        rows = [r for r in eval_records if r["input_a"] == a and r["input_b"] == b]
        combo_metrics[f"{a}{b}"] = round(
            sum(r["greedy_correct"] for r in rows) / len(rows), 4
        )

    return {
        "seed": seed,
        "condition": "online_readout_update" if learning else "frozen_readout_control",
        "train_steps": len(train_records),
        "train_greedy_accuracy": round(
            sum(r["greedy_correct"] for r in train_records) / len(train_records), 4
        ),
        "evaluation_trials": len(eval_records),
        "heldout_greedy_accuracy": round(
            sum(r["greedy_correct"] for r in eval_records) / len(eval_records), 4
        ),
        "heldout_accuracy_by_input_pair": combo_metrics,
        "mean_eval_spikes_per_step": round(
            sum(r["spike_count"] for r in eval_records) / len(eval_records), 4
        ),
        "non_finite_resets_during_run": agent.fault_count,
        "all_states_finite": all(
            r["finite_state"] for r in train_records + eval_records
        ),
    }


def run_benchmark(
    seeds: list[int] | None = None,
    train_steps: int = 1500,
    eval_repeats_per_pair: int = 125,
) -> dict[str, Any]:
    seeds = list(range(1, 31)) if seeds is None else seeds
    results = [
        run_condition(seed, train_steps, eval_repeats_per_pair, learning)
        for seed in seeds
        for learning in (True, False)
    ]
    online = {r["seed"]: r for r in results if r["condition"] == "online_readout_update"}
    frozen = {r["seed"]: r for r in results if r["condition"] == "frozen_readout_control"}
    online_rows = list(online.values())
    frozen_rows = list(frozen.values())
    diffs = [
        online[s]["heldout_greedy_accuracy"] - frozen[s]["heldout_greedy_accuracy"]
        for s in seeds
    ]

    summary = {}
    for name, rows in (
        ("online_readout_update", online_rows),
        ("frozen_readout_control", frozen_rows),
    ):
        summary[name] = {
            "mean_train_greedy_accuracy": round(
                statistics.mean(r["train_greedy_accuracy"] for r in rows), 4
            ),
            "mean_heldout_greedy_accuracy": round(
                statistics.mean(r["heldout_greedy_accuracy"] for r in rows), 4
            ),
            "finite_runs": sum(r["all_states_finite"] for r in rows),
            "runs": len(rows),
        }

    return {
        "experiment": "awakening_reset_state_generalization_xor",
        "training": "random online XOR stream; target withheld from observation features",
        "evaluation": (
            "balanced shuffled XOR pairs; learning/exploration disabled; "
            "recurrent and reward context reset before every trial"
        ),
        "train_steps_per_run": train_steps,
        "evaluation_trials_per_run": 4 * eval_repeats_per_pair,
        "seeds": seeds,
        "chance_accuracy_reference": 0.5,
        "summary": summary,
        "paired_heldout_greedy_accuracy_difference": mean_ci(diffs),
        "interpretation_guidance": (
            "This is a stricter reset-state test: each evaluation trial starts with zeroed "
            "fast/slow/spike/adaptation state and cleared reward/surprise context; learned "
            "cell and readout parameters are retained. A positive paired advantage supports "
            "transfer through learned parameters on this XOR task only. It does not establish "
            "English acquisition, general intelligence, autonomy, or consciousness. Confidence "
            "intervals use a normal approximation."
        ),
        "runs": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
