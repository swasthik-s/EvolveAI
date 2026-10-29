"""Controlled reset-state XOR benchmark for Awakening plus a nonlinear sanity baseline.

Compares:
1. online SNN readout update,
2. same SNN with a frozen readout,
3. a small supervised nonlinear MLP trained on the same two-bit XOR task.

The MLP is a task/pipeline sanity check, not a fair claim that it is biologically
similar or computationally equivalent to the SNN. Every SNN evaluation trial
starts with zero recurrent state, learning/exploration disabled, and stationary
input features. This makes the result diagnostic rather than just a reward score.

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

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SNN_DIR = ROOT / "snn"
if str(SNN_DIR) not in sys.path:
    sys.path.insert(0, str(SNN_DIR))

from awakening import AwakeningAgent


PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))


def observation(a: int, b: int) -> list[float]:
    return [float(a), float(b), 0.0, 0.0, 0.0, 0.0, 0.0]


def mean_ci(differences: list[float]) -> dict[str, Any]:
    n = len(differences)
    mean = statistics.mean(differences) if n else 0.0
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
    """Clear transient dynamics while preserving learned parameters."""
    agent.fast.zero_()
    agent.slow.zero_()
    agent.previous_spikes.zero_()
    agent.adaptation.zero_()
    agent.last_reward = 0.0
    agent.last_surprise = 0.0
    agent.step_count = 0


def pair_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for a, b in PAIRS:
        rows = [r for r in records if r["input_a"] == a and r["input_b"] == b]
        result[f"{a}{b}"] = round(
            sum(bool(r["greedy_correct"]) for r in rows) / len(rows), 4
        )
    return result


def run_snn_condition(
    seed: int, train_steps: int, eval_repeats_per_pair: int, learning: bool
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

    eval_pairs = [
        (a, b)
        for a, b in PAIRS
        for _ in range(eval_repeats_per_pair)
    ]
    random.Random(100_000 + seed).shuffle(eval_pairs)
    eval_records = []
    for pair in eval_pairs:
        reset_for_independent_trial(agent)
        eval_records.append(agent.step(input_pair=pair, learn=False, explore=False))

    predictions = [r["greedy_prediction"] for r in eval_records]
    positive_rate = sum(predictions) / len(predictions)
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
        "heldout_accuracy_by_input_pair": pair_metrics(eval_records),
        "heldout_prediction_1_rate": round(positive_rate, 4),
        "mean_absolute_score_margin": round(
            sum(abs(r["score_margin"]) for r in eval_records) / len(eval_records), 6
        ),
        "mean_eval_spikes_per_step": round(
            sum(r["spike_count"] for r in eval_records) / len(eval_records), 4
        ),
        "non_finite_resets_during_run": agent.fault_count,
        "all_states_finite": all(
            r["finite_state"] for r in train_records + eval_records
        ),
    }


def run_mlp_baseline(seed: int, train_steps: int, eval_repeats_per_pair: int) -> dict[str, Any]:
    """Train a tiny nonlinear supervised baseline as a task sanity check."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 50_000)
        model = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        rng = random.Random(seed + 70_000)
        model.train()
        for start in range(0, train_steps, 32):
            batch_size = min(32, train_steps - start)
            pairs = [(rng.randint(0, 1), rng.randint(0, 1)) for _ in range(batch_size)]
            x = torch.tensor([observation(a, b) for a, b in pairs], dtype=torch.float32)
            y = torch.tensor([a ^ b for a, b in pairs], dtype=torch.long)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        eval_pairs = [
            (a, b)
            for a, b in PAIRS
            for _ in range(eval_repeats_per_pair)
        ]
        random.Random(200_000 + seed).shuffle(eval_pairs)
        with torch.no_grad():
            x = torch.tensor([observation(a, b) for a, b in eval_pairs], dtype=torch.float32)
            targets = torch.tensor([a ^ b for a, b in eval_pairs], dtype=torch.long)
            logits = model(x)
            predictions = logits.argmax(dim=1)
            accuracy = float((predictions == targets).float().mean().item())
            per_pair = {}
            for a, b in PAIRS:
                idx = [i for i, pair in enumerate(eval_pairs) if pair == (a, b)]
                per_pair[f"{a}{b}"] = round(
                    float((predictions[idx] == targets[idx]).float().mean().item()), 4
                )
            return {
                "seed": seed,
                "condition": "nonlinear_mlp_sanity_baseline",
                "train_steps": train_steps,
                "evaluation_trials": len(eval_pairs),
                "heldout_greedy_accuracy": round(accuracy, 4),
                "heldout_accuracy_by_input_pair": per_pair,
                "all_states_finite": bool(torch.isfinite(logits).all().item()),
            }


def run_benchmark(
    seeds: list[int] | None = None,
    train_steps: int = 1500,
    eval_repeats_per_pair: int = 125,
) -> dict[str, Any]:
    seeds = list(range(1, 31)) if seeds is None else list(seeds)
    results = []
    for seed in seeds:
        results.append(run_snn_condition(seed, train_steps, eval_repeats_per_pair, True))
        results.append(run_snn_condition(seed, train_steps, eval_repeats_per_pair, False))
        results.append(run_mlp_baseline(seed, train_steps, eval_repeats_per_pair))

    online = {r["seed"]: r for r in results if r["condition"] == "online_readout_update"}
    frozen = {r["seed"]: r for r in results if r["condition"] == "frozen_readout_control"}
    mlp = {r["seed"]: r for r in results if r["condition"] == "nonlinear_mlp_sanity_baseline"}
    paired = [online[s]["heldout_greedy_accuracy"] - frozen[s]["heldout_greedy_accuracy"] for s in seeds]

    summary = {}
    for name, by_seed in (
        ("online_readout_update", online),
        ("frozen_readout_control", frozen),
        ("nonlinear_mlp_sanity_baseline", mlp),
    ):
        rows = list(by_seed.values())
        summary[name] = {
            "mean_heldout_greedy_accuracy": round(
                statistics.mean(r["heldout_greedy_accuracy"] for r in rows), 4
            ),
            "mean_heldout_accuracy_by_input_pair": {
                pair: round(statistics.mean(r["heldout_accuracy_by_input_pair"][pair] for r in rows), 4)
                for pair in ("00", "01", "10", "11")
            },
            "finite_runs": sum(bool(r["all_states_finite"]) for r in rows),
            "runs": len(rows),
        }
        if name != "nonlinear_mlp_sanity_baseline":
            summary[name]["mean_train_greedy_accuracy"] = round(
                statistics.mean(r["train_greedy_accuracy"] for r in rows), 4
            )

    return {
        "experiment": "awakening_reset_state_generalization_xor_v2",
        "training": (
            "random online XOR stream for SNN; target withheld from stationary input features; "
            "nonlinear MLP trained as a separate supervised sanity baseline"
        ),
        "evaluation": (
            "balanced shuffled XOR pairs; SNN learning/exploration disabled; transient recurrent "
            "state reset before every trial; stationary observation features used in train and eval"
        ),
        "train_steps_per_run": train_steps,
        "evaluation_trials_per_run": 4 * eval_repeats_per_pair,
        "seeds": seeds,
        "chance_accuracy_reference": 0.5,
        "summary": summary,
        "paired_snn_heldout_greedy_accuracy_difference": mean_ci(paired),
        "interpretation_guidance": (
            "The MLP is a positive-control sanity baseline, not an equivalent SNN comparison. "
            "If the MLP cannot learn XOR, inspect the benchmark pipeline. If MLP succeeds but the "
            "SNN fails, the SNN's fixed random representation/readout learning is the likely "
            "bottleneck. Paired SNN confidence intervals use a normal approximation. Results do "
            "not establish English acquisition, general intelligence, autonomy, or consciousness."
        ),
        "runs": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
