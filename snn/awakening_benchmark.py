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
from evolve_cell import EvolveSpikingContextCell


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
        torch.random.default_generator.manual_seed(seed + 50_000)
        model = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        # Positive control: train on the complete four-example truth table.
        # This intentionally verifies the task/evaluation pipeline, not sample
        # efficiency. A balanced full truth table prevents a weak baseline from
        # being mistaken for evidence that the benchmark itself is broken.
        train_pairs = list(PAIRS)
        x_train = torch.tensor(
            [observation(a, b) for a, b in train_pairs], dtype=torch.float32
        )
        y_train = torch.tensor([a ^ b for a, b in train_pairs], dtype=torch.long)
        epochs = max(300, train_steps // 4)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x_train), y_train)
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
                "positive_control_epochs": epochs,
                "evaluation_trials": len(eval_pairs),
                "heldout_greedy_accuracy": round(accuracy, 4),
                "heldout_accuracy_by_input_pair": per_pair,
                "all_states_finite": bool(torch.isfinite(logits).all().item()),
            }




class TrainableXORSNN(nn.Module):
    """End-to-end trainable version of the experimental spiking cell for XOR.

    Unlike AwakeningAgent's online rule, this diagnostic baseline updates the
    cell and readout with surrogate-gradient backpropagation. Each input is a
    single-step episode, so evaluation after state reset tests whether learned
    parameters encode the XOR mapping rather than relying on carry-over state.
    """
    def __init__(self):
        super().__init__()
        self.cell = EvolveSpikingContextCell(
            input_size=7, hidden_size=32, threshold=0.45,
            leak_fast=0.55, leak_slow=0.90, adaptation_rate=0.01,
        )
        self.readout = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        fast = x.new_zeros(batch, 32)
        slow = x.new_zeros(batch, 32)
        previous = x.new_zeros(batch, 32)
        adaptation = x.new_zeros(batch, 32)
        spikes, _, _, _ = self.cell(x, fast, slow, previous, adaptation)
        return self.readout(spikes), spikes


def run_end_to_end_snn_baseline(
    seed: int, train_steps: int, eval_repeats_per_pair: int
) -> dict[str, Any]:
    """Train both cell and readout, then evaluate every pair from zero state."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 90_000)
        model = TrainableXORSNN()
        # Small recurrent/context projections reduce initial instability.
        with torch.no_grad():
            model.cell.recurrent_current.weight.mul_(0.25)
            model.cell.slow_context.weight.mul_(0.10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        x_train = torch.tensor(
            [observation(a, b) for a, b in PAIRS], dtype=torch.float32
        )
        y_train = torch.tensor([a ^ b for a, b in PAIRS], dtype=torch.long)
        epochs = max(300, train_steps // 2)
        losses = []
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits, _ = model(x_train)
            loss = nn.functional.cross_entropy(logits, y_train)
            if not torch.isfinite(loss):
                break
            loss.backward()
            gradients_finite = all(
                p.grad is None or torch.isfinite(p.grad).all().item()
                for p in model.parameters()
            )
            if not gradients_finite:
                break
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                break
            losses.append(float(loss.detach().item()))

        model.eval()
        eval_pairs = [
            (a, b)
            for a, b in PAIRS
            for _ in range(eval_repeats_per_pair)
        ]
        random.Random(300_000 + seed).shuffle(eval_pairs)
        with torch.no_grad():
            x_eval = torch.tensor(
                [observation(a, b) for a, b in eval_pairs], dtype=torch.float32
            )
            y_eval = torch.tensor([a ^ b for a, b in eval_pairs], dtype=torch.long)
            logits, spikes = model(x_eval)
            predictions = logits.argmax(dim=1)
            per_pair = {}
            for a, b in PAIRS:
                idx = [i for i, pair in enumerate(eval_pairs) if pair == (a, b)]
                per_pair[f"{a}{b}"] = round(
                    float((predictions[idx] == y_eval[idx]).float().mean().item()), 4
                )
            return {
                "seed": seed,
                "condition": "end_to_end_trainable_snn",
                "train_steps": train_steps,
                "training_epochs": epochs,
                "final_training_loss": round(losses[-1], 6) if losses else None,
                "evaluation_trials": len(eval_pairs),
                "heldout_greedy_accuracy": round(
                    float((predictions == y_eval).float().mean().item()), 4
                ),
                "heldout_accuracy_by_input_pair": per_pair,
                "mean_eval_spikes_per_step": round(float(spikes.sum(dim=1).mean().item()), 4),
                "all_states_finite": bool(
                    torch.isfinite(logits).all().item()
                    and all(torch.isfinite(p).all().item() for p in model.parameters())
                ),
            }



def run_online_bptt_snn(seed: int, train_steps: int, eval_repeats_per_pair: int) -> dict[str, Any]:
    """Update cell and readout after each labeled random XOR example."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 120_000)
        model = TrainableXORSNN()
        with torch.no_grad():
            model.cell.recurrent_current.weight.mul_(0.25)
            model.cell.slow_context.weight.mul_(0.10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        rng = random.Random(400_000 + seed)
        correct = 0
        losses = []
        updates_completed = 0
        training_stable = True
        model.train()
        for _ in range(train_steps):
            a, b = rng.randint(0, 1), rng.randint(0, 1)
            x = torch.tensor([observation(a, b)], dtype=torch.float32)
            target = torch.tensor([a ^ b], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss).item():
                training_stable = False
                break
            example_correct = int(logits.argmax(dim=1).item() == int(target.item()))
            loss.backward()
            gradients_finite = all(
                p.grad is None or torch.isfinite(p.grad).all().item()
                for p in model.parameters()
            )
            if not gradients_finite:
                training_stable = False
                optimizer.zero_grad(set_to_none=True)
                break
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                training_stable = False
                break
            losses.append(float(loss.detach().item()))
            correct += example_correct
            updates_completed += 1
        model.eval()
        eval_pairs = [(a, b) for a, b in PAIRS for _ in range(eval_repeats_per_pair)]
        random.Random(500_000 + seed).shuffle(eval_pairs)
        with torch.no_grad():
            x_eval = torch.tensor([observation(a, b) for a, b in eval_pairs], dtype=torch.float32)
            y_eval = torch.tensor([a ^ b for a, b in eval_pairs], dtype=torch.long)
            logits, spikes = model(x_eval)
            predictions = logits.argmax(dim=1)
            per_pair = {}
            for a, b in PAIRS:
                idx = [i for i, pair in enumerate(eval_pairs) if pair == (a, b)]
                per_pair[f"{a}{b}"] = round(float((predictions[idx] == y_eval[idx]).float().mean().item()), 4)
            return {
                "seed": seed,
                "condition": "online_end_to_end_bptt_snn",
                "train_steps": train_steps,
                "updates_completed": updates_completed,
                "mean_online_training_accuracy": round(correct / max(1, updates_completed), 4),
                "final_training_loss": round(losses[-1], 6) if losses else None,
                "training_stable": training_stable,
                "evaluation_trials": len(eval_pairs),
                "heldout_greedy_accuracy": round(float((predictions == y_eval).float().mean().item()), 4),
                "heldout_accuracy_by_input_pair": per_pair,
                "mean_eval_spikes_per_step": round(float(spikes.sum(dim=1).mean().item()), 4),
                "all_states_finite": bool(torch.isfinite(logits).all().item() and all(torch.isfinite(p).all().item() for p in model.parameters())),
            }


def parity3_observation(a: int, b: int, c: int) -> list[float]:
    """Encode three independent bits without exposing their parity target."""
    return [float(a), float(b), float(c), 0.0, 0.0, 0.0, 0.0]

def run_online_parity3_bptt_snn(seed: int, train_steps: int, eval_repeats_per_pattern: int) -> dict[str, Any]:
    """Stress-test online end-to-end learning on three-bit parity."""
    patterns = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 160_000)
        model = TrainableXORSNN()
        with torch.no_grad():
            model.cell.recurrent_current.weight.mul_(0.25)
            model.cell.slow_context.weight.mul_(0.10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        rng = random.Random(600_000 + seed)
        correct = 0
        losses = []
        updates_completed = 0
        training_stable = True
        model.train()
        for _ in range(train_steps):
            a, b, c = (rng.randint(0, 1) for _ in range(3))
            x = torch.tensor([parity3_observation(a, b, c)], dtype=torch.float32)
            target = torch.tensor([a ^ b ^ c], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss).item():
                training_stable = False
                break
            example_correct = int(logits.argmax(dim=1).item() == int(target.item()))
            loss.backward()
            gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
            if not gradients_finite:
                training_stable = False
                optimizer.zero_grad(set_to_none=True)
                break
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                training_stable = False
                break
            losses.append(float(loss.detach().item()))
            correct += example_correct
            updates_completed += 1
        model.eval()
        eval_patterns = [pattern for pattern in patterns for _ in range(eval_repeats_per_pattern)]
        random.Random(700_000 + seed).shuffle(eval_patterns)
        with torch.no_grad():
            x_eval = torch.tensor([parity3_observation(*pattern) for pattern in eval_patterns], dtype=torch.float32)
            y_eval = torch.tensor([a ^ b ^ c for a, b, c in eval_patterns], dtype=torch.long)
            logits, spikes = model(x_eval)
            predictions = logits.argmax(dim=1)
            per_pattern = {}
            for pattern in patterns:
                idx = [i for i, candidate in enumerate(eval_patterns) if candidate == pattern]
                per_pattern[''.join(map(str, pattern))] = round(float((predictions[idx] == y_eval[idx]).float().mean().item()), 4)
            return {
                'seed': seed,
                'condition': 'online_end_to_end_bptt_parity3_snn',
                'train_steps': train_steps,
                'updates_completed': updates_completed,
                'mean_online_training_accuracy': round(correct / max(1, updates_completed), 4),
                'final_training_loss': round(losses[-1], 6) if losses else None,
                'training_stable': training_stable,
                'evaluation_trials': len(eval_patterns),
                'heldout_greedy_accuracy': round(float((predictions == y_eval).float().mean().item()), 4),
                'heldout_accuracy_by_input_pattern': per_pattern,
                'mean_eval_spikes_per_step': round(float(spikes.sum(dim=1).mean().item()), 4),
                'all_states_finite': bool(torch.isfinite(logits).all().item() and all(torch.isfinite(p).all().item() for p in model.parameters())),
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
        results.append(run_end_to_end_snn_baseline(seed, train_steps, eval_repeats_per_pair))
        results.append(run_online_bptt_snn(seed, train_steps, eval_repeats_per_pair))
        results.append(run_online_parity3_bptt_snn(seed, train_steps, eval_repeats_per_pair))

    online = {r["seed"]: r for r in results if r["condition"] == "online_readout_update"}
    frozen = {r["seed"]: r for r in results if r["condition"] == "frozen_readout_control"}
    mlp = {r["seed"]: r for r in results if r["condition"] == "nonlinear_mlp_sanity_baseline"}
    end_to_end = {r["seed"]: r for r in results if r["condition"] == "end_to_end_trainable_snn"}
    online_bptt = {r["seed"]: r for r in results if r["condition"] == "online_end_to_end_bptt_snn"}
    parity3_bptt = {r["seed"]: r for r in results if r["condition"] == "online_end_to_end_bptt_parity3_snn"}
    paired = [online[s]["heldout_greedy_accuracy"] - frozen[s]["heldout_greedy_accuracy"] for s in seeds]

    summary = {}
    for name, by_seed in (
        ("online_readout_update", online),
        ("frozen_readout_control", frozen),
        ("nonlinear_mlp_sanity_baseline", mlp),
        ("end_to_end_trainable_snn", end_to_end),
        ("online_end_to_end_bptt_snn", online_bptt),
        ("online_end_to_end_bptt_parity3_snn", parity3_bptt),
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
        if name in ("online_readout_update", "frozen_readout_control"):
            summary[name]["mean_train_greedy_accuracy"] = round(
                statistics.mean(r["train_greedy_accuracy"] for r in rows), 4
            )
        if name == "online_end_to_end_bptt_snn":
            summary[name]["mean_online_training_accuracy"] = round(
                statistics.mean(r["mean_online_training_accuracy"] for r in rows), 4
            )
            summary[name]["training_stable_runs"] = sum(
                bool(r["training_stable"]) for r in rows
            )

    return {
        "experiment": "awakening_reset_state_generalization_xor_v2",
        "training": (
            "random online XOR stream for SNN; target withheld from stationary input features; "
            "nonlinear MLP positive control and offline end-to-end SNN control trained on the complete XOR truth table; online end-to-end SNN updates after each labeled random example"
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
            "The MLP is a positive-control sanity baseline; the end-to-end SNN updates cell and readout with surrogate-gradient BPTT. "
            "If the MLP cannot learn XOR, inspect the benchmark pipeline. If the end-to-end SNN "
            "succeeds but online Awakening fails, readout-only online learning is the bottleneck. "
            "If only the MLP succeeds, inspect spiking-cell gradients, firing activity and "
            "optimization. The online end-to-end BPTT condition tests whether updating cell and readout on each labeled example closes the gap. The three-bit parity condition is a harder extension test, not evidence of broad reasoning. Training-stability flags expose non-finite updates. Paired readout-control confidence intervals use a normal approximation. Results do not establish English acquisition, general intelligence, autonomy, or consciousness."
        ),
        "runs": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
