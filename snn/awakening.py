"""Controlled developmental SNN experiment with stable dynamics and honest metrics.

This toy environment tests online XOR learning. It is not a claim of
consciousness or open-ended intelligence.
"""
from __future__ import annotations

import json
import random
import time
from typing import Any

import torch

from evolve_cell import EvolveSpikingContextCell

STATE_VERSION = 2


class AwakeningAgent:
    """Online SNN agent; targets are kept out of the observation vector."""

    def __init__(self, seed: int = 42, state: dict[str, Any] | None = None):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        self.cell = EvolveSpikingContextCell(
            input_size=7, hidden_size=32, threshold=0.45,
            leak_fast=0.55, leak_slow=0.90, adaptation_rate=0.01,
        )
        # Conservative initialization helps avoid runaway recurrent currents.
        with torch.no_grad():
            for p in self.cell.parameters():
                p.clamp_(-0.15, 0.15)
            self.cell.recurrent_current.weight.mul_(0.25)
            self.cell.slow_context.weight.mul_(0.10)
        self.readout = torch.nn.Linear(32, 2)
        with torch.no_grad():
            self.readout.weight.mul_(0.1)
            self.readout.bias.zero_()
        self.learning_rate = 0.03
        self.step_count = 0
        self.last_observation = [0.0] * 7
        self.fast = torch.zeros(1, 32)
        self.slow = torch.zeros(1, 32)
        self.previous_spikes = torch.zeros(1, 32)
        self.adaptation = torch.zeros(1, 32)
        self.last_reward = 0.0
        self.last_surprise = 0.0
        self.total_reward = 0.0
        self.history: list[dict[str, Any]] = []
        self.action_counts = {"predict_0": 0, "predict_1": 0, "explore": 0}
        self.created_at = time.time()
        self.fault_count = 0
        if state:
            self.load_state(state)

    def _observation(self, a: int, b: int) -> list[float]:
        # Crucially, XOR target is not present in the observation.
        return [
            float(a), float(b),
            min(1.0, self.step_count / 100.0),
            max(-1.0, min(1.0, self.last_reward)),
            self.last_surprise,
            float((self.step_count % 2)),
            self.rng.random() * 0.02,
        ]

    def _reset_dynamics(self) -> None:
        self.fast.zero_()
        self.slow.zero_()
        self.previous_spikes.zero_()
        self.adaptation.zero_()
        self.fault_count += 1

    def step(self) -> dict[str, Any]:
        # Balanced, independently sampled input pairs avoid a constant target.
        a = self.rng.randint(0, 1)
        b = self.rng.randint(0, 1)
        target = a ^ b
        obs = self._observation(a, b)
        x = torch.tensor([obs], dtype=torch.float32)

        with torch.no_grad():
            # Guard against non-finite recurrent state before it contaminates
            # weights or subsequent observations.
            tensors = (self.fast, self.slow, self.previous_spikes, self.adaptation)
            if not all(torch.isfinite(t).all().item() for t in tensors):
                self._reset_dynamics()
            spikes, fast, slow, adaptation = self.cell(
                x, self.fast, self.slow, self.previous_spikes, self.adaptation
            )
            if not all(torch.isfinite(t).all().item() for t in (spikes, fast, slow, adaptation)):
                self._reset_dynamics()
                spikes, fast, slow, adaptation = self.cell(
                    x, self.fast, self.slow, self.previous_spikes, self.adaptation
                )
            # Clamp state values to stop recurrent runaway; track spikes honestly.
            self.fast = torch.nan_to_num(fast, nan=0.0, posinf=2.0, neginf=-2.0).clamp(-2, 2)
            self.slow = torch.nan_to_num(slow, nan=0.0, posinf=2.0, neginf=-2.0).clamp(-2, 2)
            self.adaptation = torch.nan_to_num(adaptation, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            self.previous_spikes = torch.nan_to_num(spikes, nan=0.0).clamp(0, 1)
            scores = self.readout(self.previous_spikes)
            explore = self.rng.random() < 0.08
            action = self.rng.randint(0, 1) if explore else int(scores.argmax(dim=1).item())

        predicted = action
        correct = predicted == target
        reward = 1.0 if correct else -1.0
        surprise = 0.0 if correct else 1.0

        # Prediction-error update: reinforce correct output and suppress the
        # incorrect output. This is an online readout rule, not SNN self-discovery.
        with torch.no_grad():
            target_distribution = torch.zeros_like(scores[0])
            target_distribution[target] = 1.0
            delta = self.learning_rate * (target_distribution - torch.softmax(scores[0], dim=0))
            self.readout.weight.add_(delta[:, None] * self.previous_spikes)
            self.readout.bias.add_(delta)
            self.readout.weight.clamp_(-1.0, 1.0)
            self.readout.bias.clamp_(-1.0, 1.0)

        self.step_count += 1
        self.last_observation = obs
        self.last_reward = reward
        self.last_surprise = surprise
        self.total_reward += reward
        action_name = f"predict_{predicted}"
        self.action_counts[action_name] += 1
        record = {
            "step": self.step_count,
            "input_a": a,
            "input_b": b,
            "action": action_name,
            "prediction": predicted,
            "target": target,
            "correct": correct,
            "reward": reward,
            "surprise": surprise,
            "spike_count": int(self.previous_spikes.sum().item()),
            "mean_fast_state": round(float(self.fast.mean().item()), 5),
            "mean_slow_state": round(float(self.slow.mean().item()), 5),
            "finite_state": bool(all(torch.isfinite(t).all().item() for t in (self.fast, self.slow, self.adaptation))),
        }
        self.history.append(record)
        self.history = self.history[-500:]
        return record

    def run(self, steps: int = 1) -> list[dict[str, Any]]:
        if not 1 <= int(steps) <= 100:
            raise ValueError("steps must be between 1 and 100")
        return [self.step() for _ in range(int(steps))]

    def summary(self) -> dict[str, Any]:
        recent = self.history[-100:]
        return {
            "step_count": self.step_count,
            "history_records": len(self.history),
            "recent_prediction_accuracy": round(sum(r["correct"] for r in recent) / len(recent), 3) if recent else None,
            "total_reward": round(self.total_reward, 3),
            "last_reward": self.last_reward,
            "last_surprise": self.last_surprise,
            "action_counts": dict(self.action_counts),
            "spike_count_last_step": self.history[-1]["spike_count"] if self.history else 0,
            "mean_spike_count_recent": round(sum(r["spike_count"] for r in recent) / len(recent), 3) if recent else 0,
            "non_finite_resets": self.fault_count,
            "note": "Balanced XOR toy task; accuracy is only a task metric, not intelligence or consciousness.",
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION, "seed": self.seed,
            "step_count": self.step_count, "last_observation": self.last_observation,
            "last_reward": self.last_reward, "last_surprise": self.last_surprise,
            "total_reward": self.total_reward, "action_counts": self.action_counts,
            "history": self.history, "fault_count": self.fault_count,
            "rng_state": self.rng.getstate(),
            "cell_state": {k: v.detach().cpu().tolist() for k, v in self.cell.state_dict().items()},
            "readout_state": {k: v.detach().cpu().tolist() for k, v in self.readout.state_dict().items()},
            "fast": self.fast.tolist(), "slow": self.slow.tolist(),
            "previous_spikes": self.previous_spikes.tolist(), "adaptation": self.adaptation.tolist(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("state_version") != STATE_VERSION:
            raise ValueError("This state file uses an older format. Reset the agent or export a new version-2 state.")
        self.seed = int(state.get("seed", self.seed))
        self.step_count = int(state.get("step_count", 0))
        self.last_observation = list(state.get("last_observation", [0.0] * 7))
        self.last_reward = float(state.get("last_reward", 0.0))
        self.last_surprise = float(state.get("last_surprise", 0.0))
        self.total_reward = float(state.get("total_reward", 0.0))
        self.action_counts = dict(state.get("action_counts", self.action_counts))
        self.history = list(state.get("history", []))[-500:]
        self.fault_count = int(state.get("fault_count", 0))
        if "rng_state" in state:
            self.rng.setstate(_tuple_tree(state["rng_state"]))
        self.cell.load_state_dict({k: torch.tensor(v) for k, v in state["cell_state"].items()})
        self.readout.load_state_dict({k: torch.tensor(v) for k, v in state["readout_state"].items()})
        for name in ("fast", "slow", "previous_spikes", "adaptation"):
            setattr(self, name, torch.tensor(state[name], dtype=torch.float32))


def _tuple_tree(value):
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def export_state_json(agent: AwakeningAgent) -> str:
    return json.dumps(agent.export_state(), indent=2)
