"""Minimal developmental runtime for the EvolveAI SNN research prototype.

This is a controlled toy environment, not a claim of consciousness. The agent
receives observations, chooses actions, gets feedback, adapts a small readout
with a reward-modulated local update, and records its history.
"""
from __future__ import annotations

import json
import random
import time
from typing import Any

import torch

from evolve_cell import EvolveSpikingContextCell


STATE_VERSION = 1


class AwakeningAgent:
    """An online SNN agent in a small rule-discovery environment."""

    def __init__(self, seed: int = 42, state: dict[str, Any] | None = None):
        self.seed = int(seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.cell = EvolveSpikingContextCell(input_size=8, hidden_size=32)
        self.readout = torch.nn.Linear(32, 3)
        self.optimizer_lr = 0.02
        self.step_count = 0
        self.last_observation = [0.0] * 8
        self.fast = torch.zeros(1, 32)
        self.slow = torch.zeros(1, 32)
        self.previous_spikes = torch.zeros(1, 32)
        self.adaptation = torch.zeros(1, 32)
        self.recent_bits = [random.randint(0, 1), random.randint(0, 1)]
        self.history: list[dict[str, Any]] = []
        self.action_counts = {"predict_0": 0, "predict_1": 0, "explore": 0}
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_surprise = 0.0
        self.created_at = time.time()
        if state:
            self.load_state(state)

    def _observation(self) -> list[float]:
        a, b = self.recent_bits[-2:]
        phase = (self.step_count % 12) / 11.0
        # The hidden rule is XOR. The agent is not given the target bit.
        return [
            float(a), float(b), float(a ^ b), phase,
            float(self.last_reward), float(self.last_surprise),
            min(1.0, self.step_count / 100.0),
            random.random() * 0.1,
        ]

    def step(self) -> dict[str, Any]:
        obs = self._observation()
        x = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            spikes, self.fast, self.slow, self.adaptation = self.cell(
                x, self.fast, self.slow, self.previous_spikes, self.adaptation
            )
            self.previous_spikes = spikes
            scores = self.readout(spikes)
            # A small, bounded exploration probability prevents fixed behavior.
            explore = random.random() < max(0.08, 0.35 / (1 + self.step_count / 30))
            action = 2 if explore else int(torch.argmax(scores[0, :2]).item())
        target = obs[2]
        if action == 2:
            reward = 0.15 if self.last_surprise > 0.2 else -0.02
            action_name = "explore"
            predicted = None
        else:
            predicted = action
            reward = 1.0 if predicted == int(target) else -0.4
            action_name = f"predict_{action}"
        surprise = 0.0 if predicted is None else float(predicted != int(target))

        # Reward-modulated local readout update. The SNN recurrent cell is not
        # trained here; this deliberately tests a small online-plasticity step.
        with torch.no_grad():
            chosen = action if action < 2 else 2
            self.readout.weight[chosen].add_(
                spikes[0] * (self.optimizer_lr * reward)
            )
            self.readout.weight.clamp_(-2.0, 2.0)
            self.readout.bias[chosen].add_(self.optimizer_lr * reward)
            self.readout.bias.clamp_(-2.0, 2.0)

        # Advance the hidden world. The XOR task provides a measurable signal
        # for whether online adaptation is doing better than chance.
        next_bit = int(self.recent_bits[-2] ^ self.recent_bits[-1])
        self.recent_bits.append(next_bit)
        self.recent_bits = self.recent_bits[-2:]
        self.step_count += 1
        self.last_observation = obs
        self.last_reward = float(reward)
        self.last_surprise = surprise
        self.total_reward += reward
        self.action_counts[action_name] += 1
        record = {
            "step": self.step_count,
            "observation": [round(v, 4) for v in obs],
            "action": action_name,
            "prediction": predicted,
            "target": int(target),
            "correct": None if predicted is None else predicted == int(target),
            "reward": round(float(reward), 4),
            "surprise": surprise,
            "spike_count": int(spikes.sum().item()),
            "mean_fast_state": round(float(self.fast.mean().item()), 5),
            "mean_slow_state": round(float(self.slow.mean().item()), 5),
        }
        self.history.append(record)
        self.history = self.history[-500:]
        return record

    def run(self, steps: int = 1) -> list[dict[str, Any]]:
        if not 1 <= int(steps) <= 100:
            raise ValueError("steps must be between 1 and 100")
        return [self.step() for _ in range(int(steps))]

    def summary(self) -> dict[str, Any]:
        scored = [r for r in self.history if r["correct"] is not None]
        recent = scored[-50:]
        return {
            "step_count": self.step_count,
            "history_records": len(self.history),
            "recent_prediction_accuracy": (
                round(sum(r["correct"] for r in recent) / len(recent), 3)
                if recent else None
            ),
            "total_reward": round(self.total_reward, 3),
            "last_reward": self.last_reward,
            "last_surprise": self.last_surprise,
            "action_counts": self.action_counts,
            "spike_count_last_step": self.history[-1]["spike_count"] if self.history else 0,
            "note": "Toy XOR environment; accuracy measures this task only, not intelligence or consciousness.",
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "seed": self.seed,
            "step_count": self.step_count,
            "recent_bits": self.recent_bits,
            "last_observation": self.last_observation,
            "last_reward": self.last_reward,
            "last_surprise": self.last_surprise,
            "total_reward": self.total_reward,
            "action_counts": self.action_counts,
            "history": self.history,
            "cell_state": {k: v.detach().cpu().tolist() for k, v in self.cell.state_dict().items()},
            "readout_state": {k: v.detach().cpu().tolist() for k, v in self.readout.state_dict().items()},
            "fast": self.fast.tolist(),
            "slow": self.slow.tolist(),
            "previous_spikes": self.previous_spikes.tolist(),
            "adaptation": self.adaptation.tolist(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("state_version") != STATE_VERSION:
            raise ValueError("Unsupported awakening state version")
        self.seed = int(state.get("seed", self.seed))
        self.step_count = int(state.get("step_count", 0))
        self.recent_bits = [int(x) for x in state.get("recent_bits", [0, 1])][-2:]
        self.last_observation = list(state.get("last_observation", [0.0] * 8))
        self.last_reward = float(state.get("last_reward", 0.0))
        self.last_surprise = float(state.get("last_surprise", 0.0))
        self.total_reward = float(state.get("total_reward", 0.0))
        self.action_counts = dict(state.get("action_counts", self.action_counts))
        self.history = list(state.get("history", []))[-500:]
        if "cell_state" in state:
            self.cell.load_state_dict({k: torch.tensor(v) for k, v in state["cell_state"].items()})
        if "readout_state" in state:
            self.readout.load_state_dict({k: torch.tensor(v) for k, v in state["readout_state"].items()})
        for name in ("fast", "slow", "previous_spikes", "adaptation"):
            if name in state:
                setattr(self, name, torch.tensor(state[name], dtype=torch.float32))


def export_state_json(agent: AwakeningAgent) -> str:
    return json.dumps(agent.export_state(), indent=2)
