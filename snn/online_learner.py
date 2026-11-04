"""Online SNN learner adapter for the Parallel Data Engine.

This is a bounded online fine-tuning bridge for the existing recurrent LIF
byte-prediction model. It is not a semantic reasoning objective: it learns
next-byte continuation from selected records. Keep this distinction visible
in reports and user-facing interfaces.
"""
from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
import random
import threading
from typing import Any

import torch
from torch import nn

try:
    from .model import RecurrentLIFLanguageModel
except ImportError:  # supports running from the snn directory in Colab
    from model import RecurrentLIFLanguageModel


class OnlineSNNLearner:
    """Bounded replay-buffer trainer callable by ParallelDataEngine."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "auto",
        hidden_size: int = 128,
        embedding_dim: int = 64,
        block_size: int = 128,
        batch_size: int = 8,
        learning_rate: float = 0.0003,
        replay_capacity: int = 4096,
        train_every: int = 8,
        save_every: int = 25,
        seed: int = 41,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. In Colab select "
                "Runtime > Change runtime type > T4 GPU, then rerun all cells."
            )
        self.device = torch.device(device)
        self.block_size = max(8, int(block_size))
        self.batch_size = max(1, int(batch_size))
        self.train_every = max(1, int(train_every))
        self.save_every = max(1, int(save_every))
        self.replay = deque(maxlen=max(self.batch_size, int(replay_capacity)))
        self.rng = random.Random(seed)
        self.lock = threading.RLock()
        self.accepted_records = 0
        self.updates = 0
        self.last_loss: float | None = None
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = RecurrentLIFLanguageModel(
            embedding_dim=embedding_dim, hidden_size=hidden_size
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        if self.checkpoint_path.is_file():
            self._restore()

    def _restore(self) -> None:
        payload = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        config = payload.get("config", {})
        expected = {
            "hidden_size": self.model.hidden_size,
            "embedding_dim": self.model.embedding.embedding_dim,
            "block_size": self.block_size,
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Online SNN checkpoint configuration mismatch: {mismatches}")
        self.model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self.accepted_records = int(payload.get("accepted_records", 0))
        self.updates = int(payload.get("updates", 0))
        self.last_loss = payload.get("last_loss")

    def _record(self, record: Any, evaluation: dict[str, Any]) -> None:
        decision = evaluation.get("decision")
        if decision not in {"learn_now", "investigate", "retain"}:
            return
        raw = str(record.content or "").encode("utf-8")
        if len(raw) < 2:
            return
        # Keep only a bounded byte sequence per record.
        self.replay.append(raw[: self.block_size + 1])
        self.accepted_records += 1
        if len(self.replay) >= self.batch_size and self.accepted_records % self.train_every == 0:
            self._train_step()
        if self.updates and self.updates % self.save_every == 0:
            self.save()

    def _train_step(self) -> float:
        samples = self.rng.sample(list(self.replay), self.batch_size)
        sequences = [torch.tensor(list(item), dtype=torch.long) for item in samples]
        max_len = min(self.block_size + 1, max(len(item) for item in sequences))
        x = torch.zeros((len(sequences), max_len - 1), dtype=torch.long)
        y = torch.full((len(sequences), max_len - 1), -100, dtype=torch.long)
        for index, seq in enumerate(sequences):
            seq = seq[:max_len]
            length = len(seq) - 1
            if length > 0:
                x[index, :length] = seq[:-1]
                y[index, :length] = seq[1:]

        x = x.to(self.device)
        y = y.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        self.last_loss = float(loss.detach().cpu())
        return self.last_loss

    async def __call__(self, record: Any, evaluation: dict[str, Any]) -> None:
        # Keep GPU work off the asyncio event loop so ingestion can continue
        # until the bounded queue applies backpressure.
        await asyncio.to_thread(self.add, record, evaluation)

    def add(self, record: Any, evaluation: dict[str, Any]) -> None:
        with self.lock:
            self._record(record, evaluation)

    def save(self) -> None:
        with self.lock:
            payload = {
                "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "optimizer_state": self.optimizer.state_dict(),
                "config": {
                    "hidden_size": self.model.hidden_size,
                    "embedding_dim": self.model.embedding.embedding_dim,
                    "block_size": self.block_size,
                },
                "accepted_records": self.accepted_records,
                "updates": self.updates,
                "last_loss": self.last_loss,
                "objective": "byte-level next-byte prediction; not semantic reasoning",
            }
            tmp = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
            torch.save(payload, tmp)
            tmp.replace(self.checkpoint_path)

    def status(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "accepted_records": self.accepted_records,
            "replay_records": len(self.replay),
            "replay_capacity": self.replay.maxlen,
            "updates": self.updates,
            "last_loss": self.last_loss,
            "checkpoint": str(self.checkpoint_path),
            "learning_objective": "byte-level next-byte prediction",
        }
