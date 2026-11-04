"""Integrated EvolveAI runtime: parallel data -> evaluation -> SNN learning.

Connectors are registered by the caller. This keeps the engine source-agnostic
and makes the system portable between Colab and a persistent GPU host.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .experience_evaluator import ExperienceEvaluator
from .parallel_data_engine import (
    ConnectorRegistry, ExperienceStore, ParallelDataEngine, SourceConnector,
)


class EvolveAISystem:
    def __init__(
        self,
        *,
        state_dir: str = "evolveai_state",
        device: str = "auto",
        queue_size: int = 256,
        evaluation_workers: int = 2,
        replay_capacity: int = 4096,
        batch_size: int = 8,
        train_every: int = 8,
        hidden_size: int = 128,
        embedding_dim: int = 64,
        block_size: int = 128,
    ) -> None:
        root = Path(state_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.registry = ConnectorRegistry()
        # The current SNN objective is general byte-level language modeling, so non-duplicate text remains trainable even when lexical novelty is low. Replace this threshold with a learned task-aware policy as that capability is validated.\n        self.evaluator = ExperienceEvaluator(learn_threshold=0.15)
        self.store = ExperienceStore(str(root / "experiences.sqlite3"))

        # Import PyTorch learner lazily so ingestion-only simulations can run
        # without initializing CUDA or allocating model memory.
        from snn.online_learner import OnlineSNNLearner

        self.learner = OnlineSNNLearner(
            str(root / "snn_online_checkpoint.pt"),
            device=device,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            block_size=block_size,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            train_every=train_every,
        )
        self.engine = ParallelDataEngine(
            self.registry,
            evaluator=self.evaluator,
            store=self.store,
            learner=self.learner,
            queue_size=queue_size,
            workers=evaluation_workers,
        )

    def register(self, connector: SourceConnector) -> None:
        self.registry.register(connector)

    async def run(self, *, source_ids: Iterable[str] | None = None,
                  max_records: int | None = None) -> dict:
        return await self.engine.run(source_ids=source_ids, max_records=max_records)

    def status(self) -> dict:
        return {
            "system": "EvolveAI integrated runtime",
            "registered_sources": self.registry.ids(),
            "pipeline": self.engine.status(),
            "learner": self.learner.status(),
            "capability_boundary": (
                "The SNN is trained on byte-level next-byte prediction. "
                "Evaluation, memory and source orchestration do not imply "
                "semantic reasoning or verified factual understanding."
            ),
        }

    def save(self) -> None:
        self.learner.save()
