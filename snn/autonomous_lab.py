"""Bounded trial-and-error controller for EvolveAI SNN experiments.

Only reviewed configurations are proposed. Runs can use CUDA when available,
and optional checkpoints preserve the best model state for later evaluation.
"""
from __future__ import annotations
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import torch
from train import train


@dataclass(frozen=True)
class TrialConfig:
    architecture: str
    hidden_size: int
    learning_rate: float
    seed: int


def propose_trials(max_trials: int = 4, seed: int = 42) -> list[TrialConfig]:
    """Small, deterministic search over a reviewed configuration space."""
    candidates = [
        TrialConfig("evolve_context", 64, 0.001, seed),
        TrialConfig("baseline", 64, 0.001, seed),
        TrialConfig("evolve_context", 32, 0.001, seed),
        TrialConfig("evolve_context", 64, 0.0005, seed),
        TrialConfig("baseline", 32, 0.001, seed),
        TrialConfig("baseline", 64, 0.0005, seed),
    ]
    return candidates[:max(1, min(int(max_trials), len(candidates)))]


def run_trials(corpus: str, max_trials: int = 4, steps: int = 40,
               block_size: int = 32, batch_size: int = 4, seed: int = 42,
               progress=None, device: str = "auto",
               checkpoint_path: str | None = None) -> dict:
    if len(corpus.encode("utf-8")) < 100:
        raise ValueError("Corpus must contain at least 100 UTF-8 bytes.")
    if not 1 <= steps <= 300:
        raise ValueError("steps must be between 1 and 300.")
    if not 1 <= max_trials <= 6:
        raise ValueError("max_trials must be between 1 and 6.")
    configs = propose_trials(max_trials, seed)
    records = []
    best_loss = float("inf")
    started = time.time()
    for index, config in enumerate(configs, start=1):
        trial_start = time.time()
        try:
            result = train(
                corpus, steps=steps, block_size=block_size, batch_size=batch_size,
                hidden_size=config.hidden_size, embedding_dim=24,
                learning_rate=config.learning_rate, seed=config.seed,
                architecture=config.architecture, device=device,
            )
            final = result["history"][-1]
            record = {
                **asdict(config),
                "trial": index,
                "status": "completed",
                "steps": steps,
                "train_loss": final["train_loss"],
                "validation_loss": final["val_loss"],
                "validation_perplexity": final["val_perplexity"],
                "elapsed_seconds": round(time.time() - trial_start, 3),
                "device": result["device"],
                "config": result["config"],
            }
            if checkpoint_path and record["validation_loss"] < best_loss:
                best_loss = record["validation_loss"]
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": {k: v.detach().cpu() for k, v in result["model"].state_dict().items()},
                    "config": result["config"],
                    "validation_loss": record["validation_loss"],
                    "trial": index,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }, path)
                record["best_checkpoint_saved"] = True
            else:
                best_loss = min(best_loss, record["validation_loss"])
        except Exception as exc:
            record = {
                **asdict(config), "trial": index, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - trial_start, 3),
            }
        records.append(record)
        if progress:
            progress(index, len(configs), record)
    completed = [r for r in records if r["status"] == "completed"]
    ranked = sorted(completed, key=lambda r: r["validation_loss"])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "minimize final validation byte-level cross-entropy",
        "corpus_bytes": len(corpus.encode("utf-8")),
        "device_requested": device,
        "device_used": next((r.get("device") for r in completed), None),
        "trials_requested": len(configs),
        "trials_completed": len(completed),
        "elapsed_seconds": round(time.time() - started, 3),
        "best_trial": ranked[0] if ranked else None,
        "checkpoint_path": checkpoint_path if checkpoint_path and ranked else None,
        "leaderboard": ranked,
        "all_trials": records,
        "limitations": [
            "Short runs and a single seed are screening experiments, not a robust model selection protocol.",
            "Lower validation loss does not prove coherent dialogue, reasoning, consciousness, or general intelligence.",
            "Use multiple seeds and an untouched test set before selecting a model.",
            "The current candidate list is fixed and does not invent new learning rules.",
        ],
    }


def export_json(result: dict) -> str:
    return json.dumps(result, indent=2)
