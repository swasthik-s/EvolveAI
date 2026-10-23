"""Bounded trial-and-error controller for EvolveAI SNN experiments.

The controller only varies allowlisted configurations, uses a fixed corpus and
seed policy, records every outcome, and ranks trials by validation loss. This is
automated experimentation, not proof of better language quality.
"""
from __future__ import annotations
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
               progress=None) -> dict:
    if len(corpus.encode("utf-8")) < 100:
        raise ValueError("Corpus must contain at least 100 UTF-8 bytes.")
    if not 1 <= steps <= 300:
        raise ValueError("steps must be between 1 and 300.")
    configs = propose_trials(max_trials, seed)
    records = []
    started = time.time()
    for index, config in enumerate(configs, start=1):
        trial_start = time.time()
        try:
            result = train(
                corpus, steps=steps, block_size=block_size, batch_size=batch_size,
                hidden_size=config.hidden_size, embedding_dim=24,
                learning_rate=config.learning_rate, seed=config.seed,
                architecture=config.architecture,
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
                "config": result["config"],
            }
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
        "trials_requested": len(configs),
        "trials_completed": len(completed),
        "elapsed_seconds": round(time.time() - started, 3),
        "best_trial": ranked[0] if ranked else None,
        "leaderboard": ranked,
        "all_trials": records,
        "limitations": [
            "One seed and short runs are only a screening experiment.",
            "Lower validation loss does not prove coherent dialogue or reasoning.",
            "Use multiple seeds and an untouched test set before selecting a production model.",
        ],
    }


def export_json(result: dict) -> str:
    return json.dumps(result, indent=2)
