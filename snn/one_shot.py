"""Single-invocation, bounded EvolveAI training run.

Runs exactly one model training job. It does not launch an experiment sweep,
loop over cycles, or modify governor_state.json / the historical champion.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from governor import validate_dataset_manifest, MAX_CORPUS_BYTES
from train import train


def run_one_shot(
    corpus: str,
    manifest: dict,
    output_dir: str | Path,
    *,
    steps: int = 600,
    block_size: int = 32,
    batch_size: int = 8,
    hidden_size: int = 64,
    embedding_dim: int = 24,
    learning_rate: float = 0.002,
    seed: int = 42,
    architecture: str = "baseline",
    device: str = "cuda",
) -> dict:
    """Validate provenance, train one fresh model, and save an isolated result."""
    validate_dataset_manifest(manifest)
    payload = corpus.encode("utf-8")
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != manifest.get("corpus_sha256"):
        raise ValueError("Corpus hash does not match the provenance manifest.")
    if len(payload) < 100 or len(payload) > MAX_CORPUS_BYTES:
        raise ValueError("Corpus size is outside the governor's approved limits.")
    if not 1 <= int(steps) <= 5000:
        raise ValueError("One-shot training steps must be between 1 and 5000.")
    if architecture not in {"baseline", "evolve_context"}:
        raise ValueError("Architecture is not approved.")
    if int(hidden_size) not in {32, 64, 96, 128}:
        raise ValueError("Hidden size is not approved.")
    if not 0.0001 <= float(learning_rate) <= 0.005:
        raise ValueError("Learning rate is outside the approved range.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "one_shot_checkpoint.pt"
    report_path = output / "one_shot_report.json"

    started = time.time()
    result = train(
        corpus,
        steps=int(steps),
        block_size=int(block_size),
        batch_size=int(batch_size),
        hidden_size=int(hidden_size),
        embedding_dim=int(embedding_dim),
        learning_rate=float(learning_rate),
        seed=int(seed),
        architecture=architecture,
        device=device,
        initial_state_dict=None,
    )
    final = result["history"][-1]
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in result["model"].state_dict().items()},
        "config": result["config"],
        "validation_loss": final["val_loss"],
        "validation_perplexity": final["val_perplexity"],
        "corpus_sha256": actual_hash,
        "validation_protocol": "deterministic evenly-spaced validation windows v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temp_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temp_checkpoint)
    temp_checkpoint.replace(checkpoint_path)

    report = {
        "mode": "one_shot_single_training_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": result["device"],
        "corpus_bytes": len(payload),
        "corpus_sha256": actual_hash,
        "dataset_id": manifest["dataset_id"],
        "train_bytes": result["train_bytes"],
        "validation_bytes": result["validation_bytes"],
        "test_bytes_reserved_not_evaluated": result["test_bytes"],
        "validation_protocol": "deterministic evenly-spaced validation windows v3",
        "training_config": result["config"],
        "steps_completed": int(steps),
        "final_train_loss": final["train_loss"],
        "final_validation_loss": final["val_loss"],
        "final_validation_perplexity": final["val_perplexity"],
        "training_history": result["history"],
        "elapsed_seconds": round(time.time() - started, 3),
        "checkpoint_path": str(checkpoint_path),
        "limitations": [
            "Exactly one fresh model was trained; no hyperparameter sweep or repeated cycles were run.",
            "The reserved final corpus segment was not evaluated.",
            "Validation loss is used for monitoring, not an independent generalization claim.",
            "This run does not modify the governed experiment history or historical champion checkpoint.",
        ],
    }
    temp_report = report_path.with_suffix(".json.tmp")
    temp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temp_report.replace(report_path)
    return report
