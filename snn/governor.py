"""Governed autonomous learning loop for EvolveAI.

The governor controls approved dataset provenance, corpus size, experiment
count, training steps, architectures, and durable experiment history. It does
not browse arbitrary URLs, execute downloaded code, or grant the model shell access.
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autonomous_lab import run_trials

ALLOWED_DATASETS = {
    "roneneldan/TinyStories": {
        "host": "huggingface.co",
        "purpose": "Small-story language-pattern research",
        "license_note": "Review the current upstream dataset card before use.",
    },
}
ALLOWED_ARCHITECTURES = {"baseline", "evolve_context"}
MAX_CORPUS_BYTES = 10 * 1024 * 1024
MAX_TRIALS_PER_CYCLE = 6
MAX_STEPS_PER_TRIAL = 300
MAX_CYCLES = 20
STATE_VERSION = 1


def validate_dataset_manifest(manifest: dict[str, Any]) -> None:
    """Reject datasets not in the explicit allowlist or with unexpected provenance."""
    dataset_id = manifest.get("dataset_id")
    if dataset_id not in ALLOWED_DATASETS:
        raise ValueError("Governor blocked this dataset: ID is not allowlisted.")
    expected_prefix = "https://huggingface.co/datasets/"
    url = manifest.get("dataset_url", "")
    if not isinstance(url, str) or not url.startswith(expected_prefix):
        raise ValueError("Governor blocked this dataset: unexpected source URL.")
    if url.rstrip("/").split("/")[-1] != dataset_id.split("/")[-1]:
        raise ValueError("Governor blocked this dataset: URL and dataset ID disagree.")


def load_state(state_path: str | Path) -> dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        return {"state_version": STATE_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "cycles": [], "best_validation_loss": None}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("state_version") != STATE_VERSION or not isinstance(state.get("cycles"), list):
        raise ValueError("Governor state file has an unsupported or invalid schema.")
    return state


def save_state(state: dict[str, Any], state_path: str | Path) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp_path.replace(path)


def run_governed_cycle(
    corpus: str,
    manifest: dict[str, Any],
    state_path: str | Path,
    checkpoint_path: str | Path,
    max_trials: int = 3,
    steps: int = 60,
    block_size: int = 32,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Run one bounded cycle and persist its report plus a compact history."""
    validate_dataset_manifest(manifest)
    corpus_payload = corpus.encode("utf-8")
    corpus_bytes = len(corpus_payload)
    supplied_hash = manifest.get("corpus_sha256")
    actual_hash = hashlib.sha256(corpus_payload).hexdigest()
    if not supplied_hash or supplied_hash != actual_hash:
        raise ValueError("Governor blocked this corpus: its hash does not match the provenance manifest.")
    if corpus_bytes < 100:
        raise ValueError("Governor requires at least 100 bytes of corpus.")
    if corpus_bytes > MAX_CORPUS_BYTES:
        raise ValueError(f"Governor corpus limit is {MAX_CORPUS_BYTES // (1024 * 1024)} MiB.")
    if not 1 <= int(max_trials) <= MAX_TRIALS_PER_CYCLE:
        raise ValueError(f"Governor permits 1–{MAX_TRIALS_PER_CYCLE} trials per cycle.")
    if not 1 <= int(steps) <= MAX_STEPS_PER_TRIAL:
        raise ValueError(f"Governor permits 1–{MAX_STEPS_PER_TRIAL} steps per trial.")
    if not 4 <= int(block_size) <= 96:
        raise ValueError("Governor permits block sizes from 4 to 96.")
    if not 1 <= int(batch_size) <= 32:
        raise ValueError("Governor permits batch sizes from 1 to 32.")

    state = load_state(state_path)
    if len(state["cycles"]) >= MAX_CYCLES:
        raise RuntimeError(f"Governor reached the {MAX_CYCLES}-cycle session history limit.")
    previous_best = state.get("best_validation_loss")
    previous_best_trial = state.get("best_trial")
    cycle_seed = int(seed) + len(state["cycles"]) * 1009
    final_checkpoint = Path(checkpoint_path)
    same_corpus_as_previous = bool(
        state["cycles"]
        and state["cycles"][-1].get("corpus_sha256") == manifest.get("corpus_sha256")
    )
    warm_start_path = str(final_checkpoint) if same_corpus_as_previous and final_checkpoint.is_file() else None
    candidate_checkpoint = final_checkpoint.with_name(final_checkpoint.stem + "_candidate" + final_checkpoint.suffix)
    report = run_trials(
        corpus=corpus,
        max_trials=int(max_trials),
        steps=int(steps),
        block_size=int(block_size),
        batch_size=int(batch_size),
        seed=cycle_seed,
        device=device,
        checkpoint_path=str(candidate_checkpoint),
        previous_best=previous_best_trial,
        warm_start_checkpoint=warm_start_path,
    )
    best = report.get("best_trial")
    if best and (previous_best is None or best["validation_loss"] < previous_best) and candidate_checkpoint.exists():
        candidate_checkpoint.replace(final_checkpoint)
        report["checkpoint_path"] = str(final_checkpoint)
    else:
        if candidate_checkpoint.exists():
            candidate_checkpoint.unlink()
        report["checkpoint_path"] = str(final_checkpoint) if final_checkpoint.exists() else None
    report["governor"] = {
        "approved": True,
        "search_strategy": report.get("search_strategy"),
        "cycle_seed": cycle_seed,
        "warm_start_enabled": warm_start_path is not None,
        "warm_start_note": "Compatible candidates continue from the prior best checkpoint only when the corpus hash is unchanged.",
        "dataset_id": manifest["dataset_id"],
        "corpus_sha256": manifest.get("corpus_sha256"),
        "corpus_limit_bytes": MAX_CORPUS_BYTES,
        "max_trials_per_cycle": MAX_TRIALS_PER_CYCLE,
        "max_steps_per_trial": MAX_STEPS_PER_TRIAL,
        "internet_policy": "Allowlisted Hugging Face dataset provenance only; no arbitrary URL fetching or downloaded code execution.",
    }
    if best and (previous_best is None or best["validation_loss"] < previous_best):
        state["best_validation_loss"] = best["validation_loss"]
        state["best_trial"] = best
    state["cycles"].append({
        "cycle": len(state["cycles"]) + 1,
        "created_at_utc": report["created_at_utc"],
        "dataset_id": manifest["dataset_id"],
        "corpus_sha256": manifest.get("corpus_sha256"),
        "trials_completed": report["trials_completed"],
        "best_validation_loss": best["validation_loss"] if best else None,
        "device_used": report.get("device_used"),
        "elapsed_seconds": report["elapsed_seconds"],
        "checkpoint_path": report.get("checkpoint_path"),
    })
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_state(state, state_path)
    report["governor"]["cycle_number"] = len(state["cycles"])
    report["governor"]["historical_best_validation_loss"] = state.get("best_validation_loss")
    return report
