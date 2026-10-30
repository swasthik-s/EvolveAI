"""One-invocation multi-source streaming training run."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import torch

from streaming_gatekeeper import StreamingGatekeeper, SOURCE_REGISTRY
from streaming_train import train_streaming


def run_streaming_one_shot(
    output_dir,
    *,
    source_ids=None,
    steps=3000,
    block_size=32,
    batch_size=8,
    hidden_size=64,
    embedding_dim=24,
    learning_rate=0.002,
    seed=42,
    architecture="baseline",
    device="cuda",
    eval_interval=100,
    patience=5,
    min_delta=0.002,
):
    source_ids = source_ids or ["tinystories", "wikitext", "ag_news", "gsm8k"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "streaming_one_shot_report.json"
    checkpoint_path = output / "streaming_one_shot_checkpoint.pt"

    print("\n[EVOLVEAI] ONE-SHOT MULTI-SOURCE STREAMING TRAINING")
    print("[EVOLVEAI] No complete corpus download; bounded in-memory buffers only.")
    started = time.time()
    gatekeeper = StreamingGatekeeper(source_ids, seed=seed)
    gatekeeper.warm_buffers()

    def progress(record):
        print(
            f"\n[TRAIN] step={record['step']}/{record['max_steps']} "
            f"train_loss={record['train_loss']:.4f} "
            f"val_loss={record['validation_loss']:.4f} "
            f"val_ppl={record['validation_perplexity']:.2f} "
            f"best_step={record['best_step']} "
            f"stale={record['stale_evaluations']}/{patience}"
        )
        print(f"[GATEKEEPER] latest={record['latest_source']} "
              f"domain={record['latest_domain']} "
              f"sample={record['latest_sample']!r}")
        status = record["gatekeeper"]
        print(f"[GATEKEEPER] admitted={status['records_admitted']} "
              f"rejected={status['records_rejected']} "
              f"buffers=train:{status['train_buffer_records']} "
              f"validation:{status['validation_buffer_records']}")
        for sid, counts in status["sources"].items():
            print(f"  [SOURCE] {sid}: {counts}")

    result = train_streaming(
        gatekeeper, steps=steps, block_size=block_size, batch_size=batch_size,
        hidden_size=hidden_size, embedding_dim=embedding_dim,
        learning_rate=learning_rate, seed=seed, architecture=architecture,
        device=device, eval_interval=eval_interval, patience=patience,
        min_delta=min_delta, progress=progress,
    )
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in result["model"].state_dict().items()},
        "config": result["config"],
        "best_validation_loss": result["best_validation_loss"],
        "best_step": result["best_step"],
        "sources": result["gatekeeper"]["sources"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(checkpoint_path)

    report = {
        "mode": "one_shot_multi_source_streaming",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": result["device"],
        "training_config": result["config"],
        "steps_completed": result["steps_completed"],
        "stopped_early": result["stopped_early"],
        "best_step": result["best_step"],
        "best_validation_loss": result["best_validation_loss"],
        "training_history": result["history"],
        "gatekeeper": result["gatekeeper"],
        "sources": {
            sid: {"dataset": SOURCE_REGISTRY[sid]["path"],
                  "url": SOURCE_REGISTRY[sid]["url"],
                  "domain": SOURCE_REGISTRY[sid]["domain"],
                  "license_note": SOURCE_REGISTRY[sid]["license_note"]}
            for sid in source_ids
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "checkpoint_path": str(checkpoint_path),
        "limitations": [
            "Sources are read with Hugging Face streaming=True; no full corpus is downloaded or persisted.",
            "Only bounded in-memory train and validation replay buffers are retained.",
            "The SNN is trained on byte-level next-token prediction; it is not yet a structured reasoning system.",
            "The validation buffer is a small online sample, not a fully independent benchmark.",
            "Review each upstream dataset card and content rights before enabling a source.",
        ],
    }
    tmp_report = report_path.with_suffix(".json.tmp")
    tmp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp_report.replace(report_path)
    print("\n[EVOLVEAI] COMPLETE")
    print(f"[EVOLVEAI] steps={report['steps_completed']} best_step={report['best_step']} "
          f"best_val_loss={report['best_validation_loss']:.4f} "
          f"elapsed={report['elapsed_seconds']}s")
    print(f"[EVOLVEAI] report={report_path}")
    print(f"[EVOLVEAI] checkpoint={checkpoint_path}")
    return report
