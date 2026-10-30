"""Single-pass streaming trainer for EvolveAI's recurrent SNN."""
from __future__ import annotations

import math
import random
import time
import torch
import torch.nn.functional as F

from train import resolve_device, build_model


def _evaluate_stream(model, gatekeeper, block_size, batch_size, device, batches=4):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = gatekeeper.make_batch(batch_size, block_size, device, validation=True)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    loss = sum(losses) / len(losses)
    return {"loss": loss, "perplexity": math.exp(min(loss, 20.0))}


def train_streaming(
    gatekeeper,
    *,
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
    progress=None,
):
    if steps < 1 or eval_interval < 1 or patience < 1:
        raise ValueError("steps, eval_interval and patience must be positive.")
    device = resolve_device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    gatekeeper.warm_buffers()
    model = build_model({"architecture": architecture, "hidden_size": hidden_size,
                         "embedding_dim": embedding_dim}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = None
    best_step = 0
    stale = 0
    history = []
    last_train_loss = None
    started = time.time()

    for step in range(1, steps + 1):
        # Refill from all live sources while keeping memory bounded.
        gatekeeper.ingest(records=4)
        x, y = gatekeeper.make_batch(batch_size, block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_train_loss = float(loss.item())

        if step % eval_interval == 0 or step == 1 or step == steps:
            val = _evaluate_stream(model, gatekeeper, block_size, min(batch_size, 8), device)
            improved = val["loss"] < best_loss - min_delta
            if improved:
                best_loss = val["loss"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            batch_records = gatekeeper.last_batch_records
            batch_sources = sorted({item.source_id for item in batch_records})
            sample_record = batch_records[0] if batch_records else None
            record = {
                "step": step, "max_steps": steps, "train_loss": last_train_loss,
                "validation_loss": val["loss"], "validation_perplexity": val["perplexity"],
                "improved": improved, "best_step": best_step, "stale_evaluations": stale,
                "batch_sources": batch_sources,
                "batch_domains": sorted({item.domain for item in batch_records}),
                "batch_sample_preview": sample_record.text[:180] if sample_record else None,
                "gatekeeper": gatekeeper.status(),
            }
            history.append(record)
            if progress:
                progress(record)
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model, "history": history, "best_validation_loss": best_loss,
        "best_step": best_step, "steps_completed": history[-1]["step"] if history else 0,
        "stopped_early": bool(history and history[-1]["step"] < steps),
        "elapsed_seconds": time.time() - started, "device": device,
        "gatekeeper": gatekeeper.status(),
        "config": {"steps_requested": steps, "block_size": block_size,
                   "batch_size": batch_size, "hidden_size": hidden_size,
                   "embedding_dim": embedding_dim, "learning_rate": learning_rate,
                   "seed": seed, "architecture": architecture, "device": device,
                   "eval_interval": eval_interval, "patience": patience,
                   "min_delta": min_delta},
    }
