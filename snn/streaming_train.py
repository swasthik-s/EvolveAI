"""Single-pass streaming trainer for EvolveAI's recurrent SNN."""
from __future__ import annotations

import math
import random
import time

import torch
import torch.nn.functional as F

from train import resolve_device, build_model


def _autocast_context(device, amp_dtype):
    if device.startswith("cuda") and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    from contextlib import nullcontext
    return nullcontext()


def _evaluate_stream(model, gatekeeper, block_size, batch_size, device,
                     amp_dtype=None, batches=4):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = gatekeeper.make_batch(
                batch_size, block_size, device, validation=True
            )
            with _autocast_context(device, amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )
            losses.append(float(loss.float().item()))
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0))}


def train_streaming(
    gatekeeper,
    *,
    steps=8000,
    block_size=128,
    batch_size=64,
    hidden_size=256,
    embedding_dim=96,
    learning_rate=0.001,
    seed=42,
    architecture="baseline",
    device="cuda",
    eval_interval=200,
    patience=15,
    min_delta=0.001,
    progress=None,
    memory=None,
    director=None,
    checkpoint_callback=None,
    initial_state_dict=None,
):
    if steps < 1 or eval_interval < 1 or patience < 1:
        raise ValueError("steps, eval_interval and patience must be positive.")
    if block_size < 4 or batch_size < 1 or hidden_size < 1:
        raise ValueError("block_size, batch_size and hidden_size must be positive.")

    device = resolve_device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Select native Tensor Core precision by compute capability.
    # Turing T4 (7.5) has FP16 Tensor Cores, but no native BF16 Tensor Cores.
    amp_dtype = None
    scaler = None
    if device.startswith("cuda"):
        capability = torch.cuda.get_device_capability(0)
        native_bf16 = capability[0] >= 8 and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
        scaler = torch.cuda.amp.GradScaler(
            enabled=(amp_dtype == torch.float16)
        )
        print(
            f"[GPU] name={torch.cuda.get_device_name(0)} "
            f"compute_capability={capability[0]}.{capability[1]} "
            f"native_bf16={native_bf16} "
            f"amp={str(amp_dtype).replace('torch.', '')} "
            f"total_vram_gib={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}"
        )
        torch.cuda.reset_peak_memory_stats()

    gatekeeper.warm_buffers()
    model = build_model({
        "architecture": architecture,
        "hidden_size": hidden_size,
        "embedding_dim": embedding_dim,
    }).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
        print("[MODEL] warm-started from supplied checkpoint weights; optimizer state is freshly initialized.")

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, fused=device.startswith("cuda")
        )
        optimizer_name = "AdamW(fused)"
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer_name = "AdamW"

    print(
        f"[MODEL] architecture={architecture} hidden={hidden_size} "
        f"embedding={embedding_dim} sequence={block_size} batch={batch_size} "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"[OPTIMIZER] {optimizer_name}; [AMP] {amp_dtype}")

    best_loss = float("inf")
    best_state = None
    best_step = 0
    stale = 0
    history = []
    last_train_loss = None
    started = time.time()
    total_tokens_seen = 0

    model.train()
    for step in range(1, steps + 1):
        gatekeeper.ingest(records=8)
        available_domains = sorted({r.domain for r in gatekeeper.train_buffer})
        selected_domain, domain_weights = (
            director.choose_domain(available_domains)
            if director else (None, {})
        )
        x, y = gatekeeper.make_batch(
            batch_size, block_size, device, preferred_domain=selected_domain
        )

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        last_train_loss = float(loss.detach().float().item())
        total_tokens_seen += int(x.numel())
        batch_records = list(gatekeeper.last_batch_records)

        if memory:
            memory.record_batch(
                step, last_train_loss, batch_records,
                {"selected_domain": selected_domain}
            )
        if director and selected_domain:
            director.curiosity.observe(selected_domain, last_train_loss)

        if step % eval_interval == 0 or step == 1 or step == steps:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = max(time.time() - started, 1e-9)
            val = _evaluate_stream(
                model, gatekeeper, block_size, min(batch_size, 16),
                device, amp_dtype=amp_dtype
            )
            improved = val["loss"] < best_loss - min_delta
            if improved:
                best_loss = val["loss"]
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1

            batch_records = gatekeeper.last_batch_records
            record = {
                "step": step,
                "max_steps": steps,
                "train_loss": last_train_loss,
                "validation_loss": val["loss"],
                "validation_perplexity": val["perplexity"],
                "improved": improved,
                "best_step": best_step,
                "stale_evaluations": stale,
                "batch_sources": sorted({item.source_id for item in batch_records}),
                "batch_domains": sorted({item.domain for item in batch_records}),
                "batch_sample_preview": (
                    batch_records[0].text[:180] if batch_records else None
                ),
                "gatekeeper": gatekeeper.status(),
                "selected_domain": selected_domain,
                "domain_weights": domain_weights,
                "memory_domains": memory.domain_summary() if memory else {},
                "tokens_seen": total_tokens_seen,
                "tokens_per_second": round(total_tokens_seen / elapsed, 1),
                "elapsed_seconds": round(elapsed, 2),
                "gpu_memory": ({
                    "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
                    "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                    "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                } if device.startswith("cuda") else None),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "amp_dtype": str(amp_dtype) if amp_dtype else "disabled",
            }
            history.append(record)
            if progress:
                progress(record)
            if checkpoint_callback:
                checkpoint_callback(
                    record,
                    model.state_dict(),
                    best_state if best_state is not None else model.state_dict()
                )
            if stale >= patience:
                print(
                    f"[EARLY STOP] no validation improvement for {patience} "
                    f"evaluations; best_step={best_step}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_validation_loss": best_loss,
        "best_step": best_step,
        "steps_completed": history[-1]["step"] if history else 0,
        "stopped_early": bool(history and history[-1]["step"] < steps),
        "elapsed_seconds": time.time() - started,
        "device": device,
        "gatekeeper": gatekeeper.status(),
        "memory_domains": memory.domain_summary() if memory else {},
        "config": {
            "steps_requested": steps,
            "block_size": block_size,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "embedding_dim": embedding_dim,
            "learning_rate": learning_rate,
            "seed": seed,
            "architecture": architecture,
            "device": device,
            "eval_interval": eval_interval,
            "patience": patience,
            "min_delta": min_delta,
            "amp_dtype": str(amp_dtype) if amp_dtype else "disabled",
            "optimizer": optimizer_name,
            "tokens_seen": total_tokens_seen,
        },
    }
