"""Training and evaluation for recurrent spiking language models."""
from __future__ import annotations
import math
import random
import time
import torch
import torch.nn.functional as F
from model import RecurrentLIFLanguageModel
from evolve_cell import EvolveContextSNNLanguageModel
from data import encode, split_three_way, get_batch


def resolve_device(device="auto"):
    """Resolve auto/cpu/cuda safely and fail clearly for unavailable CUDA."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this runtime.")
    return str(resolved)


def evaluate(model, tokens, block_size, batch_size=8, batches=8, device="cpu",
             split_name="validation"):
    """Evaluate on fixed, evenly spaced windows for repeatable comparisons."""
    model.eval()
    needed = block_size + 1
    if tokens.numel() < needed:
        repeats = (needed - 1 + tokens.numel() - 1) // tokens.numel()
        tokens = tokens.repeat(repeats)
    max_start = tokens.numel() - needed
    count = max(1, int(batches) * int(batch_size))
    starts = torch.linspace(0, max_start, steps=count).round().long().unique()
    losses = []
    with torch.no_grad():
        for offset in range(0, starts.numel(), max(1, int(batch_size))):
            windows = starts[offset:offset + max(1, int(batch_size))]
            x = torch.stack([tokens[int(i):int(i) + block_size] for i in windows]).to(device)
            y = torch.stack([tokens[int(i) + 1:int(i) + block_size + 1] for i in windows]).to(device)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0)),
            "protocol": f"deterministic evenly-spaced {split_name} windows v3",
            "windows": int(starts.numel())}


def build_model(config):
    architecture = config.get("architecture", "baseline")
    if architecture == "evolve_context":
        model = EvolveContextSNNLanguageModel(
            embedding_dim=int(config.get("embedding_dim", 24)),
            hidden_size=int(config["hidden_size"]))
    elif architecture == "baseline":
        model = RecurrentLIFLanguageModel(
            embedding_dim=int(config.get("embedding_dim", 24)),
            hidden_size=int(config["hidden_size"]))
    else:
        raise ValueError("Unsupported checkpoint architecture.")
    return model


def train(text: str, steps=100, block_size=48, batch_size=8, hidden_size=96,
          embedding_dim=24, learning_rate=0.001, seed=42, progress=None,
          architecture="baseline", device="auto", initial_state_dict=None):
    if steps < 1 or block_size < 4 or batch_size < 1:
        raise ValueError("steps/batch_size must be positive and block_size >= 4")
    device = resolve_device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    tokens = encode(text)
    train_tokens, val_tokens, test_tokens = split_three_way(tokens)
    if architecture == "evolve_context":
        model = EvolveContextSNNLanguageModel(embedding_dim=embedding_dim, hidden_size=hidden_size)
    elif architecture == "baseline":
        model = RecurrentLIFLanguageModel(embedding_dim=embedding_dim, hidden_size=hidden_size)
    else:
        raise ValueError("architecture must be 'baseline' or 'evolve_context'")
    model = model.to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    started = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(train_tokens, batch_size, block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            val = evaluate(model, val_tokens, block_size, batch_size=min(batch_size, 8),
                           batches=3, device=device, split_name="validation")
            record = {"step": step, "train_loss": float(loss.item()),
                      "val_loss": val["loss"], "val_perplexity": val["perplexity"]}
            history.append(record)
            if progress:
                progress(step, steps, record)
    return {"model": model, "history": history, "train_bytes": int(train_tokens.numel()),
            "validation_bytes": int(val_tokens.numel()), "test_bytes": int(test_tokens.numel()),
            "elapsed_seconds": time.time() - started, "device": device,
            "config": {"steps": steps, "block_size": block_size, "batch_size": batch_size,
                       "hidden_size": hidden_size, "embedding_dim": embedding_dim,
                       "architecture": architecture, "learning_rate": learning_rate,
                       "seed": seed, "device": device}}


def evaluate_holdout_checkpoint(text: str, checkpoint_path: str, expected_sha256: str,
                                confirm_independent_corpus: bool = False,
                                device="auto", batches=16, batch_size=8):
    """One-time audit of a frozen checkpoint on a genuinely independent corpus.

    The confirmation is mandatory because a chronological tail may have been exposed
    during earlier experiments. Never use this score to tune candidates.
    """
    import hashlib
    from pathlib import Path
    if not confirm_independent_corpus:
        raise ValueError("Holdout evaluation requires explicit confirmation that this corpus is independent of all prior training and validation data.")
    actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not expected_sha256 or actual_hash != expected_sha256:
        raise ValueError("Holdout evaluation blocked: corpus hash does not match the supplied manifest.")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    model = build_model(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    resolved = resolve_device(device)
    model = model.to(resolved)
    tokens = encode(text)
    _, _, test_tokens = split_three_way(tokens)
    result = evaluate(model, test_tokens, int(config.get("block_size", 32)),
                      batch_size=batch_size, batches=batches, device=resolved,
                      split_name="untouched test")
    return {
        "evaluation_type": "final holdout audit; do not use for model selection",
        "corpus_sha256": actual_hash,
        "checkpoint_path": str(path),
        "checkpoint_config": config,
        "test_bytes": int(test_tokens.numel()),
        "test_loss": result["loss"],
        "test_perplexity": result["perplexity"],
        "protocol": result["protocol"],
        "device": resolved,
    }
