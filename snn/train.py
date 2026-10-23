"""Training and evaluation for the recurrent spiking baseline."""
from __future__ import annotations
import math
import random
import time
import torch
import torch.nn.functional as F
from model import RecurrentLIFLanguageModel
from data import encode, split_tokens, get_batch


def evaluate(model, tokens, block_size, batch_size=8, batches=8, device="cpu"):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(tokens, batch_size, block_size, device)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0))}


def train(text: str, steps=100, block_size=48, batch_size=8, hidden_size=96,
          embedding_dim=24, learning_rate=0.001, seed=42, progress=None):
    if steps < 1 or block_size < 4 or batch_size < 1:
        raise ValueError("steps/batch_size must be positive and block_size >= 4")
    random.seed(seed)
    torch.manual_seed(seed)
    tokens = encode(text)
    train_tokens, val_tokens = split_tokens(tokens)
    model = RecurrentLIFLanguageModel(embedding_dim=embedding_dim, hidden_size=hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    started = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(train_tokens, batch_size, block_size)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            val = evaluate(model, val_tokens, block_size, batch_size=min(batch_size, 8), batches=3)
            record = {"step": step, "train_loss": float(loss.item()), "val_loss": val["loss"], "val_perplexity": val["perplexity"]}
            history.append(record)
            if progress:
                progress(step, steps, record)
    return {"model": model, "history": history, "train_bytes": int(train_tokens.numel()),
            "validation_bytes": int(val_tokens.numel()), "elapsed_seconds": time.time() - started,
            "config": {"steps": steps, "block_size": block_size, "batch_size": batch_size,
                       "hidden_size": hidden_size, "embedding_dim": embedding_dim}}
