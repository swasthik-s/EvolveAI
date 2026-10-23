"""Bounded, resumable-ish SNN training runner with validation-based stopping.

Example:
  python snn/controlled_train.py --train data/tinystories/train.txt \
      --validation data/tinystories/validation.txt --steps 500 --out-dir runs/snn-001

This runner never self-modifies code or fetches arbitrary data. It trains only on the
explicit training file, evaluates only on the explicit validation file, saves the best
checkpoint, and stops after a configured number of non-improving evaluations.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from model import RecurrentLIFLanguageModel
from data import encode, get_batch


def evaluate_fixed(model, tokens, block_size, batch_size, batches, seed, device):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            if tokens.numel() < block_size + 1:
                repeats = (block_size + 1 + tokens.numel() - 1) // tokens.numel()
                sample_tokens = tokens.repeat(repeats)
            else:
                sample_tokens = tokens
            starts = torch.randint(
                0, sample_tokens.numel() - block_size, (batch_size,), generator=generator
            )
            x = torch.stack([sample_tokens[int(i):int(i) + block_size] for i in starts]).to(device)
            y = torch.stack([sample_tokens[int(i) + 1:int(i) + block_size + 1] for i in starts]).to(device)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out-dir", default="runs/snn-controlled")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many evaluations without improvement")
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.patience < 1:
        parser.error("steps, eval-every, and patience must be positive")

    train_path, val_path = Path(args.train), Path(args.validation)
    if not train_path.is_file() or not val_path.is_file():
        raise SystemExit("Training and validation files must both exist.")
    train_text = train_path.read_text(encoding="utf-8", errors="replace")
    val_text = val_path.read_text(encoding="utf-8", errors="replace")
    train_tokens, val_tokens = encode(train_text), encode(val_text)
    if min(train_tokens.numel(), val_tokens.numel()) < 20:
        raise SystemExit("Both training and validation corpora must contain at least 20 bytes.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = RecurrentLIFLanguageModel(
        embedding_dim=args.embedding_dim, hidden_size=args.hidden_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_step = 0
    stale_evaluations = 0
    history = []
    started = time.time()

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_tokens, args.batch_size, args.block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            val = evaluate_fixed(
                model, val_tokens, args.block_size, args.batch_size, 8, args.seed, device
            )
            record = {"step": step, "train_loss": float(loss.item()),
                      "validation_loss": val["loss"], "validation_perplexity": val["perplexity"]}
            history.append(record)
            print(json.dumps(record))
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                best_step = step
                stale_evaluations = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {"embedding_dim": args.embedding_dim, "hidden_size": args.hidden_size},
                    "step": step, "validation_loss": best_loss, "seed": args.seed,
                }, out_dir / "best_model.pt")
            else:
                stale_evaluations += 1
                if stale_evaluations >= args.patience:
                    print(f"Early stopping: no validation improvement for {args.patience} evaluations.")
                    break

    summary = {
        "train_file": str(train_path), "validation_file": str(val_path),
        "steps_completed": step, "best_step": best_step, "best_validation_loss": best_loss,
        "best_validation_perplexity": math.exp(min(best_loss, 20.0)),
        "elapsed_seconds": round(time.time() - started, 3),
        "seed": args.seed, "device": str(device), "history": history,
        "checkpoint": str(out_dir / "best_model.pt"),
        "safety": "Bounded training only; no autonomous internet access, code changes, or unapproved actions.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
