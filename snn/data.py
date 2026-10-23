"""Byte-level corpus utilities with deterministic chronological splits."""
from __future__ import annotations
import torch

VOCAB_SIZE = 256


def encode(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8", errors="replace")), dtype=torch.long)


def decode(values) -> str:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return bytes(int(v) % 256 for v in values).decode("utf-8", errors="replace")


def split_tokens(tokens: torch.Tensor, train_fraction: float = 0.9):
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    if tokens.numel() < 20:
        raise ValueError("Corpus must contain at least 20 UTF-8 bytes")
    boundary = max(10, min(int(tokens.numel() * train_fraction), tokens.numel() - 5))
    return tokens[:boundary], tokens[boundary:]


def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device="cpu"):
    if tokens.numel() < 2:
        raise ValueError("Need at least two tokens to form a next-token batch")
    needed = block_size + 1
    if tokens.numel() < needed:
        repeats = (needed + tokens.numel() - 1) // tokens.numel()
        tokens = tokens.repeat(repeats)
    max_start = tokens.numel() - block_size - 1
    starts = torch.randint(max_start + 1, (batch_size,))
    x = torch.stack([tokens[int(i):int(i) + block_size] for i in starts])
    y = torch.stack([tokens[int(i) + 1:int(i) + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
