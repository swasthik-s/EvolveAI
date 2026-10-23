"""A small recurrent LIF spiking network for next-byte prediction.

This model deliberately contains no Transformer blocks, self-attention, or attention masks.
It is a research baseline, not a claim of human-level cognition.
"""
from __future__ import annotations
import torch
from torch import nn


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        slope = 5.0
        distance = (membrane - ctx.threshold).abs()
        grad = 1.0 / (1.0 + slope * distance).pow(2)
        return grad_output * grad, None


def spike(membrane: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    return _SurrogateSpike.apply(membrane, threshold)


class RecurrentLIFLanguageModel(nn.Module):
    """Byte-level recurrent LIF network with surrogate-gradient BPTT."""

    def __init__(self, vocab_size: int = 256, embedding_dim: int = 32,
                 hidden_size: int = 128, threshold: float = 1.0,
                 leak: float = 0.90, dropout: float = 0.05):
        super().__init__()
        if not 0.0 < leak < 1.0:
            raise ValueError("leak must be between 0 and 1")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.leak = leak
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_current = nn.Linear(embedding_dim, hidden_size)
        self.recurrent_current = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-byte logits with shape [batch, time, vocab]."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        embedded = self.embedding(tokens)
        batch, steps, _ = embedded.shape
        membrane = embedded.new_zeros(batch, self.hidden_size)
        previous_spikes = embedded.new_zeros(batch, self.hidden_size)
        outputs = []
        for t in range(steps):
            current = self.input_current(embedded[:, t]) + self.recurrent_current(previous_spikes)
            membrane = self.leak * membrane + current
            current_spikes = spike(membrane, self.threshold)
            membrane = membrane - current_spikes.detach() * self.threshold
            previous_spikes = current_spikes
            outputs.append(self.readout(self.dropout(current_spikes)))
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new_bytes: int = 160,
                 temperature: float = 0.8, device: str | torch.device = "cpu") -> bytes:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        generated = list(prompt or b" ")
        self.eval()
        for _ in range(max_new_bytes):
            context = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
            logits = self(context)[0, -1] / temperature
            next_byte = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            generated.append(next_byte)
        return bytes(generated[len(prompt):])
