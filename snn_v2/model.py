"""Event-driven recurrent adaptive-LIF network.

No Transformer, attention, autograd, BPTT, or pretrained weights. The model
processes one byte event at a time and exposes membrane/adaptation state.
Learning is implemented separately with local plasticity rules.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class NetworkState:
    membrane: torch.Tensor
    adaptation: torch.Tensor
    spikes: torch.Tensor
    refractory: torch.Tensor

    def detach(self):
        return NetworkState(self.membrane.detach(), self.adaptation.detach(),
                            self.spikes.detach(), self.refractory.detach())


class AdaptiveLIFNetwork(nn.Module):
    """Single-layer recurrent ALIF core with byte-event input and spike decoder."""

    def __init__(self, vocab_size=256, hidden_size=192, threshold=1.0,
                 membrane_decay=0.88, adaptation_decay=0.995,
                 adaptation_strength=0.15, refractory_steps=1, seed=42):
        super().__init__()
        if vocab_size < 2 or hidden_size < 8:
            raise ValueError("vocab_size must be >= 2 and hidden_size >= 8")
        if not 0 < membrane_decay < 1 or not 0 < adaptation_decay < 1:
            raise ValueError("decay values must be in (0, 1)")
        if threshold <= 0 or adaptation_strength < 0 or refractory_steps < 0:
            raise ValueError("Invalid neuron dynamics")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.threshold = float(threshold)
        self.membrane_decay = float(membrane_decay)
        self.adaptation_decay = float(adaptation_decay)
        self.adaptation_strength = float(adaptation_strength)
        self.refractory_steps = int(refractory_steps)

        # Isolated initialization does not perturb the host RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.input_weights = nn.Parameter(torch.randn(vocab_size, hidden_size) * (1.0 / vocab_size) ** 0.5)
            self.recurrent_weights = nn.Parameter(torch.randn(hidden_size, hidden_size) * (1.0 / hidden_size) ** 0.5)
            self.readout_weights = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.02)
            self.readout_bias = nn.Parameter(torch.zeros(vocab_size))
        with torch.no_grad():
            mask = torch.rand_like(self.recurrent_weights) < 0.15
            self.recurrent_weights.mul_(mask).mul_(0.35)

    def initial_state(self, batch_size=1, device=None):
        device = device or self.input_weights.device
        shape = (int(batch_size), self.hidden_size)
        zeros = torch.zeros(shape, dtype=self.input_weights.dtype, device=device)
        return NetworkState(zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone())

    @torch.no_grad()
    def step(self, symbols, state):
        """Consume one byte per batch row and return logits, spikes, next state."""
        if symbols.ndim != 1:
            raise ValueError("symbols must have shape [batch]")
        symbols = symbols.long()
        if torch.any(symbols < 0) or torch.any(symbols >= self.vocab_size):
            raise ValueError("symbol index outside vocabulary")
        input_current = self.input_weights[symbols]
        recurrent_current = state.spikes @ self.recurrent_weights
        active = (state.refractory <= 0).to(input_current.dtype)
        membrane = self.membrane_decay * state.membrane * active + input_current + recurrent_current
        dynamic_threshold = self.threshold + self.adaptation_strength * state.adaptation
        spikes = (membrane >= dynamic_threshold).to(membrane.dtype) * active
        membrane = membrane - spikes * dynamic_threshold
        adaptation = self.adaptation_decay * state.adaptation + (1 - self.adaptation_decay) * spikes
        if self.refractory_steps:
            refractory = torch.where(spikes > 0, torch.full_like(state.refractory, float(self.refractory_steps)),
                                     (state.refractory - 1).clamp_min(0))
        else:
            refractory = torch.zeros_like(state.refractory)
        next_state = NetworkState(membrane, adaptation, spikes, refractory)
        logits = spikes @ self.readout_weights + self.readout_bias
        return logits, spikes, next_state

    @torch.no_grad()
    def run(self, symbols, state=None):
        """Process [batch, time] byte indices and return per-step logits/spikes."""
        if symbols.ndim != 2:
            raise ValueError("symbols must have shape [batch, time]")
        state = state or self.initial_state(symbols.shape[0], symbols.device)
        logits_out, spikes_out = [], []
        for t in range(symbols.shape[1]):
            logits, spikes, state = self.step(symbols[:, t], state)
            logits_out.append(logits)
            spikes_out.append(spikes)
        return torch.stack(logits_out, 1), torch.stack(spikes_out, 1), state

    @torch.no_grad()
    def generate(self, prompt, max_new_bytes=160, temperature=0.9):
        if max_new_bytes < 1 or temperature <= 0:
            raise ValueError("max_new_bytes and temperature must be positive")
        raw = prompt.encode("utf-8", errors="replace") if isinstance(prompt, str) else bytes(prompt)
        if not raw:
            raw = b" "
        state = self.initial_state(1, self.input_weights.device)
        logits = None
        for value in raw:
            logits, _, state = self.step(torch.tensor([value], device=self.input_weights.device), state)
        generated = bytearray()
        for _ in range(max_new_bytes):
            probs = torch.softmax(logits[0] / temperature, dim=-1)
            token = int(torch.multinomial(probs, 1).item())
            generated.append(token)
            logits, _, state = self.step(torch.tensor([token], device=self.input_weights.device), state)
        return generated.decode("utf-8", errors="replace")
