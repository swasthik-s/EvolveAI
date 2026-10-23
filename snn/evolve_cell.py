"""Experimental EvolveAI dual-timescale adaptive-threshold spiking cell.

Hypothesis: fast membrane state captures local features; slow state carries
longer context; an activity trace adapts firing threshold. This is a testable
design hypothesis, not a claim of biological equivalence or better performance.
"""
import torch
from torch import nn
from model import spike


class EvolveSpikingContextCell(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, threshold=1.0,
                 leak_fast=0.75, leak_slow=0.97, adaptation_rate=0.02):
        super().__init__()
        if not (0 < leak_fast < leak_slow < 1):
            raise ValueError("Require 0 < leak_fast < leak_slow < 1")
        if threshold <= 0 or adaptation_rate < 0:
            raise ValueError("Invalid threshold/adaptation settings")
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.leak_fast = leak_fast
        self.leak_slow = leak_slow
        self.adaptation_rate = adaptation_rate
        self.input_current = nn.Linear(input_size, hidden_size)
        self.recurrent_current = nn.Linear(hidden_size, hidden_size, bias=False)
        self.slow_context = nn.Linear(hidden_size, hidden_size, bias=False)
        self.context_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, fast, slow, previous_spikes, adaptation):
        current = (self.input_current(x)
                   + self.recurrent_current(previous_spikes)
                   + self.slow_context(slow))
        gate = torch.sigmoid(self.context_gate(torch.cat([x, previous_spikes], dim=-1)))
        fast = self.leak_fast * fast + current
        slow = self.leak_slow * slow + gate * current
        dynamic_threshold = self.threshold + adaptation
        spikes = spike(fast + 0.25 * slow, dynamic_threshold)
        fast = fast - spikes.detach() * dynamic_threshold
        adaptation = 0.95 * adaptation + self.adaptation_rate * spikes.detach()
        return spikes, fast, slow, adaptation


class EvolveContextSNNLanguageModel(nn.Module):
    """Byte-level SNN with fast/slow state and adaptive firing thresholds.

    Uses surrogate-gradient BPTT. It is not yet the production chat engine.
    """
    def __init__(self, vocab_size=256, embedding_dim=32, hidden_size=64,
                 threshold=1.0, leak_fast=0.75, leak_slow=0.97,
                 adaptation_rate=0.02, dropout=0.05):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cell = EvolveSpikingContextCell(
            embedding_dim, hidden_size, threshold, leak_fast, leak_slow, adaptation_rate
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        embedded = self.embedding(tokens)
        batch, steps, _ = embedded.shape
        fast = embedded.new_zeros(batch, self.hidden_size)
        slow = embedded.new_zeros(batch, self.hidden_size)
        previous = embedded.new_zeros(batch, self.hidden_size)
        adaptation = embedded.new_zeros(batch, self.hidden_size)
        outputs = []
        for t in range(steps):
            previous, fast, slow, adaptation = self.cell(
                embedded[:, t], fast, slow, previous, adaptation
            )
            outputs.append(self.readout(self.dropout(previous)))
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new_bytes=160, temperature=0.8, device="cpu"):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        prefix = list(prompt or b" ")
        generated = prefix[:]
        for _ in range(max_new_bytes):
            context = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
            logits = self(context)[0, -1] / temperature
            next_byte = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            generated.append(next_byte)
        return bytes(generated[len(prefix):])
