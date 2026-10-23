"""Small decoder-only Transformer language model implemented with PyTorch."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 259
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        weights = weights.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd)
        self.position_embedding = nn.Embedding(self.cfg.block_size, self.cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(self.cfg) for _ in range(self.cfg.n_layer)])
        self.ln_f = nn.LayerNorm(self.cfg.n_embd)
        self.lm_head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.cfg.block_size: raise ValueError("Sequence exceeds block_size")
        pos = torch.arange(t, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=160, temperature=0.8, top_k=40):
        self.eval()
        temperature = max(0.05, float(temperature))
        for _ in range(max_new_tokens):
            cropped = idx[:, -self.cfg.block_size:]
            logits, _ = self(cropped)
            logits = logits[:, -1, :] / temperature
            if top_k and top_k < logits.size(-1):
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
