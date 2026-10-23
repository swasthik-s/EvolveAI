"""Training loop for EvolveAI Mini."""
import random
import torch
from .data import prepare_text, split_data, get_batch, VOCAB_SIZE
from .model import TinyGPT, ModelConfig

def train_model(text, steps=300, batch_size=16, block_size=128, learning_rate=3e-4,
                n_embd=128, n_head=4, n_layer=2, seed=42, progress=None):
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = prepare_text(text)
    train_tokens, val_tokens = split_data(tokens)
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, block_size=block_size,
                      n_embd=n_embd, n_head=n_head, n_layer=n_layer, dropout=0.1)
    model = TinyGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    eval_every = max(1, min(50, steps // 10))
    for step in range(1, steps + 1):
        model.train()
        xb, yb = get_batch(train_tokens, batch_size, block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_tokens, min(batch_size, 8), block_size, device)
                _, val_loss = model(vx, vy)
            record = {"step": step, "train_loss": float(loss.item()), "val_loss": float(val_loss.item())}
            history.append(record)
            if progress: progress(step, steps, record)
    model.eval()
    return model, cfg, history, len(tokens), device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
