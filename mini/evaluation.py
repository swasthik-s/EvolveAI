"""Simple, honest evaluation for a character/byte-level language model."""
import math
import torch
from data import prepare_text, split_data, get_batch, encode, decode

@torch.no_grad()
def evaluate_loss(model, text, batch_size=8, batches=20, device="cpu"):
    tokens = prepare_text(text)
    _, val = split_data(tokens)
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(val, min(batch_size, 8), model.cfg.block_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    avg = sum(losses) / len(losses)
    return {"loss": avg, "perplexity": math.exp(min(avg, 20)), "tokens": len(val)}

@torch.no_grad()
def complete(model, prompt, max_new_tokens=160, temperature=0.8, top_k=40, device="cpu"):
    # BOS provides a document boundary; prompt is byte-tokenized UTF-8.
    ids = torch.tensor([[257] + encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(generated[0].tolist()[1:])
