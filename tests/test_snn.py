import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn"))
import torch
from model import RecurrentLIFLanguageModel, spike
from data import encode, decode, split_tokens, get_batch
from train import train


def test_spike_forward_and_gradient():
    x = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    y = spike(x)
    assert y.tolist() == [0.0, 1.0, 1.0]
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (x.grad > 0).all()


def test_model_shape_and_finite_logits():
    model = RecurrentLIFLanguageModel(embedding_dim=8, hidden_size=16)
    x = torch.randint(0, 256, (2, 12))
    logits = model(x)
    assert logits.shape == (2, 12, 256)
    assert torch.isfinite(logits).all()


def test_data_round_trip_and_batches():
    text = "EvolveAI SNN\nlearns patterns. " * 4
    tokens = encode(text)
    assert decode(tokens) == text
    train_tokens, val_tokens = split_tokens(tokens)
    x, y = get_batch(train_tokens, 3, 8)
    assert x.shape == y.shape == (3, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert val_tokens.numel() > 0


def test_short_training_run_produces_metrics():
    corpus = ("User: hi\nAssistant: hello\nUser: learn\nAssistant: improve from feedback\n\n" * 20)
    result = train(corpus, steps=2, block_size=8, batch_size=2, hidden_size=16, embedding_dim=8, seed=7)
    assert result["history"]
    assert result["history"][-1]["val_loss"] > 0
    assert torch.isfinite(torch.tensor(result["history"][-1]["val_loss"]))
