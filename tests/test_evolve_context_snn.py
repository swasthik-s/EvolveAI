import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn"))
import torch
from evolve_cell import EvolveContextSNNLanguageModel, EvolveSpikingContextCell
from train import train


def test_evolve_context_snn_returns_finite_byte_logits():
    model = EvolveContextSNNLanguageModel(embedding_dim=8, hidden_size=16)
    tokens = torch.randint(0, 256, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, 256)
    assert torch.isfinite(logits).all()


def test_adaptive_cell_updates_activity_trace():
    cell = EvolveSpikingContextCell(input_size=8, hidden_size=12)
    x = torch.randn(2, 8)
    zeros = torch.zeros(2, 12)
    spikes, fast, slow, adaptation = cell(x, zeros, zeros, zeros, zeros)
    assert spikes.shape == fast.shape == slow.shape == adaptation.shape == (2, 12)
    assert torch.isfinite(adaptation).all()
    assert torch.all(adaptation >= 0)


def test_context_architecture_can_train_for_two_steps():
    corpus = ("User: hi\nAssistant: hello\nUser: learn\nAssistant: improve from feedback\n\n" * 20)
    result = train(corpus, steps=2, block_size=8, batch_size=2, hidden_size=16,
                   embedding_dim=8, seed=7, architecture="evolve_context")
    assert result["config"]["architecture"] == "evolve_context"
    assert torch.isfinite(torch.tensor(result["history"][-1]["val_loss"]))
