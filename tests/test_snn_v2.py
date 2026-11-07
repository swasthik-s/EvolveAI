from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from snn_v2.model import AdaptiveLIFNetwork
from snn_v2.learner import ContinualSNNLearner
from snn_v2.data import SOURCE_REGISTRY


def test_event_step_shapes_and_finite_values():
    model = AdaptiveLIFNetwork(hidden_size=24)
    state = model.initial_state(2)
    logits, spikes, next_state = model.step(torch.tensor([65, 66]), state)
    assert logits.shape == (2, 256)
    assert spikes.shape == (2, 24)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(next_state.membrane).all()


def test_sequence_run_preserves_event_state():
    model = AdaptiveLIFNetwork(hidden_size=24)
    x = torch.randint(0, 256, (2, 20))
    logits, spikes, state = model.run(x)
    assert logits.shape == (2, 20, 256)
    assert spikes.shape == (2, 20, 24)
    assert state.membrane.shape == (2, 24)


def test_local_learning_updates_without_gradients(tmp_path):
    learner = ContinualSNNLearner(
        AdaptiveLIFNetwork(hidden_size=24),
        checkpoint_path=str(tmp_path / "snn-v2.pt"),
        replay_every=100,
    )
    before = learner.model.readout_bias.detach().clone()
    result = learner.learn_record(
        "The spiking neuron learns from temporal events.",
        source_id="test-corpus",
    )
    assert result["accepted"] is True
    assert learner.stats.updates > 0
    assert not torch.equal(before, learner.model.readout_bias)
    assert all(parameter.grad is None for parameter in learner.model.parameters())


def test_interaction_learning_requires_opt_in():
    learner = ContinualSNNLearner(AdaptiveLIFNetwork(hidden_size=24))
    result = learner.learn_interaction("hello", "hi", user_opt_in=False)
    assert result == {"accepted": False, "reason": "interaction_consent_required"}


def test_interaction_learning_accepts_explicit_opt_in():
    learner = ContinualSNNLearner(AdaptiveLIFNetwork(hidden_size=24))
    result = learner.learn_interaction(
        "What is a spike?", "A discrete neural event.", user_opt_in=True
    )
    assert result["accepted"] is True
    assert learner.stats.interaction_records == 1


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.pt"
    learner = ContinualSNNLearner(
        AdaptiveLIFNetwork(hidden_size=24), checkpoint_path=str(path)
    )
    learner.learn_record("Learning from data changes synapses.", source_id="unit-test")
    learner.save()
    restored = ContinualSNNLearner(
        AdaptiveLIFNetwork(hidden_size=24), checkpoint_path=str(path)
    )
    restored.load()
    assert restored.stats.records_learned == 1
    assert torch.equal(
        learner.model.readout_weights.cpu(), restored.model.readout_weights.cpu()
    )


def test_public_sources_are_allowlisted():
    assert {"tinystories", "wikitext", "ag_news", "gsm8k"} <= set(SOURCE_REGISTRY)
