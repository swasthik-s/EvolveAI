import sys
from pathlib import Path
import json

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn"))
from awakening import AwakeningAgent, STATE_VERSION, export_state_json


def test_awakening_agent_steps_and_records_finite_experience():
    agent = AwakeningAgent(seed=7)
    records = agent.run(20)
    assert len(records) == 20
    assert agent.step_count == 20
    assert all(r["action"] in {"predict_0", "predict_1"} for r in records)
    assert all(r["finite_state"] for r in records)
    assert all(r["mean_fast_state"] == r["mean_fast_state"] for r in records)
    assert all("score_0" in r and "score_1" in r for r in records)


def test_xor_observation_is_stationary_and_does_not_include_target():
    agent = AwakeningAgent(seed=7)
    observations = [agent._observation(a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
    assert all(len(obs) == 7 for obs in observations)
    assert observations == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    record = agent.step(input_pair=(1, 0), learn=False, explore=False)
    assert record["target"] == (record["input_a"] ^ record["input_b"]) == 1


def test_fixed_pair_validation_and_evaluation_flags():
    agent = AwakeningAgent(seed=3)
    with pytest.raises(ValueError):
        agent.step(input_pair=(0, 2))
    record = agent.step(input_pair=(0, 1), learn=False, explore=False)
    assert record["explored"] is False
    assert record["prediction"] == record["greedy_prediction"]


def test_initialization_does_not_change_global_torch_rng():
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    AwakeningAgent(seed=999)
    actual = torch.rand(4)
    assert torch.equal(actual, expected)


def test_awakening_state_can_be_exported_and_resumed():
    agent = AwakeningAgent(seed=11)
    agent.run(4)
    saved = json.loads(export_state_json(agent))
    resumed = AwakeningAgent(seed=11, state=saved)
    assert saved["state_version"] == STATE_VERSION
    assert resumed.step_count == 4
    assert len(resumed.history) == 4
    assert resumed.summary()["total_reward"] == agent.summary()["total_reward"]
    assert resumed.run(1)[0]["step"] == 5


def test_old_state_version_is_rejected_clearly():
    with pytest.raises(ValueError, match="version-3"):
        AwakeningAgent(seed=1, state={"state_version": 2})


def test_benchmark_smoke_runs_all_conditions_with_balanced_evaluation():
    from awakening_benchmark import run_benchmark

    result = run_benchmark(seeds=[1], train_steps=16, eval_repeats_per_pair=2)
    assert result["evaluation_trials_per_run"] == 8
    assert set(result["summary"]) == {
        "online_readout_update",
        "frozen_readout_control",
        "nonlinear_mlp_sanity_baseline",
    }
    assert all(row["evaluation_trials"] == 8 for row in result["runs"])
    assert all(row["all_states_finite"] for row in result["runs"])
    mlp = next(row for row in result["runs"] if row["condition"] == "nonlinear_mlp_sanity_baseline")
    assert mlp["heldout_greedy_accuracy"] >= 0.99
