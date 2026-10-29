import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn"))
from awakening import AwakeningAgent, export_state_json


def test_awakening_agent_steps_and_records_finite_experience():
    agent = AwakeningAgent(seed=7)
    records = agent.run(20)
    assert len(records) == 20
    assert agent.step_count == 20
    assert all(r["action"] in {"predict_0", "predict_1"} for r in records)
    assert all(r["finite_state"] for r in records)
    assert all(r["mean_fast_state"] == r["mean_fast_state"] for r in records)


def test_xor_observation_does_not_include_target():
    agent = AwakeningAgent(seed=7)
    record = agent.step()
    # Inputs are exposed separately for evaluation; the target is not an input feature.
    assert len(agent.last_observation) == 7
    assert record["target"] == (record["input_a"] ^ record["input_b"])


def test_awakening_state_can_be_exported_and_resumed():
    agent = AwakeningAgent(seed=11)
    agent.run(4)
    saved = json.loads(export_state_json(agent))
    resumed = AwakeningAgent(seed=11, state=saved)
    assert resumed.step_count == 4
    assert len(resumed.history) == 4
    assert resumed.summary()["total_reward"] == agent.summary()["total_reward"]
    assert resumed.run(1)[0]["step"] == 5
