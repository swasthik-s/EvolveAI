import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn"))
from awakening import AwakeningAgent, export_state_json
import json


def test_awakening_agent_steps_and_records_experience():
    agent = AwakeningAgent(seed=7)
    records = agent.run(5)
    assert len(records) == 5
    assert agent.step_count == 5
    assert records[-1]["action"] in {"predict_0", "predict_1", "explore"}
    assert len(agent.history) == 5


def test_awakening_state_can_be_exported_and_resumed():
    agent = AwakeningAgent(seed=11)
    agent.run(4)
    saved = json.loads(export_state_json(agent))
    resumed = AwakeningAgent(seed=11, state=saved)
    assert resumed.step_count == 4
    assert len(resumed.history) == 4
    assert resumed.summary()["total_reward"] == agent.summary()["total_reward"]
    assert resumed.run(1)[0]["step"] == 5
