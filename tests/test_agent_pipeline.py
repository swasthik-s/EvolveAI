from evolveai.agent import EvolveAgent
from evolveai.memory import ExperienceMemory


def test_agent_emits_processing_stages_and_reasoning_plan(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "pipeline.db"))
    agent = EvolveAgent(memory, web_research=False)
    stages = []

    result = agent.run("hello", progress=stages.append)

    assert result["reasoning_plan"]["intent"] == "greeting"
    assert stages[0] == "Understanding your message"
    assert "Choosing the response path" in stages
    assert stages[-1] == "Preparing a concise response"
