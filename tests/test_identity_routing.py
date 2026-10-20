import pytest

from evolveai.agent import EvolveAgent
from evolveai.memory import ExperienceMemory


def test_run_answers_assistant_identity_before_any_knowledge_lookup(tmp_path, monkeypatch):
    memory = ExperienceMemory(str(tmp_path / "identity.db"))

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("Identity question must not query Ghost Knowledge")

    monkeypatch.setattr(memory, "search_ghost_knowledge", unexpected_lookup)
    agent = EvolveAgent(memory, web_research=True)

    result = agent.run("Who are you?")

    assert result["mode"] == "assistant_identity"
    assert "EvolveAI" in result["answer"]
    assert "Iron Man" not in result["answer"]


def test_run_answers_user_identity_from_personal_memory(tmp_path, monkeypatch):
    memory = ExperienceMemory(str(tmp_path / "personal.db"))
    memory.remember("my name", "Swasthik")

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("Personal identity question must not query Ghost Knowledge")

    monkeypatch.setattr(memory, "search_ghost_knowledge", unexpected_lookup)
    agent = EvolveAgent(memory, web_research=True)

    result = agent.run("Who am I?")

    assert result["mode"] == "personal_context"
    assert "Swasthik" in result["answer"]


def test_assistant_identity_does_not_call_research(tmp_path, monkeypatch):
    memory = ExperienceMemory(str(tmp_path / "no_research.db"))

    def unexpected_research(*args, **kwargs):
        pytest.fail("Assistant identity must not call web research")

    monkeypatch.setattr("evolveai.agent.research", unexpected_research)
    agent = EvolveAgent(memory, web_research=True)

    result = agent.run("Who are you?")

    assert result["mode"] == "assistant_identity"
