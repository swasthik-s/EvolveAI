from evolveai.agent import EvolveAgent
from evolveai.memory import ExperienceMemory


def test_ghost_knowledge_is_saved_and_retrievable(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "ghost.db"))
    memory.add_ghost_knowledge(
        question="What is the capital of Bhutan?",
        fact="Thimphu is the capital of Bhutan.",
        title="Bhutan overview",
        source_url="https://example.org/bhutan",
        confidence=0.7,
        status="candidate",
    )

    results = memory.search_ghost_knowledge("capital Bhutan")
    assert results
    assert results[0]["fact"] == "Thimphu is the capital of Bhutan."
    assert results[0]["status"] == "candidate"


def test_agent_reuses_matching_ghost_knowledge(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "agent.db"))
    memory.add_ghost_knowledge(
        question="What is the capital of Bhutan?",
        fact="Thimphu is the capital of Bhutan.",
        title="Bhutan overview",
        source_url="https://example.org/bhutan",
        confidence=0.7,
        status="candidate",
    )
    agent = EvolveAgent(memory, web_research=False)

    result = agent.run("What is the capital of Bhutan?")

    assert result["mode"] == "ghost_knowledge"
    assert "Thimphu" in result["answer"]
    assert "https://example.org/bhutan" in result["answer"]


def test_agent_searches_only_when_an_unknown_question_needs_it(tmp_path, monkeypatch):
    import evolveai.agent as agent_module

    calls = []

    def fake_research(query, max_results=5):
        calls.append(query)
        return [{
            "title": "Example facts",
            "url": "https://example.org/facts",
            "excerpt": "A useful extractive answer for this question.",
            "confidence": 0.6,
            "source_domain": "example.org",
        }]

    monkeypatch.setattr(agent_module, "research", fake_research)
    memory = ExperienceMemory(str(tmp_path / "research.db"))
    agent = EvolveAgent(memory)

    greeting = agent.run("hello")
    assert greeting["mode"] == "rules"
    assert calls == []

    result = agent.run("What is a fictional example fact?")
    assert result["mode"] == "web_research"
    assert calls == ["What is a fictional example fact?"]
    assert memory.stats()["ghost_knowledge"] == 1
