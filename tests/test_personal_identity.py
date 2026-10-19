from evolveai.agent import EvolveAgent


class DummyMemory:
    pass


def test_who_am_i_uses_saved_personal_identity():
    agent = EvolveAgent(DummyMemory(), web_research=True)
    answer = agent._personal_identity_answer(
        "Who am I?",
        [{"key": "my name", "value": "Swasthik"}],
    )
    assert answer == "You're Swasthik, the person I'm talking to. 😄"


def test_who_am_i_does_not_guess_if_name_is_missing():
    agent = EvolveAgent(DummyMemory(), web_research=True)
    answer = agent._personal_identity_answer("Who am I?", [])
    assert answer is not None
    assert "I won't guess" in answer


def test_unrelated_question_is_not_routed_as_identity():
    agent = EvolveAgent(DummyMemory(), web_research=True)
    assert agent._personal_identity_answer(
        "Who is Iron Man?", [{"key": "my name", "value": "Swasthik"}]
    ) is None
