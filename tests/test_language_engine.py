from evolveai.language_engine import (
    MessagePlan,
    Proposition,
    generate_from_meaning,
    identity_plan,
    user_identity_plan,
)


def test_identity_is_realised_from_propositions():
    answer = generate_from_meaning(identity_plan())
    assert answer.startswith("I am EvolveAI")
    assert "I use explicit rules" in answer


def test_repeated_identity_is_composed_as_multiple_sentences():
    answer = generate_from_meaning(identity_plan(repeated=True))
    assert answer.startswith("Still here.")
    assert "My identity is the same" in answer


def test_verb_agreement_and_past_tense():
    assert generate_from_meaning(
        MessagePlan((Proposition("EvolveAI", "do", "learn", tense="present"),))
    ) == "EvolveAI learns."
    assert generate_from_meaning(
        MessagePlan((Proposition("I", "do", "build", tense="past"),))
    ) == "I built."


def test_user_identity_uses_name_as_semantic_content():
    answer = generate_from_meaning(user_identity_plan("Swasthik"))
    assert answer == "You are Swasthik."
