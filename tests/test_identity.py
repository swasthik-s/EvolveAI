from evolveai.identity import answer_identity_question


def test_assistant_identity_is_answered_locally():
    answer = answer_identity_question("Who are you?")
    assert answer is not None
    assert "EvolveAI" in answer
    assert "experimental AI assistant" in answer
    assert "I use explicit rules" in answer


def test_assistant_identity_handles_punctuation_and_case():
    assert answer_identity_question("  WHO ARE YOU?!  ") is not None


def test_unrelated_question_is_not_misclassified_as_identity():
    assert answer_identity_question("Who is Iron Man?") is None
