from evolveai.identity import answer_identity_question
from evolveai.response_adaptation import adapt_response, normalize_message


def test_identity_question_ignores_trailing_slash_and_punctuation():
    assert answer_identity_question("who are you/") == answer_identity_question("who are you?")


def test_superpower_question_routes_to_assistant_identity():
    answer = answer_identity_question("whats you superpower?")
    assert answer
    assert "connecting the dots" in answer.lower()
    assert "interview" not in answer.lower()


def test_repeated_greeting_does_not_add_canned_disclaimer():
    history = [
        {"role": "user", "content": "HEY"},
        {"role": "assistant", "content": "Hey! I'm EvolveAI."},
    ]
    assert normalize_message("hey/") == "hey"
    assert adapt_response("hey", "Hey! I'm EvolveAI.", history) == "Hey! I'm EvolveAI."
