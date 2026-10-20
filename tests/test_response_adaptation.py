from evolveai.response_adaptation import adapt_response, repeated_question_count


def test_first_question_gets_direct_answer():
    history = []
    answer = "I'm EvolveAI, your experimental AI assistant."
    assert adapt_response("Who are you?", answer, history, "assistant_identity") == answer


def test_repeated_assistant_identity_changes_delivery():
    history = [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I'm EvolveAI, your experimental AI assistant."},
    ]
    answer = "I'm EvolveAI, your experimental AI assistant."
    result = adapt_response("Who are you?", answer, history, "assistant_identity")
    assert result != answer
    assert "EvolveAI" in result
    assert "😄" in result


def test_multiple_repeats_are_recognized():
    history = [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am EvolveAI."},
        {"role": "user", "content": "who are you"},
        {"role": "assistant", "content": "Still me."},
    ]
    assert repeated_question_count("WHO ARE YOU?!", history) == 2
    result = adapt_response("Who are you?", "I am EvolveAI.", history, "assistant_identity")
    assert "Checking whether I’ll change my story" in result


def test_repeated_personal_identity_uses_saved_name():
    history = [{"role": "user", "content": "Who am I?"}]
    result = adapt_response(
        "Who am I?", "You're Swasthik, the person I'm talking to. 😄",
        history, "personal_context",
    )
    assert "Swasthik" in result
    assert result != "You're Swasthik, the person I'm talking to. 😄"


def test_non_repeated_research_answer_stays_unchanged():
    answer = "The answer is 42.\n\nSources: [Example](https://example.com)"
    assert adapt_response("What is the answer?", answer, [], "web_research") == answer
