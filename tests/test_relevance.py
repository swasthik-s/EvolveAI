from evolveai.relevance import assess_relevance, is_personal_context_query


def test_personal_identity_is_never_web_knowledge():
    assert is_personal_context_query("Who am i")
    result = assess_relevance("Who am i", "Self-discovery article", "Find your purpose", "Who am i")
    assert not result.accepted


def test_relevant_iron_man_evidence_passes():
    result = assess_relevance(
        "Who is Iron Man?", "Iron Man character",
        "Iron Man is Tony Stark, a fictional Marvel superhero.",
        "Who is Iron Man?",
    )
    assert result.accepted


def test_unrelated_character_evidence_fails_for_weather():
    result = assess_relevance(
        "What is the current weather?", "Iron Man character",
        "Iron Man is Tony Stark, a fictional Marvel superhero.",
        "What is the current weather?",
    )
    assert not result.accepted
