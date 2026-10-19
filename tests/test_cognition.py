from evolveai.cognition import understand


def test_known_alias_is_canonicalized_before_search():
    frame = understand("Who is ironman?")
    assert frame.intent == "person_or_character"
    assert frame.subject == "Iron Man"
    assert "Tony Stark" not in frame.search_query  # alias is kept in the ontology, query is canonical
    assert "Iron Man" in frame.search_query
    assert frame.confidence >= 0.9
    assert "canonicalized" in frame.explanation


def test_unknown_entity_is_not_silently_guessed():
    frame = understand("Who is Zorvath?")
    assert frame.subject == "Zorvath"
    assert frame.entities == ("Zorvath",)
    assert frame.confidence < 0.9


def test_non_question_falls_back_to_original_text():
    frame = understand("Help me plan a weekend")
    assert frame.intent == "open_request"
    assert frame.search_query == "Help me plan a weekend"
