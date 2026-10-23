import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evolveai.identity import answer_identity_question


def test_superpower_question_answers_about_evolveai():
    answer = answer_identity_question("What's your superpower?")
    assert answer
    assert "connecting the dots" in answer.lower()
    assert "interview" not in answer.lower()


def test_superpower_variants_are_recognized():
    for prompt in ("what is your super power?", "what's your greatest strength?"):
        assert answer_identity_question(prompt)


def test_unrelated_question_does_not_match_identity_route():
    assert answer_identity_question("What is my superpower?") is None
