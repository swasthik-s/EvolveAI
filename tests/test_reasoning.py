from evolveai.reasoning import build_plan


def test_reasoning_plan_resolves_short_follow_up():
    plan = build_plan(
        "Why?",
        history=[
            {"role": "user", "content": "What is an egg?"},
            {"role": "assistant", "content": "An egg is a reproductive structure."},
        ],
    )
    assert plan.is_follow_up
    assert "What is an egg?" in plan.working_query
    assert plan.intent == "question"
    assert "evaluate_evidence" in plan.steps


def test_reasoning_plan_keeps_new_question_independent():
    plan = build_plan("What is the capital of Bhutan?", [])
    assert not plan.is_follow_up
    assert plan.working_query == "What is the capital of Bhutan?"
    assert plan.intent == "question"


def test_reasoning_plan_detects_fresh_information_request():
    plan = build_plan("What is the latest weather today?")
    assert plan.needs_fresh_research
