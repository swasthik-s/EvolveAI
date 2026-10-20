from evolveai.conversation.understanding import understand_message
from evolveai.conversation.state import build_conversation_state
from evolveai.learning.policy import CommunicationPolicy
from evolveai.memory import ExperienceMemory


def test_understanding_distinguishes_request_question_correction_and_statement():
    assert understand_message("Can you build this?").speech_act == "request"
    assert understand_message("Build a tracker").speech_act == "request"
    assert understand_message("No, I meant the second item").speech_act == "correction"
    assert understand_message("I work night shifts").speech_act == "statement"


def test_short_follow_up_uses_context_signal():
    history = [
        {"role": "user", "content": "Explain the recipe"},
        {"role": "assistant", "content": "Do you want a simple or detailed version?"},
    ]
    frame = understand_message("What about milk?", history)
    assert frame.speech_act == "follow_up"
    state = build_conversation_state(history, "What about milk?")
    assert state.latest_assistant_message.startswith("Do you want")


def test_policy_selects_semantic_action_without_response_templates(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "policy.db"))
    policy = CommunicationPolicy(memory)
    frame = understand_message("No, that's not what I meant")
    decision = policy.choose(frame)
    assert decision.action == "acknowledge_correction"
    assert "Selected" in decision.reason


def test_feedback_updates_strategy_statistics(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "policy-feedback.db"))
    task_id = memory.save_task("Question?", "Answer", "rules", strategy="answer_directly")
    memory.rate_task(task_id, 1.0, "Useful")
    stats = memory.strategy_statistics()
    assert stats["answer_directly"]["count"] == 1
    assert stats["answer_directly"]["mean_score"] == 1.0


def test_policy_can_change_strategy_after_repeated_negative_feedback(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "adaptive-policy.db"))
    for index in range(4):
        task_id = memory.save_task(
            f"Question {index}?", "Unhelpful answer", "rules",
            strategy="answer_directly",
        )
        memory.rate_task(task_id, 0.0, "Not useful")

    policy = CommunicationPolicy(memory)
    frame = understand_message("What is this?")
    decision = policy.choose(frame)
    assert decision.action == "ask_clarification"
