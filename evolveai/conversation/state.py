"""Compact, reconstructable dialogue state derived from recent turns."""
from dataclasses import dataclass
from .understanding import understand_message


@dataclass(frozen=True)
class ConversationState:
    turn_count: int
    latest_user_message: str
    latest_assistant_message: str
    open_question: str
    current_topic: str
    last_speech_act: str


def build_conversation_state(history=None, current_message="") -> ConversationState:
    turns = list(history or [])
    latest_user = str(current_message or "").strip()
    latest_assistant = ""
    open_question = ""
    for item in reversed(turns):
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if not latest_user and item.get("role") == "user":
            latest_user = content
        if not latest_assistant and item.get("role") == "assistant":
            latest_assistant = content
        if item.get("role") == "assistant" and "?" in content and not open_question:
            open_question = content
        if latest_user and latest_assistant and open_question:
            break
    frame = understand_message(latest_user, turns)
    return ConversationState(
        turn_count=len(turns),
        latest_user_message=latest_user,
        latest_assistant_message=latest_assistant,
        open_question=open_question,
        current_topic=latest_user,
        last_speech_act=frame.speech_act,
    )
