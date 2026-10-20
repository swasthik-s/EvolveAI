"""Select and improve communication strategies from explicit feedback.

This is a small contextual bandit-style policy: it starts with transparent
priors, then ranks strategies using observed user ratings. It learns strategy
preferences, not factual truth or model weights.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDecision:
    action: str
    confidence: float
    reason: str
    alternatives: tuple[str, ...]


# Strategy IDs are semantic labels, not response text or language-specific phrases.
_BASELINE = {
    "answer_directly": 0.55,
    "ask_clarification": 0.50,
    "acknowledge_correction": 0.65,
    "continue_context": 0.60,
    "acknowledge_statement": 0.45,
    "greet": 0.60,
}


def _strategy_for_act(speech_act):
    return {
        "greeting": "greet",
        "question": "answer_directly",
        "request": "answer_directly",
        "follow_up": "continue_context",
        "correction": "acknowledge_correction",
        "statement": "acknowledge_statement",
        "uncertain_statement": "ask_clarification",
        "empty": "ask_clarification",
    }.get(speech_act, "ask_clarification")


class CommunicationPolicy:
    def __init__(self, memory=None):
        self.memory = memory

    def choose(self, frame, state=None):
        preferred = _strategy_for_act(frame.speech_act)
        candidates = [preferred]
        if frame.confidence < 0.6 and "ask_clarification" not in candidates:
            candidates.append("ask_clarification")
        if frame.speech_act == "statement" and "answer_directly" not in candidates:
            candidates.append("answer_directly")

        stats = self.memory.strategy_statistics() if self.memory and hasattr(self.memory, "strategy_statistics") else {}
        scored = []
        for action in candidates:
            item = stats.get(action, {})
            observations = int(item.get("count", 0))
            # Shrink sparse feedback toward the baseline to avoid overreacting to one vote.
            prior = _BASELINE.get(action, 0.5)
            score = (prior * 3 + float(item.get("mean_score", 0.5)) * observations) / (3 + observations)
            scored.append((score, action, observations))
        scored.sort(reverse=True)
        score, action, observations = scored[0]
        reason = (
            f"Selected '{action}' from speech act '{frame.speech_act}'. "
            f"Observed feedback: {observations} rating(s) for this strategy."
        )
        return StrategyDecision(
            action=action,
            confidence=max(0.0, min(1.0, score)),
            reason=reason,
            alternatives=tuple(item[1] for item in scored[1:]),
        )

    def record_outcome(self, task_id, action, score, feedback=""):
        if self.memory and hasattr(self.memory, "record_strategy_outcome"):
            return self.memory.record_strategy_outcome(task_id, action, score, feedback)
        return False
