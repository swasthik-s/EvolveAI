import os
import re
from .memory import ExperienceMemory

SYSTEM_PROMPT = """You are EvolveAI, a careful task-oriented assistant. Use retrieved lessons as fallible context, not guaranteed truth. Be transparent about uncertainty. Do not claim to have executed code, browsed the web, or verified facts unless a tool actually did so. Never reveal secrets. Never propose destructive actions without explicit user approval."""


class EvolveAgent:
    def __init__(self, memory: ExperienceMemory, skills):
        self.memory = memory
        self.skills = skills

    def _model_answer(self, prompt, lessons):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._demo_answer(prompt, lessons), "demo"
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            lesson_text = "\n".join(f"- {x['lesson']} (confidence {x['confidence']:.2f})" for x in lessons) or "- No relevant stored lessons."
            response = client.chat.completions.create(model=os.getenv("EVOLVEAI_MODEL", "gpt-4.1-mini"), messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "system", "content": "Relevant past lessons:\n" + lesson_text}, {"role": "user", "content": prompt}], temperature=0.3)
            return response.choices[0].message.content or "The model returned an empty response.", "openai"
        except Exception:
            return ("The configured AI provider request failed. Check API key, model name, network access, and usage limits. Demo response follows.\n\n" + self._demo_answer(prompt, lessons)), "fallback"

    def _demo_answer(self, prompt, lessons):
        relevant = "\n".join(f"- {x['lesson']}" for x in lessons[:3]) or "- No relevant lessons found."
        return ("### Demo-mode response\n\nThe hosted language model is not configured, so this is a deterministic prototype response rather than an LLM-generated answer.\n\n" + f"**Your task:** {prompt}\n\n**Relevant stored experience:**\n{relevant}\n\n" + "**Suggested next steps:**\n1. Clarify the desired output and constraints.\n2. Break the task into small, verifiable steps.\n3. Check the result against evidence or test cases.\n4. Record a lesson only after the outcome has been evaluated.\n\nSet OPENAI_API_KEY to enable hosted-model reasoning.")

    def run(self, prompt):
        lessons = self.memory.search_lessons(prompt, limit=5)
        answer, mode = self._model_answer(prompt, lessons)
        suggestions = []
        if re.search(r"\b(skill|reusable function|python helper)\b", prompt, re.I):
            suggestions.append("You can save a reviewed helper as a skill draft in the Skills tab.")
        task_id = self.memory.save_task(prompt, answer, mode)
        return {"task_id": task_id, "answer": answer, "mode": mode, "lessons_considered": lessons, "skill_suggestions": suggestions}
