import ast
import operator
import re
from .instructions import CORE_RULES, LEARNING_POLICY
from .memory import ExperienceMemory


_SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_math(expression):
    """Evaluate a small arithmetic expression without eval() or arbitrary code."""
    try:
        tree = ast.parse(expression, mode="eval")
        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 8:
                    raise ValueError("Exponent too large")
                result = _SAFE_OPERATORS[type(node.op)](left, right)
                if abs(result) > 1e15:
                    raise ValueError("Result too large")
                return result
            if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
                return _SAFE_OPERATORS[type(node.op)](visit(node.operand))
            raise ValueError("Unsupported expression")
        return visit(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


class EvolveAgent:
    """Instruction-driven assistant with explicit memory and feedback retrieval."""

    def __init__(self, memory: ExperienceMemory, skills=None):
        self.memory = memory
        self.skills = skills

    def _teach_memory(self, prompt):
        patterns = [
            r"^(?:please )?remember that my (.+?) is (.+?)[.!]?$",
            r"^(?:please )?remember that i (prefer|like|want) (.+?)[.!]?$",
            r"^my name is (.+?)[.!]?$",
            r"^call me (.+?)[.!]?$",
            r"^i prefer (.+?)[.!]?$",
        ]
        p = prompt.strip()
        m = re.match(patterns[0], p, re.I)
        if m:
            key, value = "my " + m.group(1).strip().lower(), m.group(2).strip()
        else:
            m = re.match(patterns[1], p, re.I)
            if m:
                key, value = "preference", m.group(1).lower() + " " + m.group(2).strip()
            else:
                m = re.match(patterns[2], p, re.I)
                if m:
                    key, value = "my name", m.group(1).strip()
                else:
                    m = re.match(patterns[3], p, re.I)
                    if m:
                        key, value = "my name", m.group(1).strip()
                    else:
                        m = re.match(patterns[4], p, re.I)
                        if not m:
                            return None
                        key, value = "preference", m.group(1).strip()
        self.memory.remember(key, value)
        return f"I’ll remember **{key}**: **{value}**. You can review or delete saved memories from the menu."

    def _answer(self, prompt, lessons, memories):
        p = prompt.strip()
        low = p.lower()

        taught = self._teach_memory(p)
        if taught:
            return taught

        if any(x in low for x in ["what do you remember", "show my memories", "what have you learned about me"]):
            if not memories:
                return "I don’t have any saved memories about you yet. Tell me something with **“Remember that…”** and I can store it."
            return "Here’s what I currently remember:\n\n" + "\n".join(
                f"- **{item['key'].title()}:** {item['value']}" for item in memories
            )

        if low in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi evolveai", "hello evolveai"}:
            name = next((m["value"] for m in memories if m["key"] == "my name"), None)
            return f"Hey {name}! I’m here. What would you like to work through?" if name else "Hey! I’m EvolveAI. What would you like to work through?"

        # Basic arithmetic is handled locally, without a model or arbitrary code execution.
        math_text = re.sub(r"^(?:what is|calculate|compute)\s+", "", p, flags=re.I).rstrip(" ?")
        if re.fullmatch(r"[\d\s.+*/()%\-]+", math_text) and any(c.isdigit() for c in math_text):
            result = _safe_math(math_text)
            if result is not None:
                return f"**{math_text} = {result:g}**"

        if low in {"help", "what can you do", "what can you do?"}:
            return (
                "I work through written instructions rather than an external AI model. Right now I can:\n\n"
                "- Remember facts and preferences you explicitly teach me\n"
                "- Retrieve related lessons from earlier conversations\n"
                "- Use your 👍 / 👎 feedback to change which examples influence future replies\n"
                "- Handle basic arithmetic and simple structured tasks\n\n"
                "I’m still a prototype, so I’ll tell you when a request is beyond my current abilities."
            )

        related = [x for x in lessons if any(w in (x["task"] + " " + x["lesson"]).lower()
                                              for w in re.findall(r"\w{4,}", low))]
        memory_context = "\n".join(f"- {m['key']}: {m['value']}" for m in memories[:8])
        if related:
            experience = "\n\n**Related experience from earlier chats:**\n" + "\n".join(
                f"- {item['lesson'][:260]}" for item in related[:2]
            )
        else:
            experience = ""
        known = "\n\n**Relevant saved context:**\n" + memory_context if memory_context else ""

        return (
            "I understand the request, but my current instruction set does not yet include a reliable rule "
            "for answering it directly. I won’t invent an answer.\n\n"
            "**Let’s make progress:** tell me the outcome you want and any important constraints, or teach me "
            "a rule with **“When I ask about X, do Y.”** I can store explicit facts and learn from feedback."
            + experience + known
        )

    def run(self, prompt):
        memories = self.memory.memories()
        lessons = self.memory.search_lessons(prompt, limit=8)
        answer = self._answer(prompt, lessons, memories)
        task_id = self.memory.save_task(prompt, answer, "rules")
        return {
            "task_id": task_id, "answer": answer, "mode": "rules",
            "lessons_considered": lessons, "skill_suggestions": [],
            "instructions_count": len(CORE_RULES), "learning_rules_count": len(LEARNING_POLICY),
        }
