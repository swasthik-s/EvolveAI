import ast
import operator
import re
from .instructions import CORE_RULES, LEARNING_POLICY
from .memory import ExperienceMemory
from .research import research
from .thinking import process_evidence
from .reasoning import build_plan
from .cognition import understand
from .relevance import assess_relevance, is_personal_context_query


_QUESTION_STARTS = ("what ", "who ", "when ", "where ", "why ", "how ", "which ", "is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "could ", "would ", "should ", "will ", "tell me about ", "explain ", "define ", "find out ", "look up ")
_CURRENT_TERMS = {"today", "latest", "current", "currently", "now", "recent", "news", "weather", "price", "prices", "stock", "stocks", "exchange", "rate", "schedule", "availability", "updated"}
_STOP_WORDS = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "did", "doing", "have", "has", "had", "will", "would", "could", "should", "can", "the", "and", "for", "from", "with", "that", "this", "about", "into", "your", "you", "are", "was", "were", "is", "not"}


def _terms(text):
    return {w for w in re.findall(r"[a-zA-Z0-9]{3,}", text.lower()) if w not in _STOP_WORDS}


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

    def __init__(self, memory: ExperienceMemory, skills=None, web_research=True):
        self.memory = memory
        self.skills = skills
        self.web_research = web_research

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

    def _personal_identity_answer(self, prompt, memories):
        """Handle user-directed identity questions before general knowledge retrieval."""
        low = re.sub(r"\s+", " ", prompt.strip().lower()).rstrip(" ?.!")

        identity_questions = {
            "who am i", "what is my name", "whats my name", "what's my name",
            "do you know who i am", "do you know my name",
        }
        if low not in identity_questions:
            return None

        name = next((m.get("value", "").strip() for m in memories
                     if m.get("key", "").strip().lower() in {"my name", "name"}), "")
        if name:
            return f"You're {name}, the person I'm talking to. 😄"
        return ("You're the person I'm talking to. I don't have your name saved in my current memory, "
                "so I won't guess. What should I call you?")

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
                "- Search public web pages when I lack relevant knowledge\n"
                "- Save research findings as Ghost Knowledge for future questions\n"
                "- Retrieve related lessons and use your 👍 / 👎 feedback\n"
                "- Handle basic arithmetic and simple structured tasks\n\n"
                "I’m still a prototype: web research extracts text and scores relevance rather than understanding it like a human."
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

    def _is_researchable_question(self, prompt):
        low = prompt.strip().lower()
        if len(low) < 8 or len(low) > 400:
            return False
        if re.match(r"^(?:please )?remember\b|^(?:my name is|call me|i prefer)\b", low):
            return False
        if low.startswith(("what do you remember", "show my memories", "what have you learned about me")):
            return False
        return low.endswith("?") or low.startswith(_QUESTION_STARTS)

    def _known_answer(self, prompt):
        if is_personal_context_query(prompt):
            return None
        query_terms = _terms(prompt)
        if not query_terms or query_terms & _CURRENT_TERMS:
            return None
        for item in self.memory.search_ghost_knowledge(prompt, limit=8):
            relevance = assess_relevance(
                prompt, item.get("title", ""), item.get("fact", ""),
                saved_question=item.get("question", ""),
            )
            if not relevance.accepted:
                continue
            processed = process_evidence(prompt, [{
                "title": item["title"],
                "url": item["source_url"],
                "excerpt": item["fact"],
            }])
            if not processed.get("sources") or not processed.get("answer"):
                continue
            return (
                processed["answer"]
                + "\\n\\nSource: [" + item["title"] + "](" + item["source_url"] + ")"
                + "\\n\\nReused from Ghost Knowledge; check the source if accuracy or freshness matters."
            )
        return None

    def _research_answer(self, prompt, progress=None):
        # Defense in depth: personal-context questions are never sent to web search.
        if is_personal_context_query(prompt):
            return self._personal_identity_answer(prompt, self.memory.memories()) or (
                "You're the person I'm talking to. I don't have enough personal context saved to identify you."
            ), "personal_context"
        # Turn the message into a structured query before searching. This is a
        # transparent symbolic step, not generated language.
        frame = understand(prompt)
        research_query = frame.search_query or prompt
        if progress:
            progress('Searching public sources')
        try:
            evidence = research(research_query, max_results=5)
        except Exception:
            evidence = []
        if not evidence:
            return (
                "I’m not confident enough to answer yet, and free web search returned no usable evidence. "
                "The search service may be temporarily unavailable or rate-limited. "
                "You can rephrase the question or teach me the answer."
            ), "rules"

        if progress:
            progress('Comparing evidence and checking relevance')

        # Filter evidence against the user's actual question before saving or answering.
        relevant_evidence = []
        for item in evidence:
            relevance = assess_relevance(
                prompt, item.get("title", ""), item.get("excerpt", ""),
                saved_question=prompt,
            )
            if relevance.accepted:
                relevant_evidence.append(item)
                self.memory.add_ghost_knowledge(
                    question=prompt, fact=item["excerpt"], title=item["title"],
                    source_url=item["url"],
                    confidence=min(float(item.get("confidence", 0.4)), relevance.score),
                    status="candidate",
                )

        if not relevant_evidence:
            return (
                "I found search results, but none matched your question closely enough. "
                "I won't present unrelated pages as an answer. Could you clarify what you mean?"
            ), "rules"

        processed = process_evidence(prompt, relevant_evidence)
        if not processed["sources"]:
            answer = processed["answer"]
        else:
            source_lines = [
                f"- [{source['title']}]({source['url']})"
                for source in processed["sources"][:3]
            ]
            answer = (
                processed["answer"]
                + "\n\n**Sources**\n"
                + "\n".join(source_lines)
                + "\n\n*I combined relevant statements from public pages. "
                "The evidence is heuristic, so check the sources if precision matters.*"
            )
        return answer, "web_research"

    def run(self, prompt, history=None, progress=None):
        def stage(label):
            if progress:
                progress(label)

        stage("Understanding your message")
        plan = build_plan(prompt, history or [])
        effective_prompt = plan.working_query
        stage("Recalling relevant memory and prior experience")
        memories = self.memory.memories()
        lessons = self.memory.search_lessons(effective_prompt, limit=8)

        # Personal identity is a conversation-context question, never a web-search query.
        identity_answer = self._personal_identity_answer(prompt, memories)
        if identity_answer is not None:
            stage("Answering from personal context")
            task_id = self.memory.save_task(prompt, identity_answer, "personal_context")
            return {
                "task_id": task_id, "answer": identity_answer, "mode": "personal_context",
                "lessons_considered": lessons, "skill_suggestions": [],
                "instructions_count": len(CORE_RULES), "learning_rules_count": len(LEARNING_POLICY),
                "reasoning_plan": {"intent": "personal_identity", "follow_up": False,
                                   "steps": ["interpret_message", "retrieve_personal_memory", "compose_response"]},
            }

        stage("Checking saved knowledge")
        known = self._known_answer(effective_prompt) if plan.intent == "question" and not plan.needs_fresh_research else None
        stage("Choosing the response path")
        if known:
            answer, mode = known, "ghost_knowledge"
        else:
            answer = self._answer(prompt, lessons, memories)
            unknown = "does not yet include a reliable rule" in answer or "does not yet include" in answer
            if self.web_research and plan.intent == "question" and unknown:
                answer, mode = self._research_answer(effective_prompt, progress=progress)
            else:
                mode = "rules"
        stage("Preparing a concise response")
        task_id = self.memory.save_task(prompt, answer, mode)
        return {
            "task_id": task_id, "answer": answer, "mode": mode,
            "lessons_considered": lessons, "skill_suggestions": [],
            "instructions_count": len(CORE_RULES), "learning_rules_count": len(LEARNING_POLICY),
            "reasoning_plan": {"intent": plan.intent, "follow_up": plan.is_follow_up, "steps": list(plan.steps)},
        }
