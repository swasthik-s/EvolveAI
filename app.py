import os
import streamlit as st
import pandas as pd
from evolveai.agent import EvolveAgent
from evolveai.memory import ExperienceMemory
from evolveai.skills import SkillLibrary

st.set_page_config(page_title="EvolveAI", page_icon="🧠", layout="wide")

@st.cache_resource
def get_services():
    db_path = os.getenv("EVOLVEAI_DB_PATH", "data/evolveai.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    memory = ExperienceMemory(db_path)
    skills = SkillLibrary(os.getenv("EVOLVEAI_SKILLS_PATH", "data/skills"))
    return memory, skills, EvolveAgent(memory=memory, skills=skills)

memory, skills, agent = get_services()
st.title("🧠 EvolveAI")
st.caption("An experimental agent that improves through recorded experience, reusable skills, and measurable evaluation.")
with st.sidebar:
    st.header("Control room")
    st.write("**Learning mode:** experience + evaluation")
    st.write("**Model provider:** " + ("OpenAI API" if os.getenv("OPENAI_API_KEY") else "Demo mode"))
    st.warning("Generated code is never executed automatically. Skills are saved as drafts for review.")
    if st.button("Refresh dashboard"):
        st.rerun()

tab_chat, tab_growth, tab_memory, tab_skills, tab_settings = st.tabs(["💬 Agent", "📈 Growth", "🧠 Memory", "🧰 Skills", "⚙️ Settings"])
with tab_chat:
    st.subheader("Give the agent a task")
    st.write("The agent consults relevant past lessons and records feedback for future attempts.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("Ask a question or give EvolveAI a task…")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking and consulting relevant experience…"):
                result = agent.run(prompt)
            st.markdown(result["answer"])
            with st.expander("Task trace"):
                st.json({"mode": result["mode"], "lessons_considered": result["lessons_considered"], "skill_suggestions": result["skill_suggestions"], "task_id": result["task_id"]})
            st.caption("Rate this result to help evaluate future approaches.")
            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("👍 Helpful", key=f"good_{result['task_id']}"):
                memory.rate_task(result["task_id"], 1.0, "User marked helpful")
                st.toast("Feedback recorded.")
            if c2.button("👎 Not helpful", key=f"bad_{result['task_id']}"):
                memory.rate_task(result["task_id"], 0.0, "User marked not helpful")
                st.toast("Feedback recorded.")
        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
with tab_growth:
    st.subheader("Growth dashboard")
    stats = memory.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasks recorded", stats["tasks"])
    c2.metric("Rated tasks", stats["rated_tasks"])
    c3.metric("User feedback success", f"{stats['success_rate']:.0%}" if stats["rated_tasks"] else "—")
    c4.metric("Approved skills", skills.count_approved())
    st.caption("Metrics reflect recorded tasks and explicit user feedback, not proof that the underlying model's intelligence increased.")
    history = memory.recent_tasks(100)
    if history:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        rated = [x for x in history if x.get("score") is not None]
        if rated:
            chart = pd.DataFrame([{"Task": x["id"], "Score": x["score"]} for x in reversed(rated)])
            st.line_chart(chart.set_index("Task"))
    else:
        st.info("No experience recorded yet. Complete a task in the Agent tab to begin.")
with tab_memory:
    st.subheader("Experience memory")
    st.write("Memory stores task summaries, user feedback, and lessons. Retrieved lessons are suggestions, not unquestionable truth.")
    query = st.text_input("Search past experiences", placeholder="e.g. CSV validation")
    lessons = memory.search_lessons(query, limit=30)
    if lessons:
        st.dataframe(pd.DataFrame(lessons), use_container_width=True, hide_index=True)
    else:
        st.info("No matching lessons yet.")
    with st.expander("Add a lesson manually"):
        with st.form("manual_lesson"):
            task = st.text_input("Task or situation")
            lesson = st.text_area("Lesson learned")
            confidence = st.slider("Confidence", 0.1, 1.0, 0.7, 0.1)
            submitted = st.form_submit_button("Save lesson")
            if submitted:
                if task.strip() and lesson.strip():
                    memory.add_lesson(task.strip(), lesson.strip(), confidence, source="user")
                    st.success("Lesson saved.")
                    st.rerun()
                else:
                    st.error("Enter both a task and a lesson.")
with tab_skills:
    st.subheader("Reusable skills")
    st.write("Review code before use. This app does not execute generated code.")
    drafts = skills.list_all()
    if drafts:
        for skill in drafts:
            with st.expander(f"{skill['name']} · {skill['status']}"):
                st.markdown(skill["description"])
                st.code(skill["content"], language="python")
                if skill["status"] == "draft":
                    if st.button("Approve this skill", key=f"approve_{skill['id']}"):
                        skills.set_status(skill["id"], "approved")
                        st.success("Skill marked approved.")
                        st.rerun()
                    if st.button("Reject this skill", key=f"reject_{skill['id']}"):
                        skills.set_status(skill["id"], "rejected")
                        st.info("Skill rejected.")
                        st.rerun()
    else:
        st.info("No skill drafts yet. Add one below.")
    with st.expander("Create a skill draft manually"):
        with st.form("new_skill"):
            name = st.text_input("Skill name")
            desc = st.text_area("Purpose")
            code = st.text_area("Python code", height=180)
            if st.form_submit_button("Save as draft"):
                if name.strip() and desc.strip() and code.strip():
                    skills.create_draft(name, desc, code, source="user")
                    st.success("Draft saved for review.")
                    st.rerun()
                else:
                    st.error("Fill in all fields.")
with tab_settings:
    st.subheader("Configuration")
    st.markdown("Set OPENAI_API_KEY in Streamlit Secrets or your environment to enable hosted-model responses. Without a key, the app runs in demo mode. Local disk on some hosted platforms may be ephemeral. Never expose API keys in source code, commit history, logs, or screenshots. Arbitrary Python is not executed.")
    st.code("streamlit run app.py", language="bash")
