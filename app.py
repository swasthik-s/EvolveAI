import os
import time
import streamlit as st
from evolveai.agent import EvolveAgent
from evolveai.memory import ExperienceMemory

st.set_page_config(
    page_title="EvolveAI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { max-width: 760px; padding-top: 1rem; padding-bottom: 5rem; }
    [data-testid="stChatMessage"] { padding: .8rem .9rem; border-radius: 1.1rem; border: 1px solid rgba(128,128,128,.12); }
    [data-testid="stChatInput"] { padding-bottom: env(safe-area-inset-bottom); }
    div[data-testid="stHorizontalBlock"] { gap: .45rem; }
    .evolve-subtitle { color: var(--text-color); opacity: .68; font-size: .9rem; margin-top: -.55rem; }
    [data-testid="stStatusWidget"] { border-radius: .9rem; }
    [data-testid="stChatMessage"] p { line-height: 1.62; }
    .stButton button { border-radius: .8rem; }
    @media (max-width: 640px) {
        .block-container { padding-left: .75rem; padding-right: .75rem; padding-top: .65rem; }
        h1 { font-size: 1.7rem !important; }
        [data-testid="stChatMessage"] { padding: .55rem .6rem; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_memory():
    db_path = os.getenv("EVOLVEAI_DB_PATH", "data/evolveai.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return ExperienceMemory(db_path)


memory = get_memory()
agent = EvolveAgent(memory)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = True

top_left, top_right = st.columns([3, 1])
with top_left:
    st.title("EvolveAI")
    st.markdown('<p class="evolve-subtitle">A personal assistant that recalls, researches, and learns from feedback.</p>', unsafe_allow_html=True)
with top_right:
    st.caption("Preview")
    st.caption("Build: conversation-learning-v1")
    if st.button("New chat", key="new_chat_top", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with st.sidebar:
    st.subheader("Your EvolveAI")
    stats = memory.stats()
    st.metric("Saved memories", stats["memories"])
    st.metric("Learning examples", stats["lessons"])
    st.metric("Ghost Knowledge", stats.get("ghost_knowledge", 0))
    st.caption("On Streamlit Community Cloud, local files may reset after a restart or redeploy.")
    with st.expander("Review saved memories", expanded=False):
        saved = memory.memories()
        if not saved:
            st.caption("Nothing saved yet.")
        for item in saved:
            st.markdown(f"**{item['key'].title()}**")
            st.write(item["value"])
            if st.button("Forget", key=f"forget_{item['key']}"):
                memory.forget(item["key"])
                st.rerun()
    with st.expander("Ghost Knowledge", expanded=False):
        saved_knowledge = memory.search_ghost_knowledge("", limit=20)
        if not saved_knowledge:
            st.caption("No web research saved yet. EvolveAI searches only when a question appears outside its known rules and saved knowledge.")
        for item in saved_knowledge:
            st.markdown(f"**{item['title']}** · {item['status']}")
            st.write(item["fact"])
            st.caption(f"Relevance score: {item['confidence']:.0%} · Last checked: {item['last_checked_at'][:10]}")
            st.markdown(f"[Open source]({item['source_url']})")
    with st.expander("How it works"):
        st.write("1. It checks programmed rules, personal memory, and Ghost Knowledge first.")
        st.write("2. If a factual question is outside its knowledge, it tries free public web search.")
        st.write("3. It stores page excerpts and source links as research candidates, not guaranteed facts.")
        st.write("4. It reuses saved evidence for similar questions, but refreshes time-sensitive questions.")
        st.write("5. Feedback adds examples; it does not train model weights or execute generated code.")
    if st.button("Clear this chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if not st.session_state.chat_history:
    st.markdown("### What would you like to work on?")
    st.write("Start with a message, or try one of these:")
    examples = [
        "Remember that my name is Swasthik",
        "What can you do?",
        "What do you remember about me?",
    ]
    cols = st.columns(1)
    for i, example in enumerate(examples):
        if st.button(example, key=f"starter_{i}", use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("task_id"):
            feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 5])
            if feedback_col1.button("👍", key=f"good_{msg['task_id']}", help="Helpful"):
                memory.rate_task(msg["task_id"], 1.0, "Helpful")
                st.toast("Feedback saved.")
            if feedback_col2.button("👎", key=f"bad_{msg['task_id']}", help="Not helpful"):
                memory.rate_task(msg["task_id"], 0.0, "Not helpful")
                st.toast("Feedback saved.")

pending = st.session_state.pop("pending_prompt", None)
prompt = st.chat_input("Message EvolveAI…")
if pending:
    prompt = pending

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        status = st.status("Thinking through your message…", expanded=True)
        started_at = time.perf_counter()

        def show_stage(label):
            status.update(label=f"{label}…", state="running", expanded=True)
            time.sleep(0.12)

        try:
            result = agent.run(
                prompt,
                history=st.session_state.chat_history[:-1],
                progress=show_stage,
            )
            # Keep a brief, visible response transition for very fast local answers.
            remaining = 0.55 - (time.perf_counter() - started_at)
            if remaining > 0:
                time.sleep(remaining)
            status.update(label="Ready", state="complete", expanded=False)
        except Exception:
            status.update(label="Something went wrong while processing this message", state="error", expanded=True)
            raise
        st.markdown(result["answer"])
        st.caption("Was this useful? Your feedback changes which examples I use later.")
        c1, c2, _ = st.columns([1, 1, 5])
        if c1.button("👍 Helpful", key=f"new_good_{result['task_id']}"):
            memory.rate_task(result["task_id"], 1.0, "Helpful")
            st.toast("Feedback saved.")
        if c2.button("👎", key=f"new_bad_{result['task_id']}"):
            memory.rate_task(result["task_id"], 0.0, "Not helpful")
            st.toast("Feedback saved.")
    st.session_state.chat_history.append({
        "role": "assistant", "content": result["answer"], "task_id": result["task_id"]
    })
    st.rerun()
