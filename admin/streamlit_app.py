"""Private Streamlit control panel for the shared EvolveAI API.

Deploy this folder as a separate Streamlit app. Configure secrets:
ADMIN_PASSWORD, EVOLVEAI_ADMIN_TOKEN, EVOLVEAI_API_URL.
"""
import hmac
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="EvolveAI Control Room", page_icon="🧠", layout="wide")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or os.getenv(name, default))


ADMIN_PASSWORD = secret("ADMIN_PASSWORD")
ADMIN_TOKEN = secret("EVOLVEAI_ADMIN_TOKEN")
API_URL = secret("EVOLVEAI_API_URL").rstrip("/")

st.markdown("""
<style>
  .stApp { background: #0b1020; }
  [data-testid="stSidebar"] { background: #10182b; }
  div[data-testid="stMetric"] { background: #121c31; padding: 16px; border: 1px solid #26344e; border-radius: 12px; }
  .hero { padding: 1.2rem 1.4rem; border: 1px solid #26344e; border-radius: 16px; background: linear-gradient(120deg,#111a2f,#14243a); margin-bottom: 1rem; }
  .muted { color: #9caec8; }
  .brain-panel { padding: 1.4rem; border: 1px solid #30476d; border-radius: 18px; background: radial-gradient(circle at 85% 10%, #253d63 0%, #111a2f 45%, #0b1020 100%); margin-bottom: 1rem; box-shadow: 0 0 28px rgba(54,112,210,.10); }
  .brain-panel h2 { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.5rem 0; }
  .eyebrow { letter-spacing:.18em; color:#79c8ff; font-size:.72rem; font-weight:700; }
  .neon { color:#73e2ff; font-size:2rem; text-shadow:0 0 16px rgba(115,226,255,.35); }
  .neon small { color:#9caec8; font-size:.9rem; text-shadow:none; }
  .roadmap { display:flex; gap:1rem; align-items:flex-start; padding:.9rem 1rem; border:1px solid #26344e; border-radius:12px; background:#10182b; margin:.55rem 0; }
  .roadmap-number { color:#73e2ff; font-weight:800; font-size:1.1rem; }
</style>
""", unsafe_allow_html=True)


def api(action: str, **payload):
    if not API_URL:
        st.error("EVOLVEAI_API_URL is not configured. Set it to your deployed Vercel endpoint, e.g. https://your-app.vercel.app/api/chat.")
        return None
    if not ADMIN_TOKEN:
        st.error("EVOLVEAI_ADMIN_TOKEN is not configured in this app's secrets.")
        return None
    try:
        response = requests.post(
            API_URL,
            json={"action": action, **payload},
            headers={"X-Admin-Token": ADMIN_TOKEN},
            timeout=35,
        )
        if response.status_code == 401:
            st.error("Admin API rejected the token. Check that both deployments use the same EVOLVEAI_ADMIN_TOKEN.")
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach EvolveAI API: {exc}")
        return None


def maturity_report(stats: dict, storage: str) -> dict:
    """Evidence-based engineering readiness score, not a measure of intelligence."""
    rated = int(stats.get("rated_tasks", 0) or 0)
    lessons = int(stats.get("lessons", 0) or 0)
    ghost = int(stats.get("ghost_knowledge", 0) or 0)
    persistent = storage != "sqlite_ephemeral"
    components = [
        {"Subsystem": "Persistent memory", "Score": 20 if persistent else 0, "Max": 20,
         "State": "Configured path (verify durability)" if persistent else "Blocked · ephemeral SQLite",
         "Next step": "Verify shared durable database" if persistent else "Move memory to managed PostgreSQL"},
        {"Subsystem": "Semantic vector recall", "Score": 0, "Max": 20,
         "State": "Not implemented", "Next step": "Add embeddings + vector index + retrieval tests"},
        {"Subsystem": "Feedback loop", "Score": min(15, round(15 * rated / 25)), "Max": 15,
         "State": f"{rated} rated tasks", "Next step": "Collect at least 25 ratings and evaluate improvements"},
        {"Subsystem": "Knowledge consolidation", "Score": min(15, round(15 * (lessons + ghost) / 50)), "Max": 15,
         "State": f"{lessons} lessons · {ghost} research records",
         "Next step": "Deduplicate, link concepts, and verify candidate facts"},
        {"Subsystem": "Always-on background worker", "Score": 0, "Max": 15,
         "State": "Not implemented", "Next step": "Deploy a separate scheduled/continuous worker"},
        {"Subsystem": "Evaluation & regression tests", "Score": 0, "Max": 15,
         "State": "Not measured by runtime metrics", "Next step": "Create a benchmark set and pass/fail regression gate"},
    ]
    score = sum(item["Score"] for item in components)
    label = "Foundation" if score < 25 else "Emerging" if score < 50 else "Developing" if score < 75 else "Advanced"
    return {"score": score, "label": label, "components": components}


def render_maturity(data: dict):
    stats = data.get("stats", {})
    storage = data.get("storage", "unknown")
    report = maturity_report(stats, storage)
    score = report["score"]
    st.markdown(
        f'<div class="brain-panel"><div class="eyebrow">EVOLVEAI / COGNITIVE SYSTEM STATUS</div>'
        f'<h2>Brain maturity index <span class="neon">{score}<small>/100</small></span></h2>'
        f'<p class="muted">Stage: <b>{report["label"]}</b> · Engineering capability score, not a claim about intelligence.</p></div>',
        unsafe_allow_html=True,
    )
    st.progress(score / 100, text=f"Measured readiness · {score}%")
    a, b, c, d = st.columns(4)
    a.metric("Experiences", int(stats.get("tasks", 0) or 0))
    b.metric("Feedback signals", int(stats.get("rated_tasks", 0) or 0))
    c.metric("Lessons", int(stats.get("lessons", 0) or 0))
    d.metric("Research records", int(stats.get("ghost_knowledge", 0) or 0))
    st.subheader("Subsystem maturity")
    frame = pd.DataFrame(report["components"])
    frame["Readiness"] = frame.apply(lambda row: f'{row["Score"]}/{row["Max"]}', axis=1)
    st.dataframe(frame[["Subsystem", "Readiness", "State", "Next step"]], use_container_width=True, hide_index=True)
    st.subheader("Evolution roadmap")
    roadmap = [
        ("01", "Make memory durable", "Shared database; prove data survives cold starts and redeployments."),
        ("02", "Give recall semantic meaning", "Embedding model, vector index, hybrid search, source-aware ranking."),
        ("03", "Make learning measurable", "Feedback-driven lesson proposals, confidence tracking and controlled promotion."),
        ("04", "Run the brain continuously", "Background consolidation worker with health checks and retry-safe jobs."),
        ("05", "Prove improvement", "Benchmark the same prompts over time; reject changes that regress."),
    ]
    for number, title, detail in roadmap:
        st.markdown(
            f'<div class="roadmap"><span class="roadmap-number">{number}</span>'
            f'<div><b>{title}</b><div class="muted">{detail}</div></div></div>',
            unsafe_allow_html=True,
        )
    st.caption("The score is intentionally conservative. Vector recall, background processing, and a regression benchmark score zero until implemented and measured.")


def require_login():
    if not ADMIN_PASSWORD:
        st.error("Admin login is locked until ADMIN_PASSWORD is set in Streamlit secrets.")
        st.code('ADMIN_PASSWORD = "use-a-long-unique-password"\nEVOLVEAI_ADMIN_TOKEN = "use-a-different-long-random-secret"\nEVOLVEAI_API_URL = "https://YOUR-VERCEL-DOMAIN/api/chat"')
        st.stop()
    if st.session_state.get("admin_authenticated"):
        return
    st.markdown('<div class="hero"><h1>🔐 EvolveAI Control Room</h1><p class="muted">Private administrator access</p></div>', unsafe_allow_html=True)
    with st.form("admin_login"):
        password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Unlock control room", type="primary", use_container_width=True)
    if submitted:
        if hmac.compare_digest(password, ADMIN_PASSWORD):
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_login()

with st.sidebar:
    st.markdown("## 🧠 EvolveAI")
    st.caption("Private control room")
    page = st.radio("Control panel", ["Brain Activity", "Brain Maturity", "Circuit Board", "Engine Console", "Memory", "Learning Lab", "Diagnostics"])
    st.divider()
    st.caption("Backend")
    st.code(API_URL or "EVOLVEAI_API_URL not configured", language=None)
    if st.button("Log out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

st.markdown('<div class="hero"><h1>Control room</h1><p class="muted">One engine. Shared memory. Administrative controls only.</p></div>', unsafe_allow_html=True)

if page == "Brain Activity":
    st.subheader("Brain activity journal")
    st.caption("Inspect recorded agent actions and outcomes. This is observable telemetry, not a transcript of private internal thoughts.")
    a, b, d = st.columns([1, 1, 2])
    category = a.selectbox("Category", ["All", "user_task", "background_research", "learning", "memory", "experiment", "system"])
    status = b.selectbox("Status", ["All", "started", "running", "completed", "failed", "waiting", "proposed", "rejected"])
    query = d.text_input("Search events")
    if st.button("Refresh events"):
        st.rerun()
    result = api("admin_activity", limit=200, category="" if category == "All" else category, status="" if status == "All" else status, query=query)
    if result:
        stats = result.get("activity_stats", {})
        counts = stats.get("by_status", {})
        cols = st.columns(4)
        cols[0].metric("Events", stats.get("events", 0))
        cols[1].metric("Running", counts.get("running", 0) + counts.get("started", 0))
        cols[2].metric("Completed", counts.get("completed", 0))
        cols[3].metric("Failed", counts.get("failed", 0))
        events = result.get("events", [])
        if events:
            st.dataframe(pd.DataFrame([{"Time UTC": e.get("created_at"), "Category": e.get("category"), "Event": e.get("event"), "Status": e.get("status"), "Summary": e.get("summary"), "Task ID": e.get("task_id")} for e in events]), use_container_width=True, hide_index=True)
            with st.expander("Inspect event details"):
                options = {f'{e.get("created_at")} · {e.get("event")} · {e.get("id")}': e for e in events}
                selected = st.selectbox("Recorded event", list(options))
                st.json(options[selected])
        else:
            st.info("No events match these filters.")
    st.warning("The always-on worker is not deployed yet. This journal records instrumented API activity and will also display background tasks once a worker is added.")

elif page == "Brain Maturity":
    data = api("admin_overview")
    if data:
        render_maturity(data)

elif page == "Circuit Board":
    data = api("admin_overview")
    if data:
        stats = data.get("stats", {})
        a, b, c, d, e = st.columns(5)
        a.metric("Tasks", stats.get("tasks", 0))
        b.metric("Rated", stats.get("rated_tasks", 0))
        c.metric("Memories", stats.get("memories", 0))
        d.metric("Lessons", stats.get("lessons", 0))
        e.metric("Ghost knowledge", stats.get("ghost_knowledge", 0))
        report = maturity_report(stats, data.get("storage", "unknown"))
        st.markdown(
            f'<div class="brain-panel"><div class="eyebrow">COGNITIVE READINESS</div>'
            f'<h2>Brain maturity <span class="neon">{report["score"]}<small>/100</small></span></h2>'
            f'<p class="muted">{report["label"]} · Based on implemented and measured subsystems, not simulated activity.</p></div>',
            unsafe_allow_html=True,
        )
        st.progress(report["score"] / 100, text="Brain maturity index")
        st.subheader("Processing circuit")
        nodes = [
            ("01 · Input", "Message validation", "Ready"),
            ("02 · Understand", "Intent and conversation state", "Ready"),
            ("03 · Recall", "Personal memory and prior lessons", "Ready"),
            ("04 · Decide", "Rules, known facts, or web research", "Ready"),
            ("05 · Respond", "Response adaptation and task record", "Ready"),
            ("06 · Learn", "Feedback and strategy outcomes", f'{stats.get("strategy_feedback", 0)} feedback records'),
        ]
        for i in range(0, len(nodes), 2):
            cols = st.columns(2)
            for col, node in zip(cols, nodes[i:i + 2]):
                with col:
                    st.markdown(f"**{node[0]} — {node[2]}**")
                    st.caption(node[1])
                    st.progress(1.0)
        st.subheader("Communication strategy outcomes")
        strategies = data.get("strategies", {})
        if strategies:
            frame = pd.DataFrame([{"Strategy": key, **value} for key, value in strategies.items()])
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.info("No strategy feedback has been recorded yet.")
        if data.get("storage") == "sqlite_ephemeral":
            st.warning("Storage is currently SQLite on Vercel's ephemeral filesystem. Do not rely on this for persistent learning yet.")
        st.caption("Circuit stages are architectural indicators, not live telemetry. Open Brain Maturity for measured gaps and the roadmap.")

elif page == "Engine Console":
    st.subheader("Run a test prompt")
    st.caption("Tests use the same agent and memory as the user-facing API. Each test is saved as a task.")
    prompt = st.text_area("Prompt", placeholder="Ask EvolveAI a test question…", height=130)
    if st.button("Run engine", type="primary", disabled=not prompt.strip()):
        result = api("admin_test", message=prompt.strip())
        if result:
            st.markdown("### Response")
            st.markdown(result.get("answer", ""))
            left, right = st.columns(2)
            left.metric("Mode", result.get("mode", "unknown"))
            right.metric("Task ID", result.get("task_id", "—"))
            with st.expander("Inspect reasoning plan"):
                st.json(result.get("reasoning_plan", {}))

elif page == "Memory":
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Teach a memory")
        with st.form("add_memory"):
            key = st.text_input("Memory key", placeholder="preference")
            value = st.text_area("Memory value", placeholder="Prefers concise, direct answers")
            add = st.form_submit_button("Save memory", type="primary")
        if add:
            result = api("admin_add_memory", key=key, value=value)
            if result and result.get("saved"):
                st.success("Memory saved to the engine.")
                st.rerun()
    with right:
        st.subheader("Delete a memory")
        current = api("admin_memories")
        memories = current.get("memories", []) if current else []
        if memories:
            labels = {f'{item["key"]} — {item["value"][:70]}': item["key"] for item in memories}
            selected = st.selectbox("Select memory", list(labels))
            if st.button("Delete selected memory", type="secondary"):
                result = api("admin_delete_memory", key=labels[selected])
                if result and result.get("deleted"):
                    st.success("Memory deleted.")
                    st.rerun()
        else:
            st.info("No saved memories.")
    st.subheader("Saved memories")
    current = api("admin_memories")
    if current and current.get("memories"):
        st.dataframe(pd.DataFrame(current["memories"]), use_container_width=True, hide_index=True)
    elif current:
        st.info("No memories saved.")

elif page == "Learning Lab":
    query = st.text_input("Filter lessons", placeholder="Search lesson text or task")
    data = api("admin_lessons", query=query, limit=100)
    if data and data.get("lessons"):
        st.dataframe(pd.DataFrame(data["lessons"]), use_container_width=True, hide_index=True)
    elif data:
        st.info("No matching lessons.")
    st.subheader("Strategy feedback")
    overview = api("admin_overview")
    if overview and overview.get("strategies"):
        rows = [{"Strategy": key, **value} for key, value in overview["strategies"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Recent feedback and tasks")
    tasks = api("admin_tasks", limit=100)
    if tasks and tasks.get("tasks"):
        frame = pd.DataFrame(tasks["tasks"])
        if "prompt" in frame:
            frame["prompt"] = frame["prompt"].astype(str).str.slice(0, 180)
        if "answer" in frame:
            frame["answer"] = frame["answer"].astype(str).str.slice(0, 220)
        st.dataframe(frame, use_container_width=True, hide_index=True)

elif page == "Diagnostics":
    st.subheader("System health")
    if st.button("Refresh diagnostics"):
        st.rerun()
    health = api("admin_overview")
    if health:
        st.success("Admin API authenticated and responding.")
        st.json({
            "service": health.get("api"),
            "storage_mode": health.get("storage"),
            "database_stats": health.get("stats", {}),
        })
    st.subheader("Ghost Knowledge")
    query = st.text_input("Search stored knowledge", key="ghost_query")
    knowledge = api("admin_ghost", query=query, limit=100)
    if knowledge and knowledge.get("knowledge"):
        st.dataframe(pd.DataFrame(knowledge["knowledge"]), use_container_width=True, hide_index=True)
    elif knowledge:
        st.info("No matching stored knowledge.")
    st.subheader("Recent engine tasks")
    tasks = api("admin_tasks", limit=50)
    if tasks and tasks.get("tasks"):
        st.dataframe(pd.DataFrame(tasks["tasks"]), use_container_width=True, hide_index=True)

st.caption("EvolveAI Admin • Keep secrets out of source control. Admin endpoints are protected server-side.")
