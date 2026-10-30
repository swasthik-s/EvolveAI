"""Streamlit lab for testing a non-Transformer spiking language model."""
import streamlit as st
import pandas as pd
from train import train
from data import decode
from dataset_harvester import DATASETS, fetch_corpus
from autonomous_lab import run_trials, export_json
from awakening import AwakeningAgent, export_state_json
import json
import torch
import uuid
from pathlib import Path
from governor import run_governed_cycle, load_state

st.set_page_config(page_title="EvolveAI SNN Lab", page_icon="⚡", layout="wide")
st.title("⚡ EvolveAI — Spiking Neural Network Lab")
st.caption("Custom dual-timescale adaptive-threshold SNN · byte-level prediction · no Transformer or attention")
st.info("For NVIDIA T4 GPU experiments, open the [EvolveAI Colab notebook](https://colab.research.google.com/github/swasthik-s/EvolveAI/blob/main/snn/EvolveAI_T4_Autonomous_Learning.ipynb). This Streamlit host uses: " + ("CUDA GPU" if torch.cuda.is_available() else "CPU") + ".")
st.warning("Research prototype only. A lower validation loss is evidence of pattern learning, not proof of reasoning or human-like intelligence.")

DEFAULT_CORPUS = ("""User: Hello.\nAssistant: Hello! What would you like to explore today?\n\nUser: What is learning?\nAssistant: Learning means using experience to improve future predictions or actions.\n\nUser: I made a mistake.\nAssistant: Let's inspect what happened, identify the cause, and try a better approach.\n\nUser: What is a neuron?\nAssistant: A neuron receives signals, integrates them, and may send an electrical spike to other neurons.\n\nUser: Why do predictions fail?\nAssistant: A prediction can fail when evidence differs from expectations. The error can help update the model.\n\nUser: Can you remember a fact?\nAssistant: A learning system can store useful information and retrieve it when relevant.\n\nUser: What should we test?\nAssistant: We should measure held-out loss, sample quality, repeatability, and whether feedback improves performance.\n\n""" * 8)

with st.sidebar:
    st.header("Experiment settings")
    steps = st.slider("Training steps", 10, 300, 60, 10)
    block_size = st.slider("Sequence length (bytes)", 16, 96, 48, 8)
    hidden_size = st.select_slider("Spiking neurons", options=[32, 64, 96, 128], value=64)
    batch_size = st.select_slider("Batch size", options=[2, 4, 8, 12], value=4)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
    architecture = st.selectbox(
        "SNN architecture",
        options=["evolve_context", "baseline"],
        format_func=lambda value: "EvolveAI Context SNN (experimental)" if value == "evolve_context" else "Simple recurrent LIF baseline",
    )

st.subheader("Dataset acquisition")
st.caption("The SNN can fetch a reviewed dataset itself, within a strict size limit. It does not crawl arbitrary websites.")
dataset_label = st.selectbox("Approved dataset", options=list(DATASETS.keys()))
dataset_cap_mb = st.slider("Maximum dataset text to retain (MiB)", min_value=1, max_value=20, value=5)
max_rows = st.number_input("Maximum records (0 = stop at size cap)", min_value=0, max_value=100000, value=0, step=1000)
spec = DATASETS[dataset_label]
st.markdown(f"**Source:** [{spec['id']}]({spec['url']})  \n**License:** {spec['license']}  \n**Use note:** {spec['commercial_use']}")
st.warning("Review the source's current license before use. Fetching data does not mean it is cleared for commercial use.")
if st.button("Fetch approved dataset", type="secondary"):
    fetch_status = st.empty()
    fetch_bar = st.progress(0)
    def on_fetch_progress(records, byte_count, rows_seen):
        fetch_status.write(f"Read {rows_seen:,} rows · retained {records:,} records · {byte_count / (1024 * 1024):.2f} MiB")
        fetch_bar.progress(min(0.99, byte_count / (dataset_cap_mb * 1024 * 1024)))
    try:
        with st.spinner("Streaming approved dataset, cleaning text, and deduplicating…"):
            fetched_text, manifest = fetch_corpus(
                dataset_label, max_mb=float(dataset_cap_mb), max_rows=int(max_rows),
                progress=on_fetch_progress
            )
        st.session_state["snn_corpus"] = fetched_text
        st.session_state["snn_corpus_manifest"] = manifest
        st.session_state.pop("snn_result", None)
        fetch_bar.progress(1.0)
        st.success(f"Fetched {manifest['records_kept']:,} records ({manifest['corpus_bytes']:,} bytes).")
    except Exception as exc:
        st.error(f"Dataset fetch failed: {exc}")

uploaded = st.file_uploader("Or upload a UTF-8 .txt training corpus", type=["txt"])
if uploaded:
    corpus = uploaded.getvalue().decode("utf-8", errors="replace")
    corpus_source = f"uploaded file: {uploaded.name}"
elif st.session_state.get("snn_corpus"):
    corpus = st.session_state["snn_corpus"]
    corpus_source = "fetched dataset"
else:
    corpus = DEFAULT_CORPUS
    corpus_source = "built-in toy corpus"
st.write(f"Corpus size: **{len(corpus.encode('utf-8')):,} bytes** · source: {corpus_source}")


st.subheader("🌱 Awakening Environment")
st.caption("A controlled toy world for testing continuous SNN activity, exploration, online adaptation, and resumable developmental history.")
st.warning("Research simulation only, not evidence of consciousness. This first environment tests a small XOR rule-discovery task; English comprehension and autonomous internet learning are not implemented here.")

if "awakening_agent" not in st.session_state:
    st.session_state["awakening_agent"] = AwakeningAgent(seed=42)

awakening_agent = st.session_state["awakening_agent"]
aw_cols = st.columns([1, 1, 1, 2])
with aw_cols[0]:
    aw_steps = st.select_slider("Experience steps", options=[1, 5, 10, 25, 50], value=10)
with aw_cols[1]:
    if st.button("Run experiences", type="primary", key="aw_run"):
        try:
            awakening_agent.run(int(aw_steps))
            st.session_state["awakening_agent"] = awakening_agent
            st.session_state["awakening_notice"] = f"Processed {aw_steps} new experiences."
        except Exception as exc:
            st.error(f"Awakening step failed: {exc}")
with aw_cols[2]:
    if st.button("Reset agent", key="aw_reset"):
        st.session_state["awakening_agent"] = AwakeningAgent(seed=42)
        st.session_state.pop("awakening_notice", None)
        st.rerun()
with aw_cols[3]:
    state_upload = st.file_uploader("Resume from exported state JSON", type=["json"], key="aw_state_upload")
    if state_upload is not None and st.button("Load saved state", key="aw_load"):
        try:
            loaded_state = json.loads(state_upload.getvalue().decode("utf-8"))
            st.session_state["awakening_agent"] = AwakeningAgent(seed=int(loaded_state.get("seed", 42)), state=loaded_state)
            st.session_state["awakening_notice"] = "Loaded saved developmental state."
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load state: {exc}")

if st.session_state.get("awakening_notice"):
    st.info(st.session_state["awakening_notice"])

awakening_agent = st.session_state["awakening_agent"]
aw_summary = awakening_agent.summary()
aw_metrics = st.columns(5)
aw_metrics[0].metric("Experience steps", aw_summary["step_count"])
aw_metrics[1].metric("Recent accuracy (last 100)", "—" if aw_summary["recent_prediction_accuracy"] is None else f"{aw_summary['recent_prediction_accuracy']:.1%}")
aw_metrics[2].metric("Mean spikes / step", f"{aw_summary['mean_spike_count_recent']:.2f}")
aw_metrics[3].metric("Non-finite resets", aw_summary["non_finite_resets"])
aw_metrics[4].metric("Total reward", f"{aw_summary['total_reward']:.2f}")
st.caption(aw_summary["note"])
if awakening_agent.history:
    aw_history_df = pd.DataFrame(awakening_agent.history)
    st.line_chart(aw_history_df.set_index("step")[["reward", "surprise", "spike_count"]])
    st.dataframe(aw_history_df.tail(20), use_container_width=True, hide_index=True)
else:
    st.info("The agent has not experienced the environment yet. Run a few steps to begin.")
st.download_button(
    "Export developmental state (JSON)",
    data=export_state_json(awakening_agent),
    file_name="evolveai_awakening_state.json",
    mime="application/json",
    key="aw_export",
)
st.caption("State remains in the current Streamlit session. Download the JSON and load it again after a session reset; cloud session memory is not guaranteed to persist.")

st.subheader("🛡️ Governed Autonomous Learning")
st.caption("Run a bounded learning cycle using only the fetched, allowlisted dataset. The governor verifies provenance and corpus hash, caps compute, and records experiment history.")
if "governor_run_id" not in st.session_state:
    st.session_state["governor_run_id"] = uuid.uuid4().hex
governor_dir = Path("/tmp")
governor_state_path = governor_dir / ("evolveai_governor_" + st.session_state["governor_run_id"] + ".json")
governor_checkpoint_path = governor_dir / ("evolveai_best_" + st.session_state["governor_run_id"] + ".pt")
governor_manifest = st.session_state.get("snn_corpus_manifest")
governor_has_approved_data = bool(governor_manifest and corpus_source == "fetched dataset")
state_upload = st.file_uploader("Resume governor history (JSON)", type=["json"], key="governor_state_upload")
if state_upload is not None and st.button("Load governor history", key="governor_load_state"):
    try:
        uploaded_state = json.loads(state_upload.getvalue().decode("utf-8"))
        if uploaded_state.get("state_version") != 1 or not isinstance(uploaded_state.get("cycles"), list):
            raise ValueError("Unsupported governor-state format.")
        governor_state_path.write_text(json.dumps(uploaded_state, indent=2), encoding="utf-8")
        st.success("Governor history loaded for this session.")
    except Exception as exc:
        st.error("Could not load governor history: " + str(exc))
gov_cols = st.columns(3)
with gov_cols[0]:
    gov_trials = st.slider("Governed trials", 1, 6, 4, key="gov_trials")
with gov_cols[1]:
    gov_steps = st.slider("Steps per governed trial", 10, 100, 40, 10, key="gov_steps")
with gov_cols[2]:
    gov_seed = st.number_input("Governor seed", min_value=0, max_value=999999, value=42, key="gov_seed")
if not governor_has_approved_data:
    st.info("Fetch the approved TinyStories dataset above before running a governed cycle. Uploaded or built-in text is not accepted by this internet-data governor.")
if st.button("Run governed learning cycle", type="primary", disabled=not governor_has_approved_data, key="run_governed_cycle"):
    try:
        with st.spinner("Governor checking provenance and running bounded experiments…"):
            gov_report = run_governed_cycle(
                corpus=corpus, manifest=governor_manifest, state_path=governor_state_path,
                checkpoint_path=governor_checkpoint_path, max_trials=int(gov_trials),
                steps=int(gov_steps), block_size=min(int(block_size), 32),
                batch_size=min(int(batch_size), 8), seed=int(gov_seed), device="auto",
            )
        st.session_state["snn_governed_result"] = gov_report
        st.success("Governed cycle " + str(gov_report["governor"]["cycle_number"]) + " completed on " + str(gov_report.get("device_used")) + ".")
    except Exception as exc:
        st.error("Governed cycle failed or was blocked: " + str(exc))
gov_report = st.session_state.get("snn_governed_result")
if gov_report:
    if gov_report.get("best_trial"):
        st.metric("Governed best validation loss", f"{gov_report['best_trial']['validation_loss']:.3f}")
    if gov_report.get("validation_protocol_migrated"):
        st.warning("Validation protocol changed from the previous cycle. EvolveAI retained the prior checkpoint but is establishing a new comparable validation baseline.")
    plan = gov_report.get("learning_plan", {})
    if plan:
        st.markdown("#### EvolveAI's current learning agenda")
        st.markdown(f"**Focus:** {plan.get('focus', 'not recorded')}  \n**Objective:** {plan.get('objective', '')}")
        st.info("Hypothesis: " + plan.get("hypothesis", "No hypothesis recorded."))
        st.caption("Why this agenda: " + plan.get("rationale", ""))
        if plan.get("proposed_experiments"):
            st.dataframe(pd.DataFrame(plan["proposed_experiments"]), use_container_width=True, hide_index=True)
    st.json(gov_report.get("governor", {}))
    st.download_button("Download governed experiment report", data=json.dumps(gov_report, indent=2), file_name="evolveai_governed_report.json", mime="application/json", key="gov_report_download")
if governor_state_path.exists():
    state_bytes = governor_state_path.read_bytes()
    st.download_button("Download persistent governor history", data=state_bytes, file_name="evolveai_governor_state.json", mime="application/json", key="gov_state_download")
if governor_checkpoint_path.exists():
    st.download_button("Download best governed checkpoint", data=governor_checkpoint_path.read_bytes(), file_name="evolveai_best_snn_checkpoint.pt", mime="application/octet-stream", key="gov_checkpoint_download")

st.subheader("Autonomous Experiment Lab")
st.caption("Run a bounded trial-and-error search. It varies reviewed model settings, records failures, and ranks runs by validation loss.")
auto_cols = st.columns(3)
with auto_cols[0]:
    auto_trials = st.slider("Experiment trials", min_value=1, max_value=6, value=3)
with auto_cols[1]:
    auto_steps = st.slider("Steps per trial", min_value=10, max_value=150, value=30, step=10)
with auto_cols[2]:
    auto_seed = st.number_input("Experiment seed", min_value=0, max_value=999999, value=42, key="auto_seed")
if st.button("Start autonomous trial-and-error", type="primary"):
    auto_bar = st.progress(0)
    auto_status = st.empty()
    def on_trial_progress(index, total, record):
        auto_bar.progress(index / total)
        if record["status"] == "completed":
            auto_status.write(
                f"Trial {index}/{total}: {record['architecture']} · hidden={record['hidden_size']} · "
                f"lr={record['learning_rate']} · validation loss={record['validation_loss']:.3f}"
            )
        else:
            auto_status.write(f"Trial {index}/{total} failed: {record.get('error', 'unknown error')}")
    try:
        with st.spinner("Running bounded experiments. Keep this Streamlit session open until the run finishes."):
            auto_result = run_trials(
                corpus, max_trials=int(auto_trials), steps=int(auto_steps),
                block_size=min(int(block_size), 32), batch_size=int(batch_size),
                seed=int(auto_seed), progress=on_trial_progress,
            )
        st.session_state["snn_autonomous_result"] = auto_result
        st.success(f"Finished {auto_result['trials_completed']}/{auto_result['trials_requested']} trials in {auto_result['elapsed_seconds']:.1f}s.")
    except Exception as exc:
        st.error(f"Autonomous experiment failed: {exc}")

auto_result = st.session_state.get("snn_autonomous_result")
if auto_result:
    st.markdown("#### Experiment leaderboard")
    leaderboard = auto_result["leaderboard"]
    if leaderboard:
        st.dataframe(pd.DataFrame(leaderboard)[[
            "trial", "architecture", "hidden_size", "learning_rate",
            "validation_loss", "validation_perplexity", "elapsed_seconds"
        ]], use_container_width=True, hide_index=True)
        st.metric("Best validation loss", f"{auto_result['best_trial']['validation_loss']:.3f}")
        st.caption("Lowest validation byte-level cross-entropy wins this screening round; this is not a reasoning score.")
    else:
        st.warning("No trials completed successfully.")
    with st.expander("Full experiment log and limitations"):
        st.json(auto_result)
    st.download_button("Download experiment report JSON", data=export_json(auto_result),
                       file_name="evolveai_autonomous_experiments.json", mime="application/json")
if st.session_state.get("snn_corpus_manifest") and corpus_source == "fetched dataset":
    manifest = st.session_state["snn_corpus_manifest"]
    with st.expander("Dataset provenance and license metadata"):
        st.json(manifest)
        st.download_button("Download corpus manifest JSON", data=__import__("json").dumps(manifest, indent=2),
                           file_name="evolveai_corpus_manifest.json", mime="application/json")
        st.download_button("Download fetched corpus TXT", data=corpus,
                           file_name="evolveai_fetched_corpus.txt", mime="text/plain")
if len(corpus.encode("utf-8")) < 100:
    st.error("Provide at least 100 bytes of training text.")
else:
    if st.button("Train SNN from random initialization", type="primary"):
        bar = st.progress(0)
        status = st.empty()
        def on_progress(step, total, record):
            bar.progress(step / total)
            status.write(f"Step {step}/{total} · train loss {record['train_loss']:.3f} · validation loss {record['val_loss']:.3f}")
        with st.spinner("Training recurrent spiking network on the available device…"):
            result = train(corpus, steps=int(steps), block_size=int(block_size), batch_size=int(batch_size), hidden_size=int(hidden_size), seed=int(seed), progress=on_progress, architecture=architecture, device="auto")
        st.session_state["snn_result"] = result
        st.success(f"Finished in {result['elapsed_seconds']:.1f}s")

result = st.session_state.get("snn_result")
if result:
    history = pd.DataFrame(result["history"]).set_index("step")
    left, right = st.columns(2)
    left.metric("Final training loss", f"{history['train_loss'].iloc[-1]:.3f}")
    right.metric("Final held-out loss", f"{history['val_loss'].iloc[-1]:.3f}")
    st.line_chart(history[["train_loss", "val_loss"]])
    st.caption("Validation perplexity: {:.2f}. Compare experiments with the same split and settings.".format(history["val_perplexity"].iloc[-1]))
    prompt = st.text_input("Prompt the trained byte-level model", value="User: Hello.\nAssistant:")
    amount = st.slider("New bytes to sample", 20, 240, 100, 20)
    if st.button("Generate continuation"):
        continuation = result["model"].generate(prompt.encode("utf-8"), max_new_bytes=amount, device=result.get("device", "cpu"))
        st.text_area("Raw generated continuation (may be gibberish)", decode(continuation), height=180)
    st.download_button("Download training metrics CSV", history.reset_index().to_csv(index=False), file_name="evolveai_snn_metrics.csv", mime="text/csv")

with st.expander("What this experiment proves—and does not prove"):
    st.markdown("""
- **Tests:** whether a recurrent leaky-integrate-and-fire (LIF) network can learn byte-sequence statistics using surrogate-gradient backpropagation through time.
- **Does not prove:** human-like cognition, general reasoning, robust dialogue, or energy efficiency on ordinary CPUs.
- **Important:** a tiny corpus can make validation metrics misleading. Use a clean, sufficiently large corpus and keep a separate test set for serious comparisons.
- **EvolveAI Context SNN:** fast and slow membrane traces, gated slow context, and activity-adaptive firing thresholds. This is a testable design hypothesis, not a claim of biological equivalence.
- **Baseline:** simpler recurrent LIF cell for controlled comparison.
- Both models use surrogate-gradient backpropagation through time and byte-level prediction. Neither is a Transformer or uses pretrained weights.
""")
