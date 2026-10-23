"""Streamlit lab for testing a non-Transformer spiking language model."""
import streamlit as st
import pandas as pd
from train import train
from data import decode
from dataset_harvester import DATASETS, fetch_corpus

st.set_page_config(page_title="EvolveAI SNN Lab", page_icon="⚡", layout="wide")
st.title("⚡ EvolveAI — Spiking Neural Network Lab")
st.caption("Custom dual-timescale adaptive-threshold SNN · byte-level prediction · no Transformer or attention")
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
        with st.spinner("Training recurrent spiking network on CPU…"):
            result = train(corpus, steps=int(steps), block_size=int(block_size), batch_size=int(batch_size), hidden_size=int(hidden_size), seed=int(seed), progress=on_progress, architecture=architecture)
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
        continuation = result["model"].generate(prompt.encode("utf-8"), max_new_bytes=amount, device="cpu")
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
