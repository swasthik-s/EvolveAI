from pathlib import Path
import io
import json
import streamlit as st
import torch
from model import TinyGPT, ModelConfig
from data import encode, decode, VOCAB_SIZE
from training import train_model, count_parameters
from evaluation import complete, evaluate_loss

st.set_page_config(page_title="EvolveAI Mini", page_icon="🧠", layout="wide")
st.title("🧠 EvolveAI Mini")
st.caption("A tiny decoder-only Transformer trained from random initialization. No pretrained weights or hosted model API.")
st.warning("This is a learning experiment, not a capable general-purpose assistant. A small model trained on a small corpus will imitate patterns and may produce nonsense. Streamlit Community Cloud has limited CPU/RAM, and files in its runtime are not durable.")

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.cfg = None
    st.session_state.device = "cpu"
    st.session_state.history = []
    st.session_state.train_metrics = []
    st.session_state.corpus = ""

with st.sidebar:
    st.header("Model configuration")
    n_embd = st.select_slider("Embedding width", options=[64, 96, 128, 160], value=96)
    n_head = st.select_slider("Attention heads", options=[2, 3, 4, 5, 6, 8], value=4)
    if n_embd % n_head:
        st.error("Embedding width must divide evenly by attention heads.")
    n_layer = st.slider("Transformer blocks", 1, 4, 2)
    block_size = st.select_slider("Context length (bytes)", options=[64, 96, 128, 192, 256], value=128)
    batch_size = st.select_slider("Batch size", options=[4, 8, 16, 24], value=8)
    steps = st.slider("Training steps", 20, 1000, 200, step=20)
    learning_rate = st.select_slider("Learning rate", options=[0.0001, 0.0003, 0.0005, 0.001], value=0.0003)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)

chat_tab, train_tab, eval_tab, about_tab = st.tabs(["💬 Chat", "🏋️ Train", "📊 Evaluate", "ℹ️ About"])

with train_tab:
    st.subheader("Train from scratch")
    st.write("Upload a UTF-8 .txt corpus or paste text. Blank lines separate documents. Do not upload private data.")
    uploaded = st.file_uploader("Training corpus (.txt)", type=["txt"])
    default_corpus = """User: Hey.\nAssistant: Hey again! What's on your mind?\n\nUser: How are you?\nAssistant: I'm ready to help. What's happening?\n\nUser: I said hey again.\nAssistant: I noticed. Is something on your mind, or are we testing my conversational skills?\n\nUser: That is not what I meant.\nAssistant: Got it. What did you mean? I'll adjust.\n\nUser: Tell me a joke.\nAssistant: I tried to write a joke about RAM, but I kept forgetting the punchline.\n\nUser: I feel frustrated.\nAssistant: That sounds frustrating. Want to tell me what happened?\n\nUser: What do you know?\nAssistant: I can only answer from what I've learned and the context available to me. I should be honest when I don't know.\n\nUser: Can you remember this?\nAssistant: I can use the context provided in this conversation, but durable memory needs to be implemented separately.\n\nUser: You repeated yourself.\nAssistant: Fair point. Let me try answering the actual question instead.\n\nUser: What should you do if you are unsure?\nAssistant: If the uncertainty matters, I should ask a clarifying question instead of pretending I know."""
    if uploaded:
        try: corpus = uploaded.getvalue().decode("utf-8")
        except UnicodeDecodeError: corpus = ""
        if not corpus: st.error("File must be valid UTF-8.")
    else:
        corpus = st.text_area("Or paste a training corpus", value=default_corpus, height=260)
    st.caption(f"Corpus size: {len(corpus.encode('utf-8')):,} bytes. A small sample demonstrates mechanics only; it is far too small to teach general language.")
    if st.button("🚀 Train new model", type="primary", disabled=(n_embd % n_head != 0 or len(corpus.encode('utf-8')) < 100), use_container_width=True):
        st.session_state.corpus = corpus
        progress = st.progress(0)
        status = st.empty()
        def on_progress(step, total, record):
            progress.progress(step / total)
            status.write(f"Step {step}/{total} · train loss {record['train_loss']:.3f} · validation loss {record['val_loss']:.3f}")
        try:
            model, cfg, metrics, token_count, device = train_model(
                corpus, steps=steps, batch_size=batch_size, block_size=block_size,
                learning_rate=learning_rate, n_embd=n_embd, n_head=n_head,
                n_layer=n_layer, seed=int(seed), progress=on_progress)
            st.session_state.model, st.session_state.cfg = model.cpu(), cfg
            st.session_state.device = "cpu"
            st.session_state.train_metrics = metrics
            st.success(f"Training complete. {count_parameters(model):,} parameters · {token_count:,} tokens · trained from random initialization.")
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {type(exc).__name__}: {exc}")

with chat_tab:
    st.subheader("Talk to your trained model")
    if st.session_state.model is None:
        st.info("Train a model first. It will start with random weights and learn only from your corpus.")
    else:
        for item in st.session_state.history:
            with st.chat_message(item["role"]): st.markdown(item["content"])
        prompt = st.chat_input("Message EvolveAI Mini…")
        if prompt:
            st.session_state.history.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            try:
                output = complete(st.session_state.model, prompt, max_new_tokens=160, temperature=0.8, device="cpu")
                # This baseline is a next-token completer, not instruction-tuned; prompt output may be incoherent.
                with st.chat_message("assistant"): st.markdown(output)
                st.session_state.history.append({"role":"assistant","content":output})
            except Exception as exc:
                st.error(f"Generation failed: {type(exc).__name__}: {exc}")

with eval_tab:
    st.subheader("Held-out evaluation")
    if st.session_state.train_metrics:
        st.line_chart({ "train loss": [x["train_loss"] for x in st.session_state.train_metrics],
                        "validation loss": [x["val_loss"] for x in st.session_state.train_metrics]})
        st.write("Training checkpoints", st.session_state.train_metrics)
    if st.session_state.model is not None and st.session_state.corpus:
        if st.button("Evaluate held-out loss"):
            try:
                result = evaluate_loss(st.session_state.model, st.session_state.corpus, device="cpu")
                a,b,c = st.columns(3)
                a.metric("Validation loss", f"{result['loss']:.3f}")
                b.metric("Perplexity", f"{result['perplexity']:.2f}")
                c.metric("Validation tokens", f"{result['tokens']:,}")
                st.caption("Perplexity measures next-token prediction on this split. It does not measure helpfulness, truth, reasoning, or conversational quality.")
            except Exception as exc: st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")
    st.markdown("#### Manual behavior checks")
    st.write("Try held-out prompts: repeat a greeting, correct the model, ask a follow-up, change topic, and ask an out-of-domain question. Record results across checkpoints; don't judge progress from training loss alone.")

with about_tab:
    st.subheader("What this project is")
    st.markdown("""
- **Architecture:** decoder-only Transformer (causal self-attention, MLP blocks, positional embeddings).
- **Tokenizer:** UTF-8 byte IDs plus three reserved tokens; no pretrained tokenizer.
- **Training objective:** predict the next byte/token with cross-entropy.
- **Initialization:** random weights every time you click Train.
- **Memory/tools:** not implemented in this model; these belong in a separate EvolveAI controller.
- **Persistence:** Streamlit session state is temporary. Refresh/restart may lose the model; this demo does not promise durable checkpoints.
- **Honesty:** this tiny model is expected to produce weak, repetitive, or incoherent text until trained on a much larger, clean corpus.
""")
    st.code("streamlit run mini/app.py")
