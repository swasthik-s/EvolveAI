# EvolveAI Mini — from-scratch language model

A small, educational decoder-only Transformer implemented with PyTorch and exposed through Streamlit. It initializes weights randomly; it does **not** download or use a pretrained model or hosted inference API.

## Run locally

Python 3.10 or 3.11 recommended.

```bash
cd mini
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app for this repository and branch.
3. Set **Main file path** to `mini/app.py`.
4. In Advanced settings, set the Python version compatible with the selected PyTorch release (Python 3.11 is a sensible starting point).
5. Set the dependency file to `mini/requirements.txt` if the deployment form supports selecting it. If it only installs the root `requirements.txt`, add the Streamlit and PyTorch dependencies there or use a dedicated repo/deployment branch whose root requirements include them.
6. Deploy. The first install may be slow because PyTorch is a large dependency.

## What the model learns

- Tokenization is UTF-8 bytes, not words or subwords. This keeps the vocabulary fixed at 259 IDs and avoids external tokenizer packages.
- The objective is next-token prediction with cross-entropy.
- Training data is split chronologically: the final 10% is used as a validation segment.
- Evaluation reports held-out loss and perplexity, plus training/validation curves.
- Chat is a raw text continuation demo. The model is **not instruction-tuned**, so it may continue the prompt, repeat text, emit malformed text, or fail to answer questions.

The starter corpus in the app is deliberately tiny and only demonstrates the mechanics. It is not enough to teach English. Use a legally obtained, high-quality corpus with appropriate rights and remove secrets and personal data. More data, model capacity, and compute are needed for useful language behavior. A tiny model trained on a tiny dataset cannot be expected to behave like a modern assistant.

## Limits of Streamlit Community Cloud

Community Cloud is designed for lightweight apps, not sustained model training. CPU/RAM limits and dependency installation time may prevent larger configurations from finishing. Start with width 64–96, 1–2 blocks, context 64–128, batch 4–8, and 20–200 steps. Train locally or on a GPU machine for longer runs, then add an explicit checkpoint export/import flow if you need durable weights. Streamlit session state and local filesystem should not be treated as persistent storage.

## Suggested evaluation protocol

Keep a fixed held-out prompt set that is not included in training. Score each model checkpoint on:
1. Validation loss/perplexity (next-token prediction only).
2. Continuation quality and grammaticality.
3. Repetition rate and diversity.
4. Context retention over multiple turns.
5. Response relevance, factuality, and uncertainty behavior, scored manually.

Do not interpret lower loss as proof of reasoning or helpfulness. Keep model comparisons reproducible by recording dataset version/hash, seed, configuration, training steps, and evaluation results.

## Project files

- `model.py`: Transformer architecture.
- `data.py`: byte tokenizer and training batches.
- `training.py`: seeded training and validation loop.
- `evaluation.py`: held-out metrics and generation.
- `app.py`: Streamlit chat/training/evaluation interface.

This is a research prototype, not a production assistant or a substitute for the broader EvolveAI memory, tool, and policy layers.
