# EvolveAI: controlled autonomous learning

This is a bounded experiment loop, not an unrestricted self-modifying agent. It may train on an explicitly selected corpus, evaluate on a separate validation file, keep the best checkpoint, and stop when validation fails to improve. It does not browse the internet, change its own code, or take external actions.

## 1. Prepare a size-limited corpus

Run this in a hosted notebook or another environment with Python. The Streamlit UI does not need the optional `datasets` package.

```bash
pip install -r snn/requirements-corpus.txt
python snn/prepare_corpus.py --max-mb 5 --out-dir data/tinystories
```

The script streams TinyStories, caps retained text, removes empty/duplicate stories, writes whole-story train/validation/test files, and records a manifest with provenance and file hashes.

Dataset: https://huggingface.co/datasets/roneneldan/TinyStories  
Declared license reference: CDLA-Sharing-1.0. Verify the current dataset card and terms before redistributing or using commercially. This pipeline records provenance; it is not a legal determination.

## 2. Run controlled training

```bash
python snn/controlled_train.py \
  --train data/tinystories/train.txt \
  --validation data/tinystories/validation.txt \
  --steps 500 --eval-every 25 --patience 5 \
  --out-dir runs/snn-tinystories
```

The runner:
- has an explicit maximum step budget;
- evaluates on a separate validation file at fixed sampled positions;
- stops after a configurable number of evaluations without improvement;
- saves the best checkpoint and a JSON metrics history;
- does not use the test set during training.

Keep `test.txt` untouched until model selection is complete. Compare models on that same test set, using multiple seeds where possible.

## 3. Streamlit Cloud workflow

The corpus builder and longer training job are intended to run in a hosted notebook/worker. Streamlit Community Cloud can host the UI, but its filesystem is ephemeral and its CPU/RAM quota may be insufficient for larger runs. Upload a small corpus to the UI for smoke tests; use a separate cloud notebook for corpus preparation and controlled training. Download checkpoints and metrics before the session ends.

## 4. What "autonomous learning" means here

The loop autonomously adjusts model weights through the optimizer and chooses when to stop based on validation results. It is deliberately not allowed to silently expand its data sources, change its code, or self-deploy. This gives us reproducible experiments and a safety boundary.

The current learner uses surrogate-gradient backpropagation through time, not biological local STDP. That should be a separate experimental branch so the two learning rules can be compared fairly.
