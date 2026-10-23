# EvolveAI SNN Lab — non-Transformer language experiment

This isolated prototype tests whether a small recurrent spiking neural network can learn byte-level sequence statistics. It does **not** use Transformer blocks, self-attention, pretrained weights, or a hosted model API.

## Architecture

- UTF-8 bytes are tokenized as integers 0–255.
- A learned embedding feeds a recurrent leaky-integrate-and-fire (LIF) layer.
- Membrane state integrates input and recurrent current; threshold crossings emit spikes.
- A surrogate derivative makes the hard spike function trainable with backpropagation through time.
- A linear readout predicts the next byte.

This is a biologically inspired engineering model, not a faithful simulation of the human brain. Surrogate-gradient learning is not the same as biological synaptic learning such as STDP.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-snn.txt
streamlit run snn/app.py
```

## Run tests

```bash
pytest -q tests/test_snn.py
```

## Experiment protocol

1. Start with the built-in toy corpus to confirm the pipeline runs.
2. Upload a UTF-8 `.txt` corpus for a more meaningful experiment. Preserve dataset licenses and provenance.
3. Keep a separate held-out split; compare train and validation loss, not just generated samples.
4. Repeat with multiple random seeds and compare against a similarly sized non-spiking recurrent baseline before drawing conclusions.
5. Use a separate untouched test set for final comparisons. The current demo uses a chronological train/validation split and is an engineering smoke test, not a publication-grade benchmark.

## Limitations

- Byte-level generation can be incoherent, especially with a small corpus and short training.
- The recurrent loop is sequential and can be slow on CPU.
- Lower loss demonstrates statistical learning, not general reasoning or human-level intelligence.
- Streamlit Community Cloud has limited CPU/RAM; use modest settings.
- The demo currently uses surrogate-gradient BPTT. A separate experiment is needed to test local STDP or reward-modulated plasticity.
