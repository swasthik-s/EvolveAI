# EvolveAI SNN V2 — online plasticity research track

SNN V2 is an isolated research implementation. It does not replace the existing
SNN experiments or production chat engine.

## Design

- Recurrent adaptive leaky-integrate-and-fire neurons.
- Explicit membrane, adaptation, and refractory state.
- Event-by-event processing; no Transformer blocks, self-attention, pretrained
  weights, autograd, or backpropagation through time.
- Pair-based local STDP eligibility traces for input and recurrent synapses.
- A third-factor outcome modulator gates plasticity.
- A local readout delta rule learns next-byte prediction.
- Bounded replay to reduce immediate catastrophic forgetting.
- Atomic checkpoints with model configuration and source provenance.

This is a real event-driven spiking-network experiment, but not a biologically
complete brain model. It is not evidence of general reasoning, consciousness,
or human-level intelligence.

## Learning streams

Public data can be streamed from allowlisted Hugging Face sources: TinyStories,
WikiText, AG News, and GSM8K. Dataset cards and underlying content rights must
be reviewed before commercial use. Source IDs and content hashes are recorded;
the code does not establish legal permission.

Interaction learning is disabled unless the caller explicitly sets
user_opt_in=True. Do not train on private conversations by default. A
production deployment must expose a clear consent control, user-level deletion,
retention policy, and durable checkpoint storage before enabling this path.

## Quick start

Install the existing SNN requirements plus Hugging Face Datasets:

~~~bash
pip install -r requirements-snn.txt
pip install 'datasets>=3.0,<5.0'
~~~

Train continuously from several approved streams:

~~~bash
python -m snn_v2.train_stream --sources tinystories,wikitext,ag_news,gsm8k \
  --max-records 2000 --checkpoint runs/snn-v2/latest.pt
~~~

The runner is bounded by record count and record size. Run it on a dedicated
worker with persistent storage, not inside a short-lived serverless chat call.

## Interaction learning API

~~~python
from snn_v2.learner import ContinualSNNLearner

learner = ContinualSNNLearner(checkpoint_path="runs/snn-v2/latest.pt")
result = learner.learn_interaction(
    user_text="What is a spiking neuron?",
    assistant_text="A neuron that communicates using discrete spikes.",
    user_opt_in=True,
)
learner.save()
~~~

Without explicit opt-in, interaction data is rejected. For a multi-user service,
consent must be tracked per user and deletion must remove that user's records
from replay, provenance, and derived training artifacts. The single-process
research learner is not yet a multi-tenant production service.

## Evaluation requirements

Before claiming improvement, compare against V1 on identical train, validation,
and test splits and seeds. Report next-byte loss/perplexity, byte accuracy,
firing rates, weight norms, forgetting on earlier domains, and throughput. Use
an untouched test set. Lower next-byte loss alone does not demonstrate reasoning
or semantic understanding.

## Known limitations

- Byte-level next-symbol prediction is a narrow language objective.
- The local plasticity rule is a research baseline; stability/plasticity and
  long-range credit assignment require rigorous benchmarking.
- A compact readout can learn frequent local patterns but is not expected to
  match large language models.
- Public source availability and licenses can change.
- This implementation is not yet wired into the main chat API or dashboard.
  Integration requires explicit per-user consent and durable storage.
