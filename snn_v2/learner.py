"""Online three-factor plasticity and replay for EvolveAI SNN V2.

The recurrent core is updated with pair-based local STDP eligibility traces.
A scalar outcome modulator gates plasticity; the byte readout uses a local
delta rule. There is no BPTT or optimizer over a temporal computation graph.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from .model import AdaptiveLIFNetwork


@dataclass
class LearningStats:
    records_seen: int = 0
    records_learned: int = 0
    interaction_records: int = 0
    public_records: int = 0
    updates: int = 0
    predicted_bytes: int = 0
    correct_bytes: int = 0
    last_loss: float | None = None

    @property
    def accuracy(self):
        return self.correct_bytes / max(1, self.predicted_bytes)


class ContinualSNNLearner:
    """Persistent learner with bounded replay and explicit interaction consent."""

    def __init__(self, model=None, *, checkpoint_path="runs/snn-v2/latest.pt",
                 device="auto", replay_capacity=2048, replay_every=4,
                 max_record_bytes=4096, stdp_lr=0.0002, readout_lr=0.01,
                 trace_decay=0.92, seed=42):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if replay_capacity < 1 or replay_every < 1 or max_record_bytes < 2:
            raise ValueError("Invalid replay or record limits")
        self.device = torch.device(device)
        self.model = (model or AdaptiveLIFNetwork()).to(self.device)
        self.checkpoint_path = Path(checkpoint_path)
        self.replay = deque(maxlen=int(replay_capacity))
        self.replay_every = int(replay_every)
        self.max_record_bytes = int(max_record_bytes)
        self.stdp_lr = float(stdp_lr)
        self.readout_lr = float(readout_lr)
        self.trace_decay = float(trace_decay)
        self.rng = random.Random(seed)
        self.stats = LearningStats()
        self.seen_hashes = set()
        self.provenance = deque(maxlen=10000)

    @staticmethod
    def _digest(text):
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    @torch.no_grad()
    def _learn_sequence(self, text, reward_scale=1.0):
        payload = text.encode("utf-8", errors="replace")[:self.max_record_bytes]
        if len(payload) < 2:
            return {"loss": 0.0, "accuracy": 0.0, "bytes": 0}
        state = self.model.initial_state(1, self.device)
        vocab = self.model.vocab_size
        input_trace = torch.zeros(vocab, device=self.device)
        recurrent_trace = torch.zeros(self.model.hidden_size, device=self.device)
        post_trace = torch.zeros(self.model.hidden_size, device=self.device)
        losses, correct, count = 0.0, 0, 0

        for current, target in zip(payload[:-1], payload[1:]):
            previous_spikes = state.spikes[0].clone()
            logits, spikes, state = self.model.step(
                torch.tensor([current], dtype=torch.long, device=self.device), state
            )
            spike_vector = spikes[0]
            target_tensor = torch.tensor([target], dtype=torch.long, device=self.device)
            probabilities = torch.softmax(logits[0], dim=-1)
            prediction = int(probabilities.argmax().item())
            error = F.one_hot(target_tensor, num_classes=vocab).to(probabilities.dtype)[0] - probabilities
            loss = -torch.log(probabilities[target].clamp_min(1e-8))
            losses += float(loss.item())
            correct += int(prediction == target)
            count += 1

            # Three-factor modulation: correctness gates local STDP. A small
            # negative signal on errors discourages unhelpful co-activation.
            reward = (1.0 if prediction == target else -0.25) * float(reward_scale)
            pre = torch.zeros(vocab, device=self.device)
            pre[current] = 1.0
            old_input_trace = input_trace.clone()
            old_recurrent_trace = recurrent_trace.clone()
            old_post_trace = post_trace.clone()

            input_delta = torch.outer(pre, old_post_trace) - torch.outer(old_input_trace, spike_vector)
            recurrent_delta = (
                torch.outer(previous_spikes, old_post_trace)
                - torch.outer(old_recurrent_trace, spike_vector)
            )
            self.model.input_weights.add_(self.stdp_lr * reward * input_delta)
            self.model.recurrent_weights.add_(self.stdp_lr * reward * recurrent_delta)

            # Local readout delta rule: update only outgoing weights from
            # neurons that emitted a spike at this time step.
            self.model.readout_weights.add_(
                self.readout_lr * torch.outer(spike_vector, error) * float(reward_scale)
            )
            self.model.readout_bias.add_(self.readout_lr * error * float(reward_scale))

            input_trace.mul_(self.trace_decay).add_(pre)
            recurrent_trace.mul_(self.trace_decay).add_(previous_spikes)
            post_trace.mul_(self.trace_decay).add_(spike_vector)
            self.model.input_weights.clamp_(-2.0, 2.0)
            self.model.recurrent_weights.clamp_(-1.0, 1.0)
            self.model.readout_weights.clamp_(-2.0, 2.0)
            self.model.readout_bias.clamp_(-2.0, 2.0)

        mean_loss = losses / max(1, count)
        accuracy = correct / max(1, count)
        self.stats.predicted_bytes += count
        self.stats.correct_bytes += correct
        self.stats.last_loss = mean_loss
        self.stats.updates += count
        return {"loss": mean_loss, "accuracy": accuracy, "bytes": count}

    def learn_record(self, text, *, source_id, stream_type="public",
                     user_opt_in=False, quality_score=1.0):
        """Learn from one approved record; interaction data is opt-in only."""
        if stream_type not in {"public", "interaction"}:
            raise ValueError("stream_type must be 'public' or 'interaction'")
        if stream_type == "interaction" and not user_opt_in:
            return {"accepted": False, "reason": "interaction_consent_required"}
        if not 0.0 <= quality_score <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")
        clean = str(text or "").strip()
        if len(clean.encode("utf-8", errors="replace")) < 2:
            return {"accepted": False, "reason": "record_too_short"}
        digest = self._digest(clean)
        if digest in self.seen_hashes:
            return {"accepted": False, "reason": "duplicate"}
        self.seen_hashes.add(digest)
        self.stats.records_seen += 1
        if quality_score < 0.5:
            return {"accepted": False, "reason": "quality_gate_rejected"}

        self.replay.append({"text": clean[:self.max_record_bytes], "source_id": source_id})
        result = self._learn_sequence(clean, reward_scale=quality_score)
        self.stats.records_learned += 1
        if stream_type == "interaction":
            self.stats.interaction_records += 1
        else:
            self.stats.public_records += 1
        self.provenance.append({"source_id": source_id, "sha256": digest, "stream_type": stream_type})

        if self.stats.records_learned % self.replay_every == 0 and len(self.replay) > 1:
            replay_item = self.rng.choice(list(self.replay)[:-1])
            self._learn_sequence(replay_item["text"], reward_scale=0.25)
        return {"accepted": True, **result, "records_learned": self.stats.records_learned}

    def learn_interaction(self, user_text, assistant_text, *, user_opt_in):
        if not user_opt_in:
            return {"accepted": False, "reason": "interaction_consent_required"}
        text = "USER: " + str(user_text).strip() + "\nASSISTANT: " + str(assistant_text).strip()
        return self.learn_record(text, source_id="consented_interaction",
                                 stream_type="interaction", user_opt_in=True,
                                 quality_score=0.75)

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        payload = {
            "format_version": 2,
            "model_config": {
                "vocab_size": self.model.vocab_size,
                "hidden_size": self.model.hidden_size,
                "threshold": self.model.threshold,
                "membrane_decay": self.model.membrane_decay,
                "adaptation_decay": self.model.adaptation_decay,
                "adaptation_strength": self.model.adaptation_strength,
                "refractory_steps": self.model.refractory_steps,
            },
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "stats": asdict(self.stats),
            "provenance": list(self.provenance),
        }
        torch.save(payload, temp)
        temp.replace(self.checkpoint_path)
        return str(self.checkpoint_path)

    def load(self):
        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if payload.get("format_version") != 2:
            raise ValueError("Unsupported SNN V2 checkpoint format")
        config = payload["model_config"]
        if config["vocab_size"] != self.model.vocab_size or config["hidden_size"] != self.model.hidden_size:
            raise ValueError("Checkpoint architecture does not match this learner")
        self.model.load_state_dict(payload["model_state"])
        self.stats = LearningStats(**payload.get("stats", {}))
        self.provenance.extend(payload.get("provenance", []))
