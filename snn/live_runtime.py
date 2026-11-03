"""Live EvolveAI runtime: streamed learning, SNN generation, retrieval and belief ledger."""
from __future__ import annotations
import json
from pathlib import Path
import torch

from brain.memory import EpisodicMemory
from brain.semantic_memory import SemanticMemory
from brain.belief_revision import BeliefRevisionEngine
from brain.learning_director import LearningDirector
from streaming_gatekeeper import StreamingGatekeeper
from streaming_train import train_streaming
from train import build_model, resolve_device
from data import decode


class EvolveRuntime:
    def __init__(self, workdir, source_ids=None, device="auto", seed=43,
                 block_size=128, batch_size=16, hidden_size=128, embedding_dim=64,
                 learning_rate=0.0005, architecture="baseline"):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device)
        self.config = dict(block_size=block_size, batch_size=batch_size,
                           hidden_size=hidden_size, embedding_dim=embedding_dim,
                           learning_rate=learning_rate, architecture=architecture,
                           seed=seed)
        self.gatekeeper = StreamingGatekeeper(
            source_ids or ["tinystories", "wikitext", "ag_news", "gsm8k"], seed=seed
        )
        self.gatekeeper.warm_buffers()
        self.episodic = EpisodicMemory(self.workdir / "episodic_memory.jsonl")
        self.director = LearningDirector(seed=seed, memory=self.episodic)
        self.semantic = SemanticMemory(self.workdir / "semantic_memory")
        self.beliefs = BeliefRevisionEngine(self.workdir / "belief_ledger.json")
        self.checkpoint = self.workdir / "live_snn.pt"
        self.model = None
        self.cycles = 0
        if self.checkpoint.exists():
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            saved = payload.get("config", {})
            for key in ("architecture", "hidden_size", "embedding_dim"):
                if saved.get(key) != self.config[key]:
                    raise ValueError(f"Checkpoint {key} mismatch: {saved.get(key)} != {self.config[key]}")
            self.model = build_model(saved).to(self.device)
            self.model.load_state_dict(payload["state_dict"])
            self.cycles = int(payload.get("cycles", 0))
            print(f"[RUNTIME] restored checkpoint after {self.cycles} learning cycles")
        print(f"[RUNTIME] device={self.device} | sources={self.gatekeeper.source_order}")
        print(f"[RUNTIME] parameters={sum(p.numel() for p in self.model.parameters()):,}" if self.model else "[RUNTIME] model will initialize on first learning cycle")

    def learn(self, steps=100, eval_interval=25, patience=8):
        steps = int(steps)
        if not 1 <= steps <= 2000:
            raise ValueError("steps must be between 1 and 2000 per live learning cycle")
        initial = self.model.state_dict() if self.model is not None else None
        latest = {}
        def progress(record):
            latest.update(record)
            print(f"[LEARN] step={record['step']} train={record['train_loss']:.4f} "
                  f"val={record['validation_loss']:.4f} ppl={record['validation_perplexity']:.2f} "
                  f"domain={record.get('selected_domain')} tokens/s={record['tokens_per_second']:.1f}")
        result = train_streaming(
            self.gatekeeper, steps=steps, block_size=self.config["block_size"],
            batch_size=self.config["batch_size"], hidden_size=self.config["hidden_size"],
            embedding_dim=self.config["embedding_dim"], learning_rate=self.config["learning_rate"],
            seed=self.config["seed"] + self.cycles, architecture=self.config["architecture"],
            device=self.device, eval_interval=min(eval_interval, steps), patience=patience,
            progress=progress, memory=self.episodic, director=self.director,
            initial_state_dict=initial,
        )
        self.model = result["model"].eval()
        self.cycles += 1
        payload = {"state_dict": {k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                   "config": self.config, "cycles": self.cycles,
                   "best_validation_loss": result["best_validation_loss"]}
        tmp = self.checkpoint.with_suffix(".pt.tmp")
        torch.save(payload, tmp); tmp.replace(self.checkpoint)
        ingested = self.semantic.ingest_episodic_jsonl(self.episodic.path)
        print(f"[LEARN] cycle={self.cycles} best_val={result['best_validation_loss']:.4f} "
              f"memory_docs={ingested['documents_added']} checkpoint={self.checkpoint}")
        return result

    @torch.no_grad()
    def generate(self, prompt, max_new_bytes=240, temperature=0.8):
        if self.model is None:
            return "The SNN has not learned yet. Run /learn first."
        raw = self.model.generate(prompt.encode("utf-8"), max_new_bytes=max_new_bytes,
                                  temperature=temperature, device=self.device)
        return decode(raw)

    def think(self, question, top_k=5):
        hits = self.semantic.search(question, top_k=top_k)
        print(f"[THINK] query={question!r} | retrieved={len(hits)}")
        if not hits:
            print("[THINK] I do not have indexed evidence for this question.")
            return {"answer": "Insufficient evidence in current memory.", "evidence": []}
        q_terms = {x.lower() for x in question.split() if len(x) > 3}
        ranked = []
        for hit in hits:
            words = {x.lower() for x in hit["text"].split() if len(x) > 3}
            overlap = len(q_terms & words) / max(1, len(q_terms))
            ranked.append((overlap, hit))
        ranked.sort(key=lambda x: x[0], reverse=True)
        print("[THINK] Retrieved evidence (lexical relevance; not verified truth):")
        evidence = []
        for score, hit in ranked[:top_k]:
            item = {"score": round(score, 3), "domain": hit["domain"],
                    "source_ids": hit["source_ids"], "document_id": hit["document_id"],
                    "text": hit["text"][:700]}
            evidence.append(item)
            print(f"  [{item['domain']} | relevance={item['score']}] {item['text']}")
        answer = ("Memory contains potentially relevant evidence. It does not establish truth; "
                  "compare the original sources and check for missing or contradictory context.")
        print("[THINK] " + answer)
        return {"answer": answer, "evidence": evidence}

    def observe_belief(self, subject, relation, object_, evidence, source_id,
                       polarity=1, reliability=0.5, domain="user_review"):
        result = self.beliefs.observe(subject, relation, object_, evidence, source_id,
                                      polarity=polarity, reliability=reliability, domain=domain)
        print(json.dumps({k: result[k] for k in ("subject","relation","object","status",
                                                   "confidence","supporting_sources",
                                                   "contradicting_sources","revisions")}, indent=2))
        return result

    def status(self):
        return {"cycles": self.cycles, "device": self.device,
                "model_loaded": self.model is not None,
                "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                "checkpoint": str(self.checkpoint),
                "stream": self.gatekeeper.status(),
                "episodic": self.episodic.domain_summary(),
                "semantic": self.semantic.stats(),
                "beliefs": self.beliefs.report()}

    def close(self):
        self.semantic.save()
        self.beliefs.save()
        print("[RUNTIME] memory and belief ledger saved.")
