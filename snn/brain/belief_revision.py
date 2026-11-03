"""Auditable belief revision for EvolveAI.

Tracks claim observations, provenance, independent-source support, contradiction,
and revision history. It does not discover truth; outcomes remain provisional.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


def _weight(reliability: float) -> float:
    value = float(reliability)
    if not 0.0 <= value <= 1.0:
        raise ValueError("reliability must be between 0 and 1")
    return value


class BeliefRevisionEngine:
    """Maintain claim-level evidence without silently overwriting prior beliefs."""

    def __init__(self, path=None, support_threshold=0.65, contradiction_threshold=0.65,
                 conflict_margin=0.20, minimum_sources=2):
        self.path = Path(path) if path else None
        self.support_threshold = float(support_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.conflict_margin = float(conflict_margin)
        self.minimum_sources = int(minimum_sources)
        self.claims = {}
        if self.path and self.path.exists():
            self._load()

    @staticmethod
    def claim_id(subject: str, relation: str, object_: str) -> str:
        key = " | ".join(_normalise(x) for x in (subject, relation, object_))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def observe(self, subject, relation, object_, evidence, source_id, polarity=1,
                domain="unknown", reliability=0.5, observed_at=None):
        if polarity not in (-1, 1):
            raise ValueError("polarity must be +1 (support) or -1 (contradiction)")
        if not source_id or not str(source_id).strip():
            raise ValueError("source_id is required for provenance")
        if not evidence or not str(evidence).strip():
            raise ValueError("evidence text is required")
        reliability = _weight(reliability)
        cid = self.claim_id(subject, relation, object_)
        claim = self.claims.setdefault(cid, {
            "claim_id": cid, "subject": str(subject).strip(),
            "relation": str(relation).strip(), "object": str(object_).strip(),
            "observations": [], "revisions": [], "status": "insufficient_evidence",
            "confidence": 0.0, "supporting_sources": [], "contradicting_sources": [],
        })
        source_id = str(source_id).strip()
        observation_id = hashlib.sha256(
            f"{cid}|{source_id}|{polarity}|{_normalise(evidence)}".encode("utf-8")
        ).hexdigest()
        if not any(x["observation_id"] == observation_id for x in claim["observations"]):
            claim["observations"].append({
                "observation_id": observation_id, "source_id": source_id,
                "polarity": polarity, "domain": str(domain),
                "reliability": reliability, "evidence": str(evidence).strip(),
                "observed_at": observed_at or _now(),
            })
        self._reassess(claim)
        self.save()
        return self.get(cid)

    def _reassess(self, claim):
        # One source contributes at most its strongest observation per polarity.
        source_weights = {1: defaultdict(float), -1: defaultdict(float)}
        for item in claim["observations"]:
            p, sid = item["polarity"], item["source_id"]
            source_weights[p][sid] = max(source_weights[p][sid], item["reliability"])
        support = sum(source_weights[1].values())
        oppose = sum(source_weights[-1].values())
        total = support + oppose
        confidence = support / total if total else 0.0
        old = claim["status"]
        enough_support = len(source_weights[1]) >= self.minimum_sources
        enough_oppose = len(source_weights[-1]) >= self.minimum_sources

        if (enough_support and enough_oppose and support >= self.contradiction_threshold
                and oppose >= self.contradiction_threshold
                and abs(support - oppose) / max(total, 1e-9) <= self.conflict_margin):
            status = "conflicted"
        elif enough_support and confidence >= self.support_threshold:
            status = "provisionally_supported"
        elif enough_oppose and (1.0 - confidence) >= self.contradiction_threshold:
            status = "provisionally_contradicted"
        else:
            status = "insufficient_evidence"

        claim["confidence"] = round(confidence, 6)
        claim["support_weight"] = round(support, 6)
        claim["contradiction_weight"] = round(oppose, 6)
        claim["supporting_sources"] = sorted(source_weights[1])
        claim["contradicting_sources"] = sorted(source_weights[-1])
        if status != old:
            claim["revisions"].append({
                "from": old, "to": status, "at": _now(),
                "reason": "Evidence ledger reassessed using independent source weights.",
                "support_weight": round(support, 6),
                "contradiction_weight": round(oppose, 6),
            })
        claim["status"] = status

    def get(self, claim_id):
        claim = self.claims.get(claim_id)
        return json.loads(json.dumps(claim)) if claim else None

    def search(self, text):
        needle = _normalise(text)
        return [self.get(cid) for cid, claim in self.claims.items()
                if needle in _normalise(" ".join(
                    (claim["subject"], claim["relation"], claim["object"])
                ))]

    def report(self):
        counts = defaultdict(int)
        for claim in self.claims.values():
            counts[claim["status"]] += 1
        return {"claims": len(self.claims), "statuses": dict(counts),
                "observations": sum(len(c["observations"]) for c in self.claims.values())}

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "claims": self.claims},
                      stream, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def _load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported belief ledger schema")
        self.claims = data.get("claims", {})
