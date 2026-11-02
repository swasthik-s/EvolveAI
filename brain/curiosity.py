"""Progress- and coverage-aware domain curiosity; avoids chasing raw noise alone."""
from __future__ import annotations
from collections import defaultdict, deque
import math

class Curiosity:
    def __init__(self, window=20, exploration_floor=0.15):
        self.losses = defaultdict(lambda: deque(maxlen=window))
        self.counts = defaultdict(int)
        self.exploration_floor = float(exploration_floor)

    def observe(self, domain, loss):
        self.losses[domain].append(float(loss))
        self.counts[domain] += 1

    def scores(self, domains):
        domains = list(dict.fromkeys(domains))
        if not domains:
            return {}
        means = {d: (sum(self.losses[d]) / len(self.losses[d]) if self.losses[d] else 1.0) for d in domains}
        scale = max(means.values(), default=1.0) or 1.0
        total = sum(self.counts[d] for d in domains) + len(domains)
        raw = {}
        for d in domains:
            history = list(self.losses[d])
            uncertainty = min(1.0, means[d] / max(scale, 1e-8))
            coverage = math.sqrt(1.0 / (self.counts[d] + 1.0))
            progress = 0.0
            if len(history) >= 6:
                old = sum(history[:len(history)//2]) / (len(history)//2)
                new = sum(history[len(history)//2:]) / (len(history)-len(history)//2)
                progress = max(-1.0, min(1.0, (old-new) / max(abs(old), 1e-6)))
            # Hard-but-improving domains remain interesting; unlearnable noise is damped.
            raw[d] = max(0.01, 0.50*uncertainty + 0.30*coverage + 0.20*(0.5 + 0.5*progress))
        denom = sum(raw.values())
        probs = {d: max(self.exploration_floor/len(domains), (1-self.exploration_floor)*raw[d]/denom) for d in domains}
        norm = sum(probs.values())
        return {d: p/norm for d,p in probs.items()}
