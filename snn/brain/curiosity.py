"""Learning-progress-driven domain selection (bounded and interpretable)."""
from __future__ import annotations
from collections import defaultdict, deque
import math


class Curiosity:
    def __init__(self, window=12, exploration_floor=0.20):
        self.losses = defaultdict(lambda: deque(maxlen=window))
        self.exploration_floor = float(exploration_floor)

    def observe(self, domain, loss):
        if domain:
            self.losses[domain].append(float(loss))

    def scores(self, domains):
        domains = list(dict.fromkeys(domains))
        if not domains:
            return {}
        raw = {}
        for domain in domains:
            values = list(self.losses[domain])
            if len(values) < 2:
                progress = 0.0
                uncertainty = 1.0
            else:
                # Positive progress means loss is falling; negative means stalled/worsening.
                half = max(1, len(values) // 2)
                old = sum(values[:half]) / half
                new_values = values[-half:]
                new = sum(new_values) / len(new_values)
                progress = max(-1.0, min(1.0, (old - new) / max(old, 1e-6)))
                uncertainty = min(1.0, (max(values) - min(values)) / max(sum(values)/len(values), 1e-6))
            coverage = 1.0 / math.sqrt(1 + len(values))
            # Reward learning progress, but retain pressure to explore unfamiliar domains.
            raw[domain] = 0.45 * max(0.0, progress) + 0.25 * uncertainty + 0.30 * coverage
        total = sum(raw.values())
        if total <= 0:
            return {d: 1.0 / len(domains) for d in domains}
        return {d: v / total for d, v in raw.items()}

    def choose(self, domains, rng):
        weights = self.scores(domains)
        if not weights:
            return None, {}
        # Explicit exploration floor prevents a single domain monopolizing training.
        n = len(weights)
        mixed = {d: (1.0 - self.exploration_floor) * w + self.exploration_floor / n
                 for d, w in weights.items()}
        chosen = rng.choices(list(mixed), weights=list(mixed.values()), k=1)[0]
        return chosen, mixed
