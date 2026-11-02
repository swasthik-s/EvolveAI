"""Learning director: selects which available domain receives the next batch."""
from __future__ import annotations
import random
from .curiosity import Curiosity


class LearningDirector:
    def __init__(self, seed=42, memory=None):
        self.rng = random.Random(seed)
        self.curiosity = Curiosity()
        self.memory = memory
        self.decisions = 0
        # Rehydrate recent learning signals so a later run can continue its
        # domain priorities rather than starting with a blank curiosity state.
        if memory:
            for event in memory.events[-500:]:
                domain = event.get("domain")
                loss = event.get("loss")
                if domain and isinstance(loss, (int, float)):
                    self.curiosity.observe(domain, loss)

    def choose_domain(self, available_domains):
        domain, weights = self.curiosity.choose(available_domains, self.rng)
        self.decisions += 1
        if domain:
            print(f"[DIRECTOR] decision={self.decisions} selected={domain} weights={weights}")
        return domain, weights
