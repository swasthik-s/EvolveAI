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

    def choose_domain(self, available_domains):
        domain, weights = self.curiosity.choose(available_domains, self.rng)
        self.decisions += 1
        if domain:
            print(f"[DIRECTOR] decision={self.decisions} selected={domain} weights={weights}")
        return domain, weights
