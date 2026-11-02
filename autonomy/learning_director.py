"""Selects a learning domain from curiosity and coverage signals."""
import random

class LearningDirector:
    def __init__(self, curiosity, seed=42):
        self.curiosity = curiosity
        self.rng = random.Random(seed)
        self.decisions = 0

    def choose_domain(self, available_domains):
        weights = self.curiosity.scores(available_domains)
        if not weights:
            return None, {}
        domains = list(weights)
        selected = self.rng.choices(domains, weights=[weights[d] for d in domains], k=1)[0]
        self.decisions += 1
        return selected, weights
