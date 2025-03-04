import random

from src.similarity.similarity import Similarity


class RandomSimilarity(Similarity):
    def __init__(self, *args, **kwargs):
        pass

    def run(self, target, reference):
        scores = random.sample(range(len(target)), target)

        return scores
