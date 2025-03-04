from typing import List
from src.similarity.similarity import Similarity


class ConstantSimilarity(Similarity):
    def __init__(self, *args, **kwargs):
        pass

    def batch_run(
        self,
        targets: List[List[str]],
        references: List[str],
    ):
        return [1.0] * len(targets)

    def run(
        self,
        target: List[str],
        reference: str,
    ):
        return 1.0
