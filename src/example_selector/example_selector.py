from src.filler.base import Prompt

import logging

logger = logging.getLogger(__name__)


class ExampleSelector:
    @staticmethod
    def create(
        type: str,
        *args,
        **kwargs,
    ):
        if type == "CONSTANT":
            from src.example_selector.constant import ConstantExampleSelector

            return ConstantExampleSelector(*args, **kwargs)
        elif type == "BM25":
            from src.example_selector.bm25 import BM25ExampleSelector

            return BM25ExampleSelector(*args, **kwargs)
        elif type == "MPNET":
            from src.example_selector.mpnet import MPNETExampleSelector

            return MPNETExampleSelector(*args, **kwargs)
        elif type == "RANDOM":
            from src.example_selector.random import RandomExampleSelector

            return RandomExampleSelector(*args, **kwargs)
        else:
            raise ValueError(f"Unknown example selector type: {type}")

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, query: str, top_k: int = 1):
        scores = self.scores(query)
        logger.debug(f"Scores: {scores}")

        top_k_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        logger.debug(f"Top {top_k} indices: {top_k_indices}")
        return top_k_indices

    def scores(self, query: Prompt):
        raise NotImplementedError
