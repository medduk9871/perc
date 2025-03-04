from typing import List
from src.dataset_types.datapoint import Datapoint
from src.example_selector.example_selector import ExampleSelector
from src.filler.base import Prompt
from rank_bm25 import BM25Okapi


class BM25ExampleSelector(ExampleSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset: List[Datapoint] = kwargs.get("dataset")
        tokenized_dataset = [x.prompt.split(" ") for x in dataset]
        self.bm25 = BM25Okapi(tokenized_dataset)

    def scores(self, query: Prompt):
        tokenized_query = query.target.prompt.split(" ")
        scores = self.bm25.get_scores(tokenized_query)

        return scores
