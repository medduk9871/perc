from typing import List
from src.dataset_types.datapoint import Datapoint
from src.example_selector.example_selector import ExampleSelector
from src.filler.base import Prompt
from sentence_transformers import SentenceTransformer, util


class MPNETExampleSelector(ExampleSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset: List[Datapoint] = kwargs.get("dataset")
        prompt_dataset = [prompt.prompt for prompt in dataset]
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.key_embeddings = self.model.encode(prompt_dataset)

    def scores(self, query: Prompt):
        query_embedding = self.model.encode(query.target.prompt)
        scores = util.cos_sim(self.key_embeddings, query_embedding)

        return scores
