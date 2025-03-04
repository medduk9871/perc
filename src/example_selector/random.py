from src.example_selector.example_selector import ExampleSelector
from src.filler.base import Prompt
import random


class RandomExampleSelector(ExampleSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len_dataset = len(kwargs.get("dataset"))

    def scores(self, query: Prompt):
        scores = random.sample(range(self.len_dataset), self.len_dataset)

        return scores
