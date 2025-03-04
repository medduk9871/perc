from src.example_selector.example_selector import ExampleSelector
from src.filler.base import Prompt


class ConstantExampleSelector(ExampleSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ex_ids = kwargs.get("ex_ids", [])
        self.len_dataset = len(kwargs.get("dataset"))

    def scores(self, query: Prompt):
        # return high score in descending order of ex_ids
        scores = [0 for _ in range(self.len_dataset)]

        cur_score = len(self.ex_ids)
        for ex_id in self.ex_ids:
            scores[ex_id] = cur_score
            cur_score -= 1

        return scores
