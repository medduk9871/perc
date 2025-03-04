from typing import Optional

from src.prediction import Prediction
from src.problem import Problem

PRED_TEMPLATE = """\
${prediction}
"""


REF_TEMPLATE = """\
${reference}
"""


class MBPPPseudoProblem(Problem):
    @classmethod
    def from_dict(cls, datapoint: dict, prompt_trunc_len: Optional[int] = None):
        id = datapoint["id"]
        code = datapoint["code"][0]
        pseudo = datapoint["draft_plan"][0]

        prompt = datapoint["prompt"]

        return cls(
            id=id,
            prompt=prompt,
            code=code,
            pseudo=pseudo,
            solution=datapoint["solution"],
            test=datapoint["test"]
        )

class MBPPPseudoPrediction(Prediction):
    def __init__(self, **data):
        super().__init__(**data)

        self.candidates = self.completions