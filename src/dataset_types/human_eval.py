import re
import logging
from typing import Optional

from src.prediction import Prediction
from src.problem import Problem

logger = logging.getLogger(__name__)


PRED_TEMPLATE = """\
import sys
import math
import re
import numpy
import numpy as np
from typing import *

${prediction}
"""


REF_TEMPLATE = """\
${reference}
"""


class HumanEvalProblem(Problem):
    @classmethod
    def from_dict(cls, datapoint: dict, prompt_trunc_len: Optional[int] = None):
        id = datapoint["task_id"]
        entry_point = datapoint["entry_point"]
        test = [datapoint["test"] + f"check({entry_point})"]
        prompt = datapoint["prompt"]
        solution = prompt + datapoint["canonical_solution"]

        # truncate the prompt
        if prompt_trunc_len is not None and len(prompt) > prompt_trunc_len:
            prompt = prompt[:prompt_trunc_len]

        return cls(
            id=id, prompt=prompt, solution=solution, test=test, entry_point=entry_point
        )


class HumanEvalPrediction(Prediction):
    def __init__(self, **data):
        super().__init__(**data)

        self.candidates = self.completions
