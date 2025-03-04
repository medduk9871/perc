import re
import logging
from typing import Optional

from src.prediction import Prediction
from src.problem import Problem

logger = logging.getLogger(__name__)


PRED_TEMPLATE = """\
${prediction}
"""


REF_TEMPLATE = """\
${reference}
"""


class MBPPJavaProblem(Problem):
    @classmethod
    def from_dict(cls, datapoint: dict, prompt_trunc_len: Optional[int] = None):
        id = "/".join(datapoint["name"].split("_")[:2])
        test = [datapoint["tests"]]
        prompt = datapoint["prompt"]

        # truncate the prompt
        if prompt_trunc_len is not None and len(prompt) > prompt_trunc_len:
            prompt = prompt[:prompt_trunc_len]

        return cls(
            id=id, prompt=prompt, test=test
        )


class MBPPJavaPrediction(Prediction):
    def __init__(self, **data):
        super().__init__(**data)

        self.candidates = self.completions
