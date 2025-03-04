import re
from typing import Optional

from src.prediction import Prediction
from src.problem import Problem

PRED_TEMPLATE = """\
import sys
import io

completion = \"\"\"\\
import math
import re
import numpy
import numpy as np
from typing import *

${prediction}
\"\"\"

def ENTRY_POINT(input_str):
    input_buffer = io.BufferedReader(io.BytesIO(input_str.encode()))
    stdin = io.TextIOWrapper(input_buffer)
    stdout = io.StringIO()
    stderr = io.StringIO()

    sys.stdin = stdin
    sys.stdout = stdout
    sys.stderr = stderr
    exec(
        completion,
        {
            "sys": sys,
        }
    )
    sys.stdin = sys.__stdin__
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    result = stdout.getvalue()
    error_message = stderr.getvalue()
    stdout.close()

    return result, error_message
"""


REF_TEMPLATE = """\
INPUT = ""
OUTPUT = ""
ERROR_MESSAGE = ""
STD_ERROR = ""

${reference}

try:
    result, error_message = ENTRY_POINT(INPUT)
except Exception as e:
    message = f"Error: {e}"
    if ERROR_MESSAGE != "":
        message += f", Message: {ERROR_MESSAGE}"
    raise Exception(message)
else:
    if STD_ERROR != "":
        message = f"Expected stderr: {STD_ERROR}"
        if ERROR_MESSAGE != "":
            message += f", Message: {ERROR_MESSAGE}"
        assert error_message != "", message
    else:
        message = f"Expected output: {OUTPUT}, Got: {result}"
        if ERROR_MESSAGE != "":
            message += f", Message: {ERROR_MESSAGE}"
        assert result.strip() == OUTPUT.strip(), message
"""


class CodeContestsProblem(Problem):
    @classmethod
    def from_dict(
        cls,
        datapoint: dict,
        prompt_trunc_len: Optional[int] = None,
        only_useful_note: bool = False,
        remove_note: bool = False,
    ):
        id = str(datapoint["source"]) + "/" + datapoint["name"]

        # load prompt
        prompt = datapoint["description"]

        # find note
        cutoff = prompt.rfind("\nNote\n")
        if cutoff == -1:
            note = ""
        else:
            note = prompt[cutoff:]

        # check if note is useful
        if only_useful_note:
            line_len = len([line for line in note.split("\n") if line.strip()])
            is_image_in_note = "<image>" in note
            if line_len <= 1 or is_image_in_note:
                raise ValueError("Note is not useful")

        # remove note
        if remove_note:
            if cutoff != -1:
                prompt = prompt[:cutoff]

        # truncate prompt
        if prompt_trunc_len is not None and len(prompt) > prompt_trunc_len:
            prompt = prompt[:prompt_trunc_len]

        # find the shortest python3 solution
        sol_indices = [
            i for i, lang in enumerate(datapoint["solutions"]["language"]) if lang == 3
        ]

        # value_for_cpp = 2
        # value_for_java = 4
        # sol_indices = []
        # try:
        #     idx_cpp = datapoint['solutions']['language'].index(value_for_cpp)
        #     sol_indices.append(idx_cpp)
        #     solution = [datapoint["solutions"]["solution"][i] for i in sol_indices]
        # except ValueError:
        #     pass
    
        # try:
        #     idx_java = datapoint['solutions']['language'].index(value_for_java)
        #     sol_indices.append(idx_java)
        # except ValueError:
        #     pass

        # solution = [datapoint["solutions"]["solution"][i] for i in sol_indices]
        
        
        try:
            solution = min(
                [datapoint["solutions"]["solution"][i] for i in sol_indices], key=len
            )
        except:
            solution = ""

        # load test cases
        test_cases = []
        test_types = ["public_tests", "private_tests"]
        for test_type in test_types:
            inputs = datapoint[test_type]["input"]
            outputs = datapoint[test_type]["output"]
            test_cases.extend(zip(inputs, outputs))
        test = [f"INPUT = {repr(i)}\nOUTPUT = {repr(o)}" for i, o in test_cases]

        return cls(id=id, prompt=prompt, solution=solution, test=test)


class CodeContestsPrediction(Prediction):
    def __init__(self, **data):
        super().__init__(**data)
