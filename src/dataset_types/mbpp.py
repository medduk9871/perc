from typing import Optional

from src.problem import Problem


class MBPPProblem(Problem):
    datapoint: dict
    prompt_trunc_len: Optional[int] = None

    @classmethod
    def from_dict(cls, datapoint: dict, prompt_trunc_len: Optional[int] = None):
        id = datapoint["task_id"]

        def _split_head_code(code):
            cut = code.rfind("def ")
            body, head = code[:cut], code[cut:]

            return head, body

        head, code = _split_head_code(datapoint["code"])

        prompt = datapoint["prompt"]
        if prompt_trunc_len is not None and len(prompt) > prompt_trunc_len:
            prompt = prompt[:prompt_trunc_len]

        return cls(
            id=id,
            head=head,
            prompt=prompt,
            code=code,
        )
