from typing import List
from pydantic import BaseModel


class Prediction(BaseModel):
    completions: list
    candidates: list = None


class PredictionDataset(BaseModel):
    data: List[Prediction] = []
    Datapoint: Prediction = None

    @classmethod
    def from_dicts(cls, type, dicts):
        # set datapoint type
        if isinstance(type, str):
            if type == "openai_humaneval":
                from src.dataset_types.human_eval import HumanEvalPrediction

                Datapoint = HumanEvalPrediction
            elif type == "deepmind/code_contests":
                from src.dataset_types.code_contests import CodeContestsPrediction

                Datapoint = CodeContestsPrediction
            elif type == "nuprl/MultiPL-E,humaneval-cpp":
                from src.dataset_types.human_eval_cpp import HumanEvalCppPrediction

                Datapoint = HumanEvalCppPrediction
            elif type == "nuprl/MultiPL-E,humaneval-java":
                from src.dataset_types.human_eval_java import HumanEvalJavaPrediction

                Datapoint = HumanEvalJavaPrediction
            elif type == "nuprl/MultiPL-E,humaneval-py":
                from src.dataset_types.human_eval_python import HumanEvalPythonPrediction

                Datapoint = HumanEvalPythonPrediction
            elif type == "nuprl/MultiPL-E,humaneval-lua":
                from src.dataset_types.human_eval_python import HumanEvalLuaPrediction

                Datapoint = HumanEvalLuaPrediction
            elif type == "nuprl/MultiPL-E,humaneval-rb":
                from src.dataset_types.human_eval_rb import HumanEvalRubyPrediction

                Datapoint = HumanEvalRubyPrediction
            elif type == "nuprl/MultiPL-E,humaneval-r":
                from src.dataset_types.human_eval_r import HumanEvalRPrediction

                Datapoint = HumanEvalRPrediction
            elif type == "nuprl/MultiPL-E,mbpp-cpp":
                from src.dataset_types.mbpp_java import MBPPCppPrediction

                Datapoint = MBPPCppPrediction
            elif type == "nuprl/MultiPL-E,mbpp-java":
                from src.dataset_types.mbpp_java import MBPPJavaPrediction

                Datapoint = MBPPJavaPrediction
            elif type.startswith("mbpp-pseudo"):
                from src.dataset_types.mbpp_pseudo import MBPPPseudoPrediction

                Datapoint = MBPPPseudoPrediction
            else:
                raise NotImplementedError
        elif isinstance(type, (tuple, list)):
            if type[0] == "mbpp":
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return cls(data=[Datapoint(completions=d["code"]) for d in dicts])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)
