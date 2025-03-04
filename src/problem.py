from typing import Any, List, Optional, Union
import os
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from pydantic import BaseModel


class Problem(BaseModel):
    id: str = None
    prompt: str = None
    solution: str = None
    for_examples: bool = False

    test: List[str] = []
    misc: dict = {}

    def __init__(self, **data):
        super().__init__(**data)

        for key in list(data.keys()):
            if key not in self.__fields__:
                self.misc[key] = data.pop(key)

        if self.for_examples:
            del self.test

    def to_dict(self):
        d = {
            "id": self.id,
            "prompt": self.prompt,
            "solution": self.solution,
        }
        d.update(self.misc)
        if self.for_examples:
            d["code"] = self.solution
        else:
            d["test"] = self.test

        return d


class ProblemDataset(BaseModel):
    type: Any
    split: Optional[Union[List[str], str]]
    for_examples: bool = False
    datapoint_kwargs: dict = {}

    data: Optional[List[Problem]] = []
    indices: Optional[List[int]] = None
    Datapoint: Problem = None
    path: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # set datapoint type
        if isinstance(self.type, str):
            if self.type == "openai_humaneval":
                from src.dataset_types.human_eval import HumanEvalProblem

                self.Datapoint = HumanEvalProblem
            elif self.type == "deepmind/code_contests":
                from src.dataset_types.code_contests import CodeContestsProblem

                self.Datapoint = CodeContestsProblem
            elif self.type == "nuprl/MultiPL-E,humaneval-cpp":
                from src.dataset_types.human_eval_cpp import HumanEvalCppProblem

                self.Datapoint = HumanEvalCppProblem
            elif self.type == "nuprl/MultiPL-E,humaneval-java":
                from src.dataset_types.human_eval_java import HumanEvalJavaProblem

                self.Datapoint = HumanEvalJavaProblem
            elif self.type == "nuprl/MultiPL-E,humaneval-py":
                from src.dataset_types.human_eval_python import HumanEvalPythonProblem

                self.Datapoint = HumanEvalPythonProblem
            elif self.type == "nuprl/MultiPL-E,humaneval-lua":
                from src.dataset_types.human_eval_lua import HumanEvalLuaProblem

                self.Datapoint = HumanEvalLuaProblem
            elif self.type == "nuprl/MultiPL-E,humaneval-rb":
                from src.dataset_types.human_eval_rb import HumanEvalRubyProblem

                self.Datapoint = HumanEvalRubyProblem
            elif self.type == "nuprl/MultiPL-E,humaneval-r":
                from src.dataset_types.human_eval_r import HumanEvalRProblem

                self.Datapoint = HumanEvalRProblem
            elif self.type == "nuprl/MultiPL-E,mbpp-cpp":
                from src.dataset_types.mbpp_cpp import MBPPCppProblem

                self.Datapoint = MBPPCppProblem
            elif self.type == "nuprl/MultiPL-E,mbpp-java":
                from src.dataset_types.mbpp_java import MBPPJavaProblem

                self.Datapoint = MBPPJavaProblem
            elif self.type.startswith("mbpp-pseudo"):
                from src.dataset_types.mbpp_pseudo import MBPPPseudoProblem

                self.Datapoint = MBPPPseudoProblem
            else:
                raise NotImplementedError
        elif isinstance(self.type, (tuple, list)):
            if self.type[0] == "mbpp":
                from src.dataset_types.mbpp import MBPPProblem

                self.Datapoint = MBPPProblem
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # load dataset
        if self.path is not None:
            dataframe = pd.read_json(self.path, orient="records")
            self.data = []
            for _, row in dataframe.iterrows():
                self.data.append(self.Datapoint(**row, for_examples=self.for_examples))

        else:
            if isinstance(self.split, str):
                self.split = [self.split]
            dataset_list = []
            for split in self.split:
                if isinstance(self.type, str):
                    if self.type == "nuprl/MultiPL-E,humaneval-cpp" or self.type == "nuprl/MultiPL-E,humaneval-java" \
                        or self.type == "nuprl/MultiPL-E,humaneval-py" or self.type == "nuprl/MultiPL-E,humaneval-lua" \
                        or self.type == "nuprl/MultiPL-E,humaneval-rb" or self.type == "nuprl/MultiPL-E,humaneval-r" \
                        or self.type == "nuprl/MultiPL-E,mbpp-cpp" or self.type == "nuprl/MultiPL-E,mbpp-java":
                        dataset_list.append(load_dataset(*self.type.split(","), revision='5d2abbb8ced9a0e37db985c47d24c24f45a16655')[split])
                    elif self.type.startswith("mbpp-pseudo") and os.path.isfile(self.type.split(',')[1]):
                        print(self.type.split(',')[1])
                        dataset_list.append(load_dataset('json', data_files=self.type.split(',')[1])[split])
                    else:
                        dataset_list.append(load_dataset(self.type)[split])
                elif isinstance(self.type, list):
                    dataset_list.append(load_dataset(*self.type)[split])
                elif isinstance(self.type, tuple):
                    dataset_list.append(load_dataset(*self.type)[split])
                else:
                    raise NotImplementedError
            dataset = concatenate_datasets(dataset_list)

            # put it to dataframe
            for d in dataset:
                self.data.append(self.Datapoint.from_dict(d, **self.datapoint_kwargs))

        # select indices
        if self.indices is not None:
            self.data = [self.data[i] for i in self.indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def select(self, ids: List[str]):
        # select rows whose id is in ids
        indices = []
        for i, datapoint in enumerate(self.data):
            if datapoint.id in ids:
                indices.append(i)

        return ProblemDataset(
            type=self.type,
            split=self.split,
            indices=indices,
            Datapoint=self.Datapoint,
            path=self.path,
        )

    def to_dicts(self):
        return [datapoint.to_dict() for datapoint in self.data]
