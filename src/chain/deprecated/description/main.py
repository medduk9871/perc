from copy import deepcopy
from typing import Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms.base import BaseLanguageModel

from src.chain.description.code import CodeChain
from src.chain.description.templates.humaneval import HUMANEVAL_EXAMPLES


class MainChain(Chain):
    mode: Dict[str, List[str]] = {
        "plan": ["head", "prompt"],
        "ic": ["head", "prompt", "plan"],
        "constraint": ["head", "prompt", "ic"],
        "description": ["head", "prompt", "ic", "constraint"],
        "filter": ["constraint", "description"],
        "code": ["plan", "constraint"],
    }
    llm: BaseLanguageModel
    examples: Optional[List[dict]] = None
    verbose: bool = False
    dataset_type: str = "humaneval"  # "humaneval" or "codecontests"

    prompt: str = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt", "head"]

    @property
    def output_keys(self) -> List[str]:
        return [
            "plan",
            "ic",
            "description",
            "constraint",
            "reference_description",
            "reference_constraint",
            "filtered_plan",
            "filtered_ic",
            "filtered_description",
            "filtered_constraint",
            "code",
        ]

    class Config:
        arbitrary_types_allowed = True

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        orig_n = self.llm.n
        llm = deepcopy(self.llm)

        if "plan" in self.mode:
            from src.chain.description.plan import PlanChain

            chain = PlanChain(
                dataset_type=self.dataset_type,
                llm=llm,
                examples=self.examples,
                verbose=self.verbose,
            )
            plan = chain(
                {
                    "head": inputs["head"],
                    "prompt": inputs["prompt"],
                }
            )

            llm.n = 1
        else:
            plan = {"plan": ["" for _ in range(orig_n)]}

        if "ic" in self.mode:
            from src.chain.description.ic import ICChain

            chain = ICChain(
                dataset_type=self.dataset_type,
                mode=self.mode["ic"],
                llm=llm,
                examples=self.examples,
                verbose=self.verbose,
            )
            ic = chain(
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "plan": plan["plan"],
                }
            )

            llm.n = 1
        else:
            ic = {"ic": ["" for _ in range(orig_n)]}

        if "constraint" in self.mode:
            from src.chain.description.constraint import ConstraintChain

            chain = ConstraintChain(
                dataset_type=self.dataset_type,
                mode=self.mode["constraint"],
                llm=llm,
                examples=HUMANEVAL_EXAMPLES,
                verbose=self.verbose,
            )
            constraint = chain(
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "plan": plan["plan"],
                    "ic": ic["ic"],
                }
            )

            llm.n = 1
        else:
            constraint = {"constraint": ["" for _ in range(orig_n)]}

        if "description" in self.mode:
            from src.chain.description.description import DescriptionChain

            chain = DescriptionChain(
                dataset_type=self.dataset_type,
                mode=self.mode["description"],
                llm=llm,
                examples=HUMANEVAL_EXAMPLES,
                verbose=self.verbose,
            )
            description = chain(
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "plan": plan["plan"],
                    "ic": ic["ic"],
                    "constraint": constraint["constraint"],
                }
            )

            llm.n = 1
        else:
            description = {"description": ["" for _ in range(orig_n)]}

        if "filter" in self.mode:
            assert llm.n == 1

            from src.chain.description.filter import FilterChain

            chain = FilterChain(
                dataset_type=self.dataset_type,
                mode=self.mode["filter"],
                constraint_examples=HUMANEVAL_EXAMPLES,
                description_examples=HUMANEVAL_EXAMPLES,
                constraint_mode=list(
                    set(self.mode["constraint"]) - set(["plan", "ic"])
                ),
                description_mode=list(
                    set(self.mode["description"]) - set(["plan", "ic"])
                ),
                llm=llm,
                verbose=self.verbose,
            )
            filtered = chain(
                {
                    "head": inputs["head"],
                    "prompt": inputs["prompt"],
                    "plan": plan["plan"],
                    "ic": ic["ic"],
                    "description": description["description"],
                    "constraint": constraint["constraint"],
                }
            )
        else:
            filtered = {
                "reference_description": None,
                "reference_constraint": None,
                "filtered_plan": plan["plan"],
                "filtered_ic": ic["ic"],
                "filtered_description": description["description"],
                "filtered_constraint": constraint["constraint"],
            }

        chain = CodeChain(
            dataset_type=self.dataset_type,
            mode=self.mode["code"],
            llm=llm,
            examples=self.examples,
            verbose=self.verbose,
        )
        code = chain(
            {
                "head": inputs["head"],
                "prompt": inputs["prompt"],
                "plan": filtered["filtered_plan"],
                "ic": filtered["filtered_ic"],
                "constraint": filtered["filtered_constraint"],
            }
        )

        return {
            "plan": [plan["plan"]],
            "ic": [ic["ic"]],
            "description": [description["description"]],
            "constraint": [constraint["constraint"]],
            "reference_description": filtered["reference_description"],
            "reference_constraint": filtered["reference_constraint"],
            "filtered_plan": filtered["filtered_plan"],
            "filtered_ic": filtered["filtered_ic"],
            "filtered_description": filtered["filtered_description"],
            "filtered_constraint": filtered["filtered_constraint"],
            "code": code["code"],
        }
