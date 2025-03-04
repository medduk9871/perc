from typing import Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS

from src.chain.description.constraint import ConstraintChain
from src.chain.description.description import DescriptionChain
from src.chain.description.templates.humaneval import HUMANEVAL_EXAMPLES


class FilterChain(Chain):
    dataset_type: str = "humaneval"  # "humaneval" or "codecontests"
    mode: Optional[List[str]] = [
        "head",
        "prompt",
        "description",
        "constraint",
    ]
    constraint_mode: Optional[List[str]] = ["head", "prompt"]
    description_mode: Optional[List[str]] = ["head", "prompt", "constraint"]
    llm: BaseLanguageModel
    description_examples: Optional[List[dict]] = None
    constraint_examples: Optional[List[dict]] = None
    verbose: bool = False

    @property
    def input_keys(self) -> List[str]:
        return [
            "head",
            "prompt",
            "plan",
            "ic",
            "description",
            "constraint",
        ]

    @property
    def output_keys(self) -> List[str]:
        return [
            "reference_description",
            "reference_constraint",
            "filtered_plan",
            "filtered_ic",
            "filtered_description",
            "filtered_constraint",
        ]

    def _call(
        self,
        inputs: Dict[str, List[str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        if "constraint" in self.mode:
            reference_constraint = ConstraintChain(
                dataset_type=self.dataset_type,
                mode=self.constraint_mode,
                llm=self.llm,
                examples=self.constraint_examples,
                verbose=self.verbose,
            )(
                {
                    "head": inputs["head"],
                    "prompt": inputs["prompt"],
                    "plan": ["" for _ in range(self.llm.n)],
                    "ic": ["" for _ in range(self.llm.n)],
                }
            )
        else:
            reference_constraint = {"constraint": ["" for _ in range(self.llm.n)]}

        if "description" in self.mode:
            reference_description = DescriptionChain(
                dataset_type=self.dataset_type,
                mode=self.description_mode,
                llm=self.llm,
                examples=self.description_examples,
                verbose=self.verbose,
            )(
                {
                    "head": inputs["head"],
                    "prompt": inputs["prompt"],
                    "plan": ["" for _ in range(self.llm.n)],
                    "ic": ["" for _ in range(self.llm.n)],
                    "constraint": reference_constraint["constraint"],
                }
            )
        else:
            reference_description = {"description": ["" for _ in range(self.llm.n)]}

        candidates = []
        for plan, ic, description, constraint in zip(
            inputs["plan"],
            inputs["ic"],
            inputs["description"],
            inputs["constraint"],
        ):
            candidates.append(
                {
                    "plan": plan,
                    "ic": ic,
                    "description": description,
                    "constraint": constraint,
                }
            )

        selector = SemanticSimilarityExampleSelector.from_examples(
            candidates,
            OpenAIEmbeddings(),
            FAISS,
            k=1,
            input_keys=list(set(self.mode) - set(["head", "prompt"])),
        )

        selected = selector.select_examples(
            input_variables={
                "description": reference_description["description"][0],
                "constraint": reference_constraint["constraint"][0],
            }
        )[0]

        return {
            "description": inputs["description"],
            "constraint": inputs["constraint"],
            "reference_description": reference_description["description"][0],
            "reference_constraint": reference_constraint["constraint"][0],
            "filtered_plan": [selected["plan"]],
            "filtered_ic": [selected["ic"]],
            "filtered_description": [selected["description"]],
            "filtered_constraint": [selected["constraint"]],
        }
