from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma

from src.chain.constraint.constraint import ConstraintChain


class FilterChain(Chain):
    llm_kwargs: Dict[str, Any] = {}
    examples: List[dict]
    verbose: bool = False

    @property
    def input_keys(self) -> List[str]:
        return ["id", "head", "prompt", "plan", "ic", "constraint"]

    @property
    def output_keys(self) -> List[str]:
        return [
            "reference_constraint",
            "filtered_plan",
            "filtered_ic",
            "filtered_constraint",
        ]

    class Config:
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)
        self.cache_dict = data["cache_dict"]
        self.cache_dict_lock = data["cache_dict_lock"]

    def _call(
        self,
        inputs: Dict[str, List[str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # inputs["target_plan"] shape: (1, num_plans)
        inputs = {
            **inputs,
            "plan": inputs["plan"][0],
            "ic": inputs["ic"][0],
            "constraint": inputs["constraint"][0],
        }

        reference_constraint = ConstraintChain(
            mode=[],
            llm=ChatOpenAI(
                model=self.llm_kwargs.get("model"),
                n=1,
                max_tokens=self.llm_kwargs.get("max_tokens"),
                model_kwargs={"top_p": self.llm_kwargs.get("top_p")},
                temperature=self.llm_kwargs.get("temperature"),
                max_retries=100,
            ),
            examples=self.examples,
            verbose=self.verbose,
        )(inputs)

        candidates = [
            {
                "plan": inputs["plan"][i],
                "ic": inputs["ic"][i],
                "constraint": inputs["constraint"][i],
            }
            for i in range(len(inputs["plan"]))
        ]

        selector = SemanticSimilarityExampleSelector.from_examples(
            candidates,
            OpenAIEmbeddings(),
            Chroma,
            k=1,
            input_keys=["constraint"],
        )

        selected = selector.select_examples(
            input_variables={
                "constraint": reference_constraint["constraint"][0],
            }
        )[0]

        return {
            "reference_constraint": reference_constraint["constraint"][0],
            "filtered_plan": [selected["plan"]],
            "filtered_ic": [selected["ic"]],
            "filtered_constraint": selected["constraint"],
        }
