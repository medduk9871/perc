from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma

from src.chain.plan_to_prompt import Plan2PromptChain


class PlanBTFilterChain(Chain):
    llm_kwargs: Dict[str, Any] = {}
    dataset_type: str
    examples: List[dict]
    example_num: int
    verbose: bool = False
    k: int

    @property
    def input_keys(self) -> List[str]:
        return ["target_id", "target_plan", "target_prompt", "target_head"]

    @property
    def output_keys(self) -> List[str]:
        return ["target_back_prompt", "target_filtered_plan"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _call(
        self,
        inputs: Dict[str, List[str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # inputs["target_plan"] shape: (1, num_plans)
        inputs = {
            **inputs,
            "target_plan": inputs["target_plan"][0],
        }

        target_back_prompt = Plan2PromptChain(
            llm_kwargs=self.llm_kwargs,
            dataset_type=self.dataset_type,
            examples=self.examples,
            example_num=self.example_num,
            verbose=self.verbose,
        )(inputs)

        tbp_texts = target_back_prompt["target_back_prompt"]
        candidates = [
            {
                "target_plan": inputs["target_plan"][i],
                "target_prompt": inputs["target_prompt"],
                "target_head": inputs["target_head"],
                "target_id": inputs["target_id"],
                "target_back_prompt": tbp_texts[i],
            }
            for i in range(len(tbp_texts))
        ]

        selector = SemanticSimilarityExampleSelector.from_examples(
            candidates,
            OpenAIEmbeddings(),
            Chroma,
            k=self.k,
            input_keys=["target_back_prompt"],
        )

        selected = selector.select_examples(
            input_variables={
                "target_back_prompt": inputs["target_prompt"][0],
            }
        )

        return {
            "target_back_prompt": [tbp_texts],
            "target_filtered_plan": [[s["target_plan"] for s in selected]],
        }
