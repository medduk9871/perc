from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from src.chain.code_to_result import Code2ResultChain
from src.chain.plan_to_code import Plan2CodeChain

from src.evaluator import Evaluator


class PlanOracleFilterChain(Chain):
    llm_kwargs: Dict[str, Any] = {}
    dataset_type: str
    examples: List[dict]
    example_num: int
    verbose: bool = False
    k: int

    @property
    def input_keys(self) -> List[str]:
        return [
            "target_id",
            "target_plan",
            "target_prompt",
            "target_head",
            "target_test",
        ]

    @property
    def output_keys(self) -> List[str]:
        return ["target_results", "target_filtered_plan"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        inputs = {
            **inputs,
            "target_plan": inputs["target_plan"][0],
        }

        target_code = Plan2CodeChain(
            llm_kwargs=self.llm_kwargs,
            dataset_type=self.dataset_type,
            examples=self.examples,
            example_num=self.example_num,
            verbose=self.verbose,
            n=1,
        )

        result = Code2ResultChain(verbose=self.verbose)(
            {
                "target_code": target_code["target_code"],
                "target_test": inputs["target_test"],
            }
        )

        idx = (
            result["target_result"].index("passed")
            if "passed" in result["target_result"]
            else 0
        )

        return {
            "target_results": result["target_result"],
            "target_filtered_plan": [inputs["target_plan"][idx]],
        }
