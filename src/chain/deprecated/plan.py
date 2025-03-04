from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from src.chain.plan_bt_filter import PlanBTFilterChain

SYS_TEMP = """\
Write a plan and raw code snippet that solves the given problem."""

HUM_PROB_TEMP_HE = """\
${head}
    ${prompt}
    
    # Let's think step by step"""

AI_PLAN_TEMP_HE = """\
${head}
    # Let's think step by step
    ${plan}"""

HUM_CODE_TEMP_HE = """\
${head}
    ${prompt}
    
    # Let's write code"""

AI_CODE_TEMP_HE = """\
${head}
    # Let's write code
    ${code}"""

HUM_PROB_TEMP_CC = """\
${prompt}"""

AI_PLAN_TEMP_CC = """\
# Let's think step by step
${plan}"""

HUM_CODE_TEMP_CC = """\
# Let's write code"""

AI_CODE_TEMP_CC = """\
# Let's write code
${code}"""


class PlanChain(Chain):
    llm_kwargs: Dict[str, Any] = {}
    dataset_type: str
    examples: Optional[List[dict]] = None
    example_num: Optional[int] = None
    verbose: bool = False
    mode: Optional[str] = None

    prompt: str = None
    example_selector: SemanticSimilarityExampleSelector = None

    @property
    def input_keys(self) -> List[str]:
        return ["target_prompt", "target_head", "target_id"]

    @property
    def output_keys(self) -> List[str]:
        return [
            "target_plan_candidates",
            "target_back_prompt",
            "target_plan",
            "target_code",
        ]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        from src.chain.prompt_to_plan import Prompt2PlanChain

        target_plan_candidates = (
            Prompt2PlanChain(
                llm_kwargs=self.llm_kwargs,
                dataset_type=self.dataset_type,
                examples=self.examples,
                example_num=self.example_num,
                verbose=self.verbose,
            ),
        )

        if self.mode.startswith("filter_"):
            target_plan = PlanBTFilterChain(
                llm_kwargs=self.llm_kwargs,
                dataset_type=self.dataset_type,
                examples=self.examples,
                example_num=self.example_num,
                verbose=self.verbose,
                k=1,
            )(
                {
                    "target_id": inputs["target_id"],
                    "target_prompt": inputs["target_prompt"],
                    "target_head": inputs["target_head"],
                    "target_plan": [target_plan_candidates["target_plan"]],
                }
            )
            target_plan = {
                "target_back_prompt": target_plan["target_back_prompt"],
                "target_plan": target_plan["target_filtered_plan"][0],
            }
        else:
            target_plan = {
                "target_back_prompt": None,
                "target_plan": target_plan_candidates["target_plan"],
            }

        # prepare chain
        from src.chain.plan_to_code import Plan2CodeChain

        if self.mode in ["only_plan", "filter_generation", "filter_generation_greedy"]:
            if self.mode == "only_plan":
                artifact_name = "plan_to_code"
                llm_kwargs = self.llm_kwargs
            elif self.mode == "filter_generation":
                artifact_name = "plan_to_code-filter_generation"
                llm_kwargs = self.llm_kwargs
            elif self.mode == "filter_generation_greedy":
                artifact_name = f"plan_to_code-{self.mode}"
                llm_kwargs = {
                    **self.llm_kwargs,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            plan_to_code_chain = Plan2CodeChain(
                llm_kwargs=llm_kwargs,
                dataset_type=self.dataset_type,
                examples=self.examples,
                example_num=self.example_num,
                verbose=self.verbose,
                n=1,
            )

            target_code = plan_to_code_chain(
                {
                    "target_plan": target_plan["target_plan"],
                    "target_id": inputs["target_id"],
                    "target_prompt": inputs["target_prompt"],
                    "target_head": inputs["target_head"],
                }
            )

            return {
                "target_plan_candidates": [target_plan_candidates["target_plan"]],
                "target_back_prompt": target_plan["target_back_prompt"],
                "target_code": target_code["target_code"],
                "target_plan": target_plan["target_plan"],
            }

        elif self.mode.endswith("selection"):
            plan_to_code_chain = Plan2CodeChain(
                llm_kwargs=self.llm_kwargs,
                dataset_type=self.dataset_type,
                examples=self.examples,
                example_num=self.example_num,
                verbose=self.verbose,
                n=1,
            )
            target_code = plan_to_code_chain(target_plan_candidates)

            idx = target_plan_candidates["target_plan"].index(
                target_plan["target_plan"][0]
            )

            return {
                "target_plan_candidates": [target_plan_candidates["target_plan"]],
                "target_back_prompt": target_plan["target_back_prompt"],
                "target_code": [target_code["target_code"][idx]],
                "target_plan": target_plan["target_plan"],
            }

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
