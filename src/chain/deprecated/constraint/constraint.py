from typing import Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel

from src.chain.constraint.main import (
    AI_CONST,
    AI_IC,
    AI_PLAN,
    HUM_CONST,
    HUM_IC,
    HUM_PLAN,
    HUM_PROB,
    SYS,
)
from src.utils.indent import FewShotIndChatPromptTemplate


class ConstraintChain(Chain):
    llm: BaseChatModel
    mode: List[str] = ["plan", "ic"]
    examples: Optional[List[dict]] = None
    verbose: bool = False

    prompt: str = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt", "head", "id", "plan", "ic"]

    @property
    def output_keys(self) -> List[str]:
        return ["constraint"]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        prompt = FewShotIndChatPromptTemplate(
            examples=self.examples,
            prefix=[SYS],
            example_prompt=[HUM_PROB]
            + ([HUM_PLAN, AI_PLAN] if "plan" in self.mode else [])
            + ([HUM_IC, AI_IC] if "ic" in self.mode else [])
            + [HUM_CONST, AI_CONST],
            messages=[HUM_PROB]
            + ([HUM_PLAN, AI_PLAN] if "plan" in self.mode else [])
            + ([HUM_IC, AI_IC] if "ic" in self.mode else [])
            + [HUM_CONST],
            input_variables=["prompt", "head"]
            + (["plan"] if "plan" in self.mode else [])
            + (["ic"] if "ic" in self.mode else []),
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=self.verbose,
        )

        generations = chain.generate(
            [
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "plan": plan,
                    "ic": ic,
                }
                for plan, ic in zip(inputs["plan"], inputs["ic"])
            ]
        ).generations
        constraint = {"constraint": [g[0].text for g in generations]}

        return constraint
