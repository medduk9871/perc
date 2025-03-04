from typing import Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel

from src.chain.constraint.main import (
    AI_CODE,
    AI_CONST,
    AI_IC,
    AI_PLAN,
    HUM_CODE,
    HUM_CONST,
    HUM_IC,
    HUM_PLAN,
    SYS,
)
from src.utils.indent import FewShotIndChatPromptTemplate


class CodeChain(Chain):
    llm: BaseChatModel
    mode: List[str] = ["plan", "ic", "constraint"]
    examples: Optional[List[dict]] = None
    verbose: bool = False

    prompt: str = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt", "head", "id", "plan", "constraint", "ic"]

    @property
    def output_keys(self) -> List[str]:
        return ["code"]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        prompt = FewShotIndChatPromptTemplate(
            examples=self.examples,
            prefix=[SYS],
            example_prompt=([HUM_PLAN, AI_PLAN] if "plan" in self.mode else [])
            + ([HUM_IC, AI_IC] if "ic" in self.mode else [])
            + ([HUM_CONST, AI_CONST] if "constraint" in self.mode else [])
            + [HUM_CODE, AI_CODE],
            messages=([HUM_PLAN, AI_PLAN] if "plan" in self.mode else [])
            + ([HUM_IC, AI_IC] if "ic" in self.mode else [])
            + ([HUM_CONST, AI_CONST] if "constraint" in self.mode else [])
            + [HUM_CODE],
            input_variables=["prompt", "head"]
            + (["plan"] if "plan" in self.mode else [])
            + (["ic"] if "ic" in self.mode else [])
            + (["constraint"] if "constraint" in self.mode else []),
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
                    "plan": p,
                    "ic": ic,
                    "constraint": c,
                }
                for p, ic, c in zip(inputs["plan"], inputs["ic"], inputs["constraint"])
            ]
        ).generations
        code = {"code": [g[0].text for g in generations]}

        return code
