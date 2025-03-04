from typing import Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel

from src.chain.constraint.main import AI_IC, AI_PLAN, HUM_IC, HUM_PLAN, HUM_PROB, SYS
from src.utils.indent import FewShotIndChatPromptTemplate


class ICChain(Chain):
    llm: BaseChatModel
    mode: Optional[List[str]] = ["plan"]
    examples: Optional[List[dict]] = None
    verbose: bool = False

    prompt: str = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt", "head", "id", "plan"]

    @property
    def output_keys(self) -> List[str]:
        return ["ic"]

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
            + [HUM_IC, AI_IC],
            messages=[HUM_PROB]
            + ([HUM_PLAN, AI_PLAN] if "plan" in self.mode else [])
            + [HUM_IC],
            input_variables=["prompt", "head", "plan"],
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
                }
                for p in inputs["plan"]
            ]
        ).generations
        ic = {"ic": [g[0].text for g in generations]}

        return ic
