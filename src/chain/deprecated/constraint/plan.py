from typing import Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel

from src.chain.constraint.main import AI_PLAN, HUM_PLAN, HUM_PROB, SYS
from src.utils.indent import FewShotIndChatPromptTemplate


class PlanChain(Chain):
    llm: BaseChatModel
    examples: Optional[List[dict]] = None
    verbose: bool = False
    mode: Optional[str] = None

    prompt: str = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt", "head", "id"]

    @property
    def output_keys(self) -> List[str]:
        return ["plan"]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        plan_prompt = FewShotIndChatPromptTemplate(
            examples=self.examples,
            prefix=[SYS],
            example_prompt=[HUM_PROB, HUM_PLAN, AI_PLAN],
            messages=[HUM_PROB, HUM_PLAN],
            input_variables=["prompt", "head"],
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=plan_prompt,
            verbose=self.verbose,
        )

        generation = chain.generate([inputs]).generations[0]
        plan = {"plan": [g.text for g in generation]}

        return plan
