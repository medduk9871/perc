from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel

import src.chain.description.templates.codecontests as CODECONTESTS_TEMPLATES
import src.chain.description.templates.humaneval as HUMANEVAL_TEMPLATES
from src.utils.indent import FewShotIndChatPromptTemplate


class ConstraintChain(Chain):
    dataset_type: str = "humaneval"  # "humaneval" or "codecontests"
    mode: List[str] = ["head", "prompt", "plan", "ic"]
    llm: BaseChatModel
    examples: Optional[List[dict]] = None
    verbose: bool = False

    prompt: str = None
    templates: Any = None

    @property
    def input_keys(self) -> List[str]:
        return ["head", "prompt", "plan", "ic"]

    @property
    def output_keys(self) -> List[str]:
        return ["constraint"]

    def __init__(self, **data):
        super().__init__(**data)

        if self.dataset_type == "openai_humaneval":
            self.templates = HUMANEVAL_TEMPLATES
        elif self.dataset_type == "deepmind/code_contests":
            self.templates = CODECONTESTS_TEMPLATES
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        self.prompt = FewShotIndChatPromptTemplate(
            examples=self.examples,
            example_prompt=[]
            + (
                [self.templates.PROMPT["human"]]
                if self.dataset_type == "deepmind/code_contests"
                else []
            )
            + (
                [self.templates.PLAN["human"], self.templates.PLAN["ai"]]
                if "plan" in self.mode
                else []
            )
            + (
                [self.templates.IC["human"], self.templates.IC["ai"]]
                if "ic" in self.mode
                else []
            )
            + [self.templates.CONST["human"], self.templates.CONST["ai"]],
            messages=[]
            + (
                [self.templates.PROMPT["human"]]
                if self.dataset_type == "deepmind/code_contests"
                else []
            )
            + (
                [self.templates.PLAN["human"], self.templates.PLAN["ai"]]
                if "plan" in self.mode
                else []
            )
            + (
                [self.templates.IC["human"], self.templates.IC["ai"]]
                if "ic" in self.mode
                else []
            )
            + [self.templates.CONST["human"]],
        )

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=self.verbose,
        )

        if self.llm.n == 1:
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
            constraint = {
                "constraint": [
                    self.templates.PARSER().parse(g[0].text) for g in generations
                ]
            }
        else:
            generations = chain.generate([inputs]).generations[0]
            constraint = {
                "constraint": [
                    self.templates.PARSER().parse(g.text) for g in generations
                ]
            }

        return constraint
