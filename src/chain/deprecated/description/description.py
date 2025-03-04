from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel

import src.chain.description.templates.codecontests as CODECONTESTS_TEMPLATES
import src.chain.description.templates.humaneval as HUMANEVAL_TEMPLATES
from src.utils.indent import FewShotIndChatPromptTemplate


class DescriptionChain(Chain):
    dataset_type: str = "humaneval"  # "humaneval" or "codecontests"
    mode: List[str] = ["head", "prompt", "plan", "ic", "constraint"]
    llm: BaseChatModel
    examples: Optional[List[dict]] = None
    verbose: bool = False

    prompt: str = None
    templates: Any = None

    @property
    def input_keys(self) -> List[str]:
        return ["head", "prompt", "plan", "ic", "constraint"]

    @property
    def output_keys(self) -> List[str]:
        return ["description"]

    def __init__(self, **data):
        super().__init__(**data)

        if self.dataset_type == "openai_humaneval":
            self.templates = HUMANEVAL_TEMPLATES
        elif self.dataset_type == "deepmind/code_contests":
            self.templates = CODECONTESTS_TEMPLATES
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
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
            + (
                [self.templates.CONST["human"], self.templates.CONST["ai"]]
                if "constraint" in self.mode
                else []
            )
            + [self.templates.DESC["human"], self.templates.DESC["ai"]],
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
            + (
                [self.templates.CONST["human"], self.templates.CONST["ai"]]
                if "constraint" in self.mode
                else []
            )
            + [self.templates.DESC["human"]],
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=self.verbose,
        )

        assert self.llm.n == 1
        generations = chain.generate(
            [
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "plan": plan,
                    "ic": ic,
                    "constraint": constraint,
                }
                for plan, ic, constraint in zip(
                    inputs["plan"],
                    inputs["ic"],
                    inputs["constraint"],
                )
            ]
        ).generations
        description = {
            "description": [
                self.templates.PARSER().parse(g[0].text) for g in generations
            ]
        }

        return description
