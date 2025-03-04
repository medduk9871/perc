from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel

import src.chain.description.templates.codecontests as CODECONTESTS_TEMPLATES
import src.chain.description.templates.humaneval as HUMANEVAL_TEMPLATES
from src.utils.indent import FewShotIndChatPromptTemplate


class CodeChain(Chain):
    dataset_type: str = "humaneval"  # "humaneval" or "codecontests"
    llm: BaseChatModel
    mode: List[str] = ["head", "prompt", "plan", "ic", "constraint"]
    examples: Optional[List[dict]] = None
    verbose: bool = False

    prompt: str = None
    templates: Any = None

    @property
    def input_keys(self) -> List[str]:
        return ["head", "prompt", "plan", "ic", "constraint"]

    @property
    def output_keys(self) -> List[str]:
        return ["code"]

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
            example_prompt=(
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
            + [self.templates.CODE["human"], self.templates.CODE["ai"]],
            messages=(
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
            + [self.templates.CODE["human"]],
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

        generations = chain.generate(
            [
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "plan": p,
                    "ic": ic,
                    "constraint": c,
                }
                for p, ic, c in zip(
                    inputs["plan"],
                    inputs["ic"],
                    inputs["constraint"],
                )
            ]
        ).generations
        code = {
            "code": [
                self.templates.PARSER().parse_code(
                    head=inputs["head"],
                    text=g[0].text,
                )
                for g in generations
            ]
        }

        return code
