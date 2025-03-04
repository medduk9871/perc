from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import FewShotChatMessagePromptTemplate

import src.chain.contract.templates.codecontests as CODECONTESTS_TEMPLATES
import src.chain.contract.templates.humaneval as HUMANEVAL_TEMPLATES


class CodeChain(Chain):
    dataset_type: str = "humaneval"  # "humaneval" or "codecontests"
    llm_kwargs: Dict[str, Any] = {}
    examples: Optional[List[dict]] = None
    verbose: bool = False
    parents: List[str]

    prompt: str = None
    templates: Any = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt", "plan", "contract"]

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

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=self.templates.CODE.human + self.templates.CODE.ai,
            examples=self.examples,
        )
        self.prompt = (
            self.templates.CODE.system + fewshot_prompt + self.templates.CODE.human
        )

        llm = AzureChatOpenAI(
            deployment_name=self.llm_kwargs["model"],
            max_tokens=self.llm_kwargs["max_tokens"],
            temperature=self.llm_kwargs["temperature"],
            n=self.llm_kwargs["n"],
            model_kwargs={"top_p": self.llm_kwargs["top_p"]},
        )

        chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            verbose=self.verbose,
        )

        if llm.n == 1:
            generations = chain.generate(
                [
                    {
                        "prompt": inputs["prompt"],
                        "plan": p,
                        "contract": c,
                    }
                    for p, c in zip(
                        inputs["plan"],
                        inputs["contract"],
                    )
                ]
            ).generations
            code = {"code": [self.templates.IMPORTS + g[0].text for g in generations]}
        else:
            generations = chain.generate([inputs]).generations[0]
            code = {"code": [self.templates.IMPORTS + g.text for g in generations]}

        return code
