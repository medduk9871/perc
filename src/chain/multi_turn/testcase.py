from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.chain.multi_turn.selector import SelectorChain

import src.chain.multi_turn.templates.codecontests as CODECONTESTS_TEMPLATES
import src.chain.multi_turn.templates.humaneval as HUMANEVAL_TEMPLATES


class TestcaseChain(Chain):
    dataset_type: str
    llm_kwargs: Dict[str, Any]
    examples: Optional[List[dict]] = None
    verbose: bool = False
    parents: List[str]

    prompt: str = None
    templates: Any = None

    @property
    def input_keys(self) -> List[str]:
        keys = ["prompt"]
        full_list = ["draft_plan", "requirements", "final_plan"]
        for k in full_list:
            if k in self.parents:
                keys.append(k)
        return keys

    @property
    def output_keys(self) -> List[str]:
        return ["gen_tc"]

    def __init__(self, **data):
        super().__init__(**data)

        if self.dataset_type == "openai_humaneval":
            self.templates = HUMANEVAL_TEMPLATES
        elif self.dataset_type == "deepmind/code_contests":
            self.templates = CODECONTESTS_TEMPLATES
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        prefix_messages = [self.templates.PROMPT.human]
        if "draft_plan" in self.parents:
            prefix_messages.append(self.templates.DRAFT_PLAN.human)
            prefix_messages.append(self.templates.DRAFT_PLAN.ai)
        if "requirements" in self.parents:
            prefix_messages.append(self.templates.REQUIREMENTS.human)
            prefix_messages.append(self.templates.REQUIREMENTS.ai)
        if "final_plan" in self.parents:
            prefix_messages.append(self.templates.FINAL_PLAN.human)
            prefix_messages.append(self.templates.FINAL_PLAN.ai)
        if len(prefix_messages) > 1 and self.dataset_type == "openai_humaneval":
            prefix_messages.append(self.templates.PROMPT.human)
        prefix_messages.append(self.templates.TESTCASE.human)
        example_messages = prefix_messages + [self.templates.TESTCASE.ai]

        example_prompt = ChatPromptTemplate.from_messages(example_messages)
        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [fewshot_prompt] + prefix_messages
        )

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        llm = AzureChatOpenAI(
            deployment_name=self.llm_kwargs["model"],
            max_tokens=self.llm_kwargs["max_tokens"],
            temperature=self.llm_kwargs["temperature"],
            n=self.llm_kwargs["n"],
            model_kwargs={"top_p": self.llm_kwargs["top_p"]},
            max_retries=self.llm_kwargs["max_retries"],
        )

        chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            verbose=self.verbose,
        )

        if llm.n == 1:
            n = 1
            for k in self.input_keys:
                if isinstance(inputs[k], list):
                    n = len(inputs[k])
                    break

            generations = chain.generate(
                [
                    {
                        "prompt": inputs["prompt"],
                        **{k: inputs[k][i] for k in set(self.input_keys) - {"prompt"}},
                    }
                    for i in range(n)
                ]
            ).generations
            gen_tc = {"gen_tc": [g[0].text for g in generations]}
        else:
            generations = chain.generate([inputs]).generations[0]
            gen_tc = {"gen_tc": [g.text for g in generations]}

        return gen_tc


class TestcaseSelectorChain(SelectorChain):
    key: str = "gen_tc"
    dataset_type: str

    def __init__(self, **data):
        super().__init__(**data)

        if self.dataset_type == "openai_humaneval":
            self.template = HUMANEVAL_TEMPLATES.TESTCASE
        elif self.dataset_type == "deepmind/code_contests":
            self.template = CODECONTESTS_TEMPLATES.TESTCASE
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
