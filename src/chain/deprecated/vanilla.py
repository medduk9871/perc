from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.utils.indent import HumanIndMessagePromptTemplate as HIMPT
from src.utils.indent import SystemIndMessagePromptTemplate as SIMPT

SYSTEM_TEMPLATE = """\
Write a raw code snippet that solves the given problem."""

HUM_TEMP_HE = """\
${head}
    ${prompt}
    raise NotImplementedError"""

HUM_TEMP_CC = """\
${prompt}"""


class VanillaChain(Chain):
    llm_kwargs: Dict[str, Any] = {}
    dataset_type: str = "openai_humaneval"

    hum_temp: HIMPT = None
    sys_temp: SIMPT = None
    llm: ChatOpenAI = None

    @property
    def input_keys(self) -> List[str]:
        return ["target_prompt", "target_head", "target_id"]

    @property
    def output_keys(self) -> List[str]:
        return ["target_code"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.dataset_type == "openai_humaneval":
            self.hum_temp = HUM_TEMP_HE.replace("${", "${target_")
        elif self.dataset_type == "deepmind/code_contests":
            self.hum_temp = HUM_TEMP_CC.replace("${", "${target_")
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.sys_temp = SIMPT.from_template(template=SYSTEM_TEMPLATE)
        self.hum_temp = HIMPT.from_template(template=self.hum_temp)

        self.llm = ChatOpenAI(
            model=self.llm_kwargs.get("model"),
            n=self.llm_kwargs.get("n"),
            max_tokens=self.llm_kwargs.get("max_tokens"),
            model_kwargs={"top_p": self.llm_kwargs.get("top_p")},
            temperature=self.llm_kwargs.get("temperature"),
            max_retries=100,
        )

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        chat_prompt = ChatPromptTemplate.from_messages([self.sys_temp, self.hum_temp])

        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        result = chain.generate([inputs]).generations[0]

        return {self.output_key: [r.text for r in result]}

    @property
    def _chain_type(self) -> str:
        return "vanilla_chain"
