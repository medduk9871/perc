from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma

from src.utils.indent import AIIndMessagePromptTemplate as AIMPT
from src.utils.indent import FewShotIndChatPromptTemplate
from src.utils.indent import HumanIndMessagePromptTemplate as HIMPT
from src.utils.indent import SystemIndMessagePromptTemplate as SIMPT

SYS_TEMP = """\
Write a raw code snippet that solves the given problem."""

HUM_TEMP_HE = """\
${head}
    ${prompt}
    raise NotImplementedError"""

HUM_TEMP_CC = """\
${prompt}"""

AI_TEMP_HE = """\
${head}
    ${code}"""

AI_TEMP_CC = """\
${code}"""


class FewShotChain(Chain):
    llm_kwargs: Dict[str, Any] = {}
    dataset_type: str
    examples: List[dict]
    example_num: int
    verbose: bool = False

    sys_temp: SIMPT = None
    ex_hum_temp: HIMPT = None
    ex_ai_temp: AIMPT = None
    tar_hum_temp: HIMPT = None
    llm: ChatOpenAI = None

    @property
    def input_keys(self) -> List[str]:
        return ["target_prompt", "target_head", "target_id"]

    @property
    def output_keys(self) -> List[str]:
        return ["target_code"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # prepare templates
        if self.dataset_type == "openai_humaneval":
            hum_temp = HUM_TEMP_HE
            ai_temp = AI_TEMP_HE
        elif self.dataset_type == "deepmind/code_contests":
            hum_temp = HUM_TEMP_CC
            ai_temp = AI_TEMP_CC
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.ex_hum_temp = hum_temp.replace("${", "${example_")
        self.ex_ai_temp = ai_temp.replace("${", "${example_")
        self.tar_hum_temp = hum_temp.replace("${", "${target_")

        self.sys_temp = SIMPT.from_template(template=SYS_TEMP)
        self.ex_hum_temp = HIMPT.from_template(template=self.ex_hum_temp)
        self.ex_ai_temp = AIMPT.from_template(template=self.ex_ai_temp)
        self.tar_hum_temp = HIMPT.from_template(template=self.tar_hum_temp)

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
    ) -> Dict[str, List[str]]:
        # select examples
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            OpenAIEmbeddings(),
            Chroma,
            k=self.example_num,
            input_keys=["example_prompt"],
        )
        examples = example_selector.select_examples(
            input_variables={
                "example_prompt": inputs["target_prompt"],
            }
        )

        # create chat prompt
        chat_prompt = FewShotIndChatPromptTemplate(
            examples=examples,
            example_prompt=[
                self.ex_hum_temp,
                self.ex_ai_temp,
            ],
            prefix=[self.sys_temp],
            messages=[self.tar_hum_temp],
            input_variables=["target_prompt", "target_head"],
        )

        chain = LLMChain(llm=self.llm, prompt=chat_prompt, verbose=self.verbose)
        result = chain.generate([inputs]).generations[0]

        return {"target_code": [r.text for r in result]}
