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

from src.chain.plan import (
    SYS_TEMP,
    HUM_PROB_TEMP_HE,
    AI_PLAN_TEMP_HE,
    HUM_CODE_TEMP_HE,
    AI_CODE_TEMP_HE,
    HUM_PROB_TEMP_CC,
    AI_PLAN_TEMP_CC,
    HUM_CODE_TEMP_CC,
    AI_CODE_TEMP_CC,
)


class Prompt2PlanChain(Chain):
    llm_kwargs: Dict[str, Any] = {}
    dataset_type: str
    examples: List[dict]
    example_num: int
    verbose: bool = False

    sys_temp: SIMPT = None
    ex_hum_prob_temp: HIMPT = None
    ex_ai_plan_temp: AIMPT = None
    ex_hum_code_temp: HIMPT = None
    ex_ai_code_temp: AIMPT = None
    tar_hum_prob_temp: HIMPT = None
    llm: ChatOpenAI = None

    @property
    def input_keys(self) -> List[str]:
        return ["target_prompt", "target_head", "target_id"]

    @property
    def output_keys(self) -> List[str]:
        return ["target_plan"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # prepare templates
        if self.dataset_type == "openai_humaneval":
            hum_prob_temp = HUM_PROB_TEMP_HE
            ai_plan_temp = AI_PLAN_TEMP_HE
            hum_code_temp = HUM_CODE_TEMP_HE
            ai_code_temp = AI_CODE_TEMP_HE
        elif self.dataset_type == "deepmind/code_contests":
            hum_prob_temp = HUM_PROB_TEMP_CC
            ai_plan_temp = AI_PLAN_TEMP_CC
            hum_code_temp = HUM_CODE_TEMP_CC
            ai_code_temp = AI_CODE_TEMP_CC
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.ex_hum_prob_temp = hum_prob_temp.replace("${", "${example_")
        self.ex_ai_plan_temp = ai_plan_temp.replace("${", "${example_")
        self.ex_hum_code_temp = hum_code_temp.replace("${", "${example_")
        self.ex_ai_code_temp = ai_code_temp.replace("${", "${example_")
        self.tar_hum_prob_temp = hum_prob_temp.replace("${", "${target_")

        self.sys_temp = SIMPT.from_template(template=SYS_TEMP)
        self.ex_hum_prob_temp = HIMPT.from_template(template=self.ex_hum_prob_temp)
        self.ex_ai_plan_temp = AIMPT.from_template(template=self.ex_ai_plan_temp)
        self.ex_hum_code_temp = HIMPT.from_template(template=self.ex_hum_code_temp)
        self.ex_ai_code_temp = AIMPT.from_template(template=self.ex_ai_code_temp)
        self.tar_hum_prob_temp = HIMPT.from_template(template=self.tar_hum_prob_temp)

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
                self.ex_hum_prob_temp,
                self.ex_ai_plan_temp,
                self.ex_hum_code_temp,
                self.ex_ai_code_temp,
            ],
            prefix=[self.sys_temp],
            messages=[
                self.tar_hum_prob_temp,
            ],
            input_variables=["target_prompt", "target_head"],
        )

        # prepare llmchain
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        result = chain.generate([inputs]).generations[0]

        return {"target_plan": [r.text for r in result]}
