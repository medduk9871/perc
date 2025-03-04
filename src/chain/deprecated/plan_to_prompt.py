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
Write a problem description that can be solved by the given plan."""

HUM_PLAN_TEMP = """\
${plan}"""

AI_PROB_TEMP = """\
${prompt}"""


class Plan2PromptChain(Chain):
    llm_kwargs: Dict[str, Any] = {}
    dataset_type: str
    examples: List[dict]
    example_num: int
    verbose: bool = False

    sys_temp: SIMPT = None
    ex_hum_plan_temp: HIMPT = None
    ex_ai_prob_temp: AIMPT = None
    tar_hum_plan_temp: HIMPT = None
    llm: ChatOpenAI = None

    @property
    def input_keys(self) -> List[str]:
        return ["target_plan", "target_id"]

    @property
    def output_keys(self) -> List[str]:
        return ["target_back_prompt"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # prepare templates
        hum_plan_temp = HUM_PLAN_TEMP
        ai_prob_temp = AI_PROB_TEMP

        self.ex_hum_plan_temp = hum_plan_temp.replace("${", "${example_")
        self.ex_ai_prob_temp = ai_prob_temp.replace("${", "${example_")
        self.tar_hum_plan_temp = hum_plan_temp.replace("${", "${target_")

        self.sys_temp = SIMPT.from_template(template=SYS_TEMP)
        self.ex_hum_plan_temp = HIMPT.from_template(template=self.ex_hum_plan_temp)
        self.ex_ai_prob_temp = AIMPT.from_template(template=self.ex_ai_prob_temp)
        self.tar_hum_plan_temp = HIMPT.from_template(template=self.tar_hum_plan_temp)

        self.llm = ChatOpenAI(
            model=self.llm_kwargs.get("model"),
            n=1,
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
        # create chat prompt
        chat_prompt = FewShotIndChatPromptTemplate(
            examples=self.examples,
            example_prompt=[
                self.ex_hum_plan_temp,
                self.ex_ai_prob_temp,
            ],
            prefix=[self.sys_temp],
            messages=[
                self.tar_hum_plan_temp,
            ],
            input_variables=["target_plan"],
        )

        # prepare llmchain
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        result = chain.generate(
            [
                {
                    "target_plan": target_plan,
                }
                for target_plan in inputs["target_plan"]
            ]
        ).generations

        return {"target_back_prompt": [r[0].text for r in result]}
