from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from src.chain.reflexion.templates.humaneval import (
    FEEDBACK,
    PREV_CODE,
    PROMPT,
    CODE,
    REFLECTION,
)


class InitCodeChain(Chain):
    dataset_type: str = "openai_humaneval"
    llm_kwargs: Dict[str, Any]
    examples: Optional[Dict[str, List[Dict[str, str]]]]
    verbose: bool = False
    parents: List[str] = None

    chain: Chain = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt"]

    @property
    def output_keys(self) -> List[str]:
        return ["code"]

    def __init__(self, **data):
        super().__init__(**data)

        llm = AzureChatOpenAI(
            deployment_name=self.llm_kwargs["model"],
            max_tokens=self.llm_kwargs["max_tokens"],
            temperature=self.llm_kwargs["temperature"],
            n=self.llm_kwargs["n"],
            model_kwargs={"top_p": self.llm_kwargs["top_p"]},
        )

        prompt = ChatPromptTemplate.from_messages([PROMPT.human, CODE.human])

        self.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=self.verbose,
        )

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        result = self.chain(inputs)

        inputs.update({"code": result["text"]})
        inputs.update({"code_parsed": CODE.parse(result["text"])})

        return inputs


class ReflexionCodeChain(Chain):
    dataset_type: str = "openai_humaneval"
    llm_kwargs: Dict[str, Any]
    examples: List[Dict[str, Any]]
    verbose: bool = False
    parents: List[str] = None

    chain: Chain = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt", "code", "feedback", "reflection"]

    @property
    def output_keys(self) -> List[str]:
        return ["code"]

    def __init__(self, **data):
        super().__init__(**data)

        llm = AzureChatOpenAI(
            deployment_name=self.llm_kwargs["model"],
            max_tokens=self.llm_kwargs["max_tokens"],
            temperature=self.llm_kwargs["temperature"],
            n=self.llm_kwargs["n"],
            model_kwargs={"top_p": self.llm_kwargs["top_p"]},
        )

        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages(
                [
                    PROMPT.human,
                    PREV_CODE.human,
                    PREV_CODE.ai,
                    FEEDBACK.human,
                    FEEDBACK.ai,
                    REFLECTION.human,
                    REFLECTION.ai,
                    CODE.human,
                    CODE.ai,
                ]
            ),
            examples=self.examples,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                fewshot_prompt,
                PROMPT.human,
                PREV_CODE.human,
                PREV_CODE.ai,
                FEEDBACK.human,
                FEEDBACK.ai,
                REFLECTION.human,
                REFLECTION.ai,
                CODE.human,
            ]
        )

        self.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=self.verbose,
        )

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        inputs["prev_code"] = inputs["code"]

        result = self.chain(inputs)

        inputs.update({"code": result["text"]})
        inputs.update({"code_parsed": CODE.parse(result["text"])})

        return inputs
