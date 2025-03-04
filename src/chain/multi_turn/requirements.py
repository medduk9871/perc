from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.chain.multi_turn.selector import SelectorChain

import src.chain.multi_turn.templates.codecontests as CODECONTESTS_TEMPLATES
import src.chain.multi_turn.templates.humaneval as HUMANEVAL_TEMPLATES
import src.chain.multi_turn.templates.humaneval_cpp as HUMANEVAL_CPP_TEMPLATES


class RequirementsChain(Chain):
    dataset_type: str
    llm_kwargs: Dict[str, Any]
    examples: Optional[Dict[str, List[Dict[str, str]]]] = None
    verbose: bool = False
    parents: List[str]

    prompt: str = None
    templates: Any = None

    @property
    def input_keys(self) -> List[str]:
        keys = ["prompt"]
        full_list = ["draft_plan"]
        for k in full_list:
            if k in self.parents:
                keys.append(k)
        return keys

    @property
    def output_keys(self) -> List[str]:
        return ["requirements"]

    def __init__(self, **data):
        super().__init__(**data)

        if self.dataset_type == "openai_humaneval":
            self.templates = HUMANEVAL_TEMPLATES
        elif self.dataset_type == "deepmind/code_contests":
            self.templates = CODECONTESTS_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,humaneval-cpp":
            self.templates = HUMANEVAL_CPP_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,humaneval-java":
            self.templates = HUMANEVAL_CPP_TEMPLATES
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        prefix_messages = [self.templates.PROMPT.human]
        if "draft_plan" in self.parents:
            prefix_messages.append(self.templates.DRAFT_PLAN.human)
            prefix_messages.append(self.templates.DRAFT_PLAN.ai)
        if len(prefix_messages) > 1 and self.dataset_type == "openai_humaneval":
            prefix_messages.append(self.templates.PROMPT.human)
        prefix_messages.append(self.templates.REQUIREMENTS.human)
        example_messages = prefix_messages + [self.templates.REQUIREMENTS.ai]

        example_prompt = ChatPromptTemplate.from_messages(example_messages)
        
        # final_examples = []
        
        # top1_score = float(self.examples[inputs['id']][0]['sim_score'])
       
        # for example in self.examples[inputs['id']]:
        #     cur_sim_score = float(example['sim_score'])
            
        #     if (top1_score - cur_sim_score) <= 0.4 and cur_sim_score > 0.3:
        #         final_examples.append(example)
                
        #     if len(final_examples) >= 8:
        #         break

        # fewshot_prompt = FewShotChatMessagePromptTemplate(
        #     example_prompt=example_prompt,
        #     examples=final_examples
        # )
        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples[inputs['id']][0:3]
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [fewshot_prompt] + prefix_messages
        )

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
            requirements = {"requirements": [g[0].text for g in generations]}
        else:
            generations = chain.generate([inputs]).generations[0]
            requirements = {"requirements": [g.text for g in generations]}

        return requirements


class RequirementsSelectorChain(SelectorChain):
    key: str = "requirements"
    dataset_type: str

    def __init__(self, **data):
        super().__init__(**data)

        if self.dataset_type == "openai_humaneval":
            self.template = HUMANEVAL_TEMPLATES.REQUIREMENTS
        elif self.dataset_type == "deepmind/code_contests":
            self.template = CODECONTESTS_TEMPLATES.REQUIREMENTS
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
