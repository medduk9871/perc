from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
# from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

import src.chain.multi_turn.templates.codecontests as CODECONTESTS_TEMPLATES
import src.chain.multi_turn.templates.humaneval as HUMANEVAL_TEMPLATES
import src.chain.multi_turn.templates.humaneval_cpp as HUMANEVAL_CPP_TEMPLATES
import src.chain.multi_turn.templates.humaneval_java as HUMANEVAL_JAVA_TEMPLATES
import src.chain.multi_turn.templates.humaneval_python as HUMANEVAL_PYTHON_TEMPLATES
import src.chain.multi_turn.templates.humaneval_lua as HUMANEVAL_LUA_TEMPLATES
import src.chain.multi_turn.templates.humaneval_rb as HUMANEVAL_RUBY_TEMPLATES
import src.chain.multi_turn.templates.humaneval_r as HUMANEVAL_R_TEMPLATES
import src.chain.multi_turn.templates.mbpp_cpp as MBPP_CPP_TEMPLATES
import src.chain.multi_turn.templates.mbpp_java as MBPP_JAVA_TEMPLATES
import src.chain.multi_turn.templates.mbpp_pseudo as MBPP_PSEUDO_TEMPLATES


class CodeChain(Chain):
    dataset_type: str # "humaneval" or "codecontests"
    llm_kwargs: Dict[str, Any] = {}
    examples: Optional[Dict[str, List[Dict[str, str]]]] = None
    verbose: bool = False
    parents: List[str]

    prompt: str = None
    templates: Any = None

    @property
    def input_keys(self) -> List[str]:
        keys = ["prompt"]
        full_list = ["draft_plan"]
        # full_list = []
        for k in full_list:
            if k in self.parents:
                keys.append(k)

        if "requirements_selector" in self.parents:
            keys.append("requirements")
        
        return keys

    @property
    def output_keys(self) -> List[str]:
        return ["code"]

    def __init__(self, **data):
        super().__init__(**data)

        if self.dataset_type == "openai_humaneval":
            self.templates = HUMANEVAL_TEMPLATES
        elif self.dataset_type == "deepmind/code_contests":
            self.templates = CODECONTESTS_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,humaneval-cpp":
            self.templates = HUMANEVAL_CPP_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,humaneval-java":
            self.templates = HUMANEVAL_JAVA_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,humaneval-lua":
            self.templates = HUMANEVAL_LUA_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,humaneval-py":
            self.templates = HUMANEVAL_PYTHON_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,humaneval-rb":
            self.templates = HUMANEVAL_RUBY_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,humaneval-r":
            self.templates = HUMANEVAL_R_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,mbpp-cpp":
            self.templates = MBPP_CPP_TEMPLATES
        elif self.dataset_type == "nuprl/MultiPL-E,mbpp-java":
            self.templates = MBPP_JAVA_TEMPLATES
        elif self.dataset_type.startswith("mbpp-pseudo"):
            self.templates = MBPP_PSEUDO_TEMPLATES
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
        if "requirements" in self.parents or "requirements_selector" in self.parents:
            prefix_messages.append(self.templates.REQUIREMENTS.human)
            prefix_messages.append(self.templates.REQUIREMENTS.ai)
        if "final_plan" in self.parents:
            prefix_messages.append(self.templates.FINAL_PLAN.human)
            prefix_messages.append(self.templates.FINAL_PLAN.ai)
        if len(prefix_messages) > 1 and self.dataset_type == "openai_humaneval":
            prefix_messages.append(self.templates.PROMPT.human)
        prefix_messages.append(self.templates.CODE.human)
        example_messages = prefix_messages + [self.templates.CODE.ai]

        example_prompt = ChatPromptTemplate.from_messages(example_messages)
        
        # final_examples = []
        
        # top1_score = float(self.examples[inputs['id']][0]['sim_score'])
       
        # for example in self.examples[inputs['id']]:
        #     cur_sim_score = float(example['sim_score'])
            
        #     if (top1_score - cur_sim_score) < 0.05:
        #         final_examples.append(example)
                
        #     if len(final_examples) >= 3:
        #         break

        # print(inputs['id'], len(final_examples))
        # fewshot_prompt = FewShotChatMessagePromptTemplate(
        #     example_prompt=example_prompt,
        #     examples=final_examples
        # )
        
        if self.llm_kwargs["n"] == 0:
            code = {"code": [inputs['solution']]}
            return code
        
        if "all" in self.examples:
            fewshot_examples = self.examples["all"]
        else:
            fewshot_examples = self.examples[inputs['id']][:3]
        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=fewshot_examples
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [fewshot_prompt] + prefix_messages
        )


        llm = ChatOpenAI(
            model=self.llm_kwargs["model"],
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
            code = {"code": [g[0].text for g in generations]}
        else:
            generations = chain.generate([inputs]).generations[0]
            code = {"code": [g.text for g in generations]}

        return code
