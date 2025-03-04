from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from src.chain.reflexion.code import ReflexionCodeChain

from src.chain.reflexion.feedback import FeedbackChain
from src.chain.reflexion.reflection import ReflectionChain
from src.chain.reflexion.templates.humaneval import CODE, CURRICULUM_TESTCASE, TESTCASE


class ReflexionChain(Chain):
    chain_model: str
    examples: Optional[Dict[str, List[Dict[str, Any]]]] = None
    max_iters: int
    llm_kwargs: Dict[str, Dict[str, Any]] = {}
    dataset_type: str = "openai_humaneval"
    verbose: bool = False

    feedback_chain: FeedbackChain = None
    reflection_chain: ReflectionChain = None
    code_chain: ReflexionCodeChain = None

    @property
    def input_keys(self) -> List[str]:
        return ["gen_tc", "code", "gen_tc_parsed", "code_parsed"]

    @property
    def output_keys(self) -> List[str]:
        return ["codes", "gen_tc_results", "feedbacks", "reflections", "code"]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.feedback_chain = FeedbackChain(
            dataset_type=self.dataset_type,
            chain_model=self.chain_model,
            verbose=self.verbose,
        )
        self.reflection_chain = ReflectionChain(
            dataset_type=self.dataset_type,
            llm_kwargs=self.llm_kwargs["reflection"],
            examples=self.examples["reflection"],
            verbose=self.verbose,
        )
        self.code_chain = ReflexionCodeChain(
            dataset_type=self.dataset_type,
            llm_kwargs=self.llm_kwargs["reflexion_code"],
            examples=self.examples["reflexion_code"],
            verbose=self.verbose,
        )

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        inputs["codes"] = [inputs["code"]]
        inputs["gen_tc_results"] = []
        inputs["feedbacks"] = []
        inputs["reflections"] = []

        for _ in range(self.max_iters):
            feedback = self.feedback_chain(
                {
                    "reference": inputs["gen_tc_parsed"],
                    "candidate": inputs["code_parsed"],
                }
            )
            inputs["gen_tc_results"].append(feedback["gen_tc_result"])
            inputs["feedbacks"].append(feedback["feedback"])
            inputs["feedback"] = feedback["feedback"]
            if feedback["passed"]:
                break

            reflection = self.reflection_chain(inputs)
            inputs["reflections"].append(reflection["reflection"])
            inputs["reflection"] = reflection["reflection"]

            code = self.code_chain(inputs)
            inputs["codes"].append(code["code"])
            inputs["code"] = code["code"]

        return inputs


class CurriculumReflexionChain(Chain):
    chain_model: str
    examples: Optional[Dict[str, List[Dict[str, Any]]]] = None
    max_iters: int
    llm_kwargs: Dict[str, Dict[str, Any]] = {}
    dataset_type: str = "openai_humaneval"
    verbose: bool = False

    feedback_chain: FeedbackChain = None
    reflection_chain: ReflectionChain = None
    code_chain: ReflexionCodeChain = None

    @property
    def input_keys(self) -> List[str]:
        return ["gen_tc", "code"]

    @property
    def output_keys(self) -> List[str]:
        return ["codes", "gen_tc_results", "feedbacks", "reflections", "code"]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.feedback_chain = FeedbackChain(
            dataset_type=self.dataset_type,
            chain_model=self.chain_model,
            verbose=self.verbose,
        )
        self.reflection_chain = ReflectionChain(
            dataset_type=self.dataset_type,
            llm_kwargs=self.llm_kwargs["reflection"],
            examples=self.examples["reflection"],
            verbose=self.verbose,
        )
        self.code_chain = ReflexionCodeChain(
            dataset_type=self.dataset_type,
            llm_kwargs=self.llm_kwargs["reflexion_code"],
            examples=self.examples["reflexion_code"],
            verbose=self.verbose,
        )

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        inputs["codes"] = [inputs["code"]]
        inputs["gen_tc_results"] = []
        inputs["feedbacks"] = []
        inputs["reflections"] = []

        curriculum_step = 1
        curriculum = ["invalid", "general", "edge"]
        for _ in range(self.max_iters):
            feedback = self.feedback_chain(
                {
                    "reference": CURRICULUM_TESTCASE.parse(
                        inputs["gen_tc"],
                        curriculum=curriculum[:curriculum_step],
                    ),
                    "candidate": inputs["code_parsed"],
                }
            )
            inputs["gen_tc_results"].append(feedback["gen_tc_result"])
            inputs["feedbacks"].append(feedback["feedback"])
            inputs["feedback"] = feedback["feedback"]

            if len(feedback["gen_tc_result"]) == 0:
                if curriculum_step >= len(curriculum):
                    break
                else:
                    curriculum_step += 1
                    continue
            elif (
                sum([r == "passed" for r in feedback["gen_tc_result"]])
                / len(feedback["gen_tc_result"])
                > 0.8
            ):
                if curriculum_step >= len(curriculum):
                    if feedback["passed"]:
                        break
                else:
                    curriculum_step += 1

            reflection = self.reflection_chain(inputs)
            inputs["reflections"].append(reflection["reflection"])
            inputs["reflection"] = reflection["reflection"]

            code = self.code_chain(inputs)
            inputs["codes"].append(code["code"])
            inputs["code"] = code["code"]

        return inputs
