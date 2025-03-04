from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI

from src.utils.indent import AIIndMessagePromptTemplate as AIMPT
from src.utils.indent import HumanIndMessagePromptTemplate as HIMPT
from src.utils.indent import SystemIndMessagePromptTemplate as SIMPT

SYS = SIMPT.from_template(
    """
Write a plan, constraints and raw code snippet that solves the given problem."""
)

HUM_PROB = HIMPT.from_template(
    """\
${head}
    ${prompt}"""
)

HUM_PLAN = HIMPT.from_template(
    """\
${head}
    ${prompt}

    # Let's think step by step"""
)

AI_PLAN = AIMPT.from_template(
    """\
${head}
    ${prompt}

    # Let's think step by step
    ${plan}"""
)

HUM_IC = HIMPT.from_template(
    """\
${head}
    ${prompt}

    # Let's write comments as detailed as possible"""
)

AI_IC = AIMPT.from_template(
    """\
${head}
    ${prompt}

    # Let's write comments as detailed as possible
    ${ic}"""
)

HUM_CONST = HIMPT.from_template(
    """\
${head}
    ${prompt}

    # Think about constraints"""
)

AI_CONST = AIMPT.from_template(
    """\
${head}
    ${prompt}

    # Think about constraints
    ${constraint}"""
)

HUM_CODE = HIMPT.from_template(
    """\
${head}
    ${prompt}

    # Let's write code"""
)

AI_CODE = AIMPT.from_template(
    """\
${head}
    ${prompt}

    # Let's write code
    ${code}"""
)


class MainChain(Chain):
    llm_kwargs: Dict[str, Any]
    mode: Optional[List[str]] = ["plan", "ic", "constraint"]
    examples: Optional[List[dict]] = None
    verbose: bool = False

    prompt: str = None

    @property
    def input_keys(self) -> List[str]:
        return ["prompt", "head", "id"]

    @property
    def output_keys(self) -> List[str]:
        return [
            "plan",
            "ic",
            "constraint",
            "reference_constraint",
            "filtered_plan",
            "filtered_ic",
            "filtered_constraint",
            "code",
        ]

    class Config:
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)
        self.cache_dict = data["cache_dict"]
        self.cache_dict_lock = data["cache_dict_lock"]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        if "plan" in self.mode:
            from src.chain.constraint.plan import PlanChain

            plan = PlanChain(
                llm=ChatOpenAI(
                    model=self.llm_kwargs.get("model"),
                    n=self.llm_kwargs.get("n"),
                    max_tokens=self.llm_kwargs.get("max_tokens"),
                    model_kwargs={"top_p": self.llm_kwargs.get("top_p")},
                    temperature=self.llm_kwargs.get("temperature"),
                    max_retries=100,
                ),
                examples=self.examples,
                verbose=self.verbose,
            )(
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "id": inputs["id"],
                }
            )
        else:
            plan = {"plan": ["" for _ in range(self.llm_kwargs.get("n"))]}

        if "ic" in self.mode:
            from src.chain.constraint.ic import ICChain

            ic = ICChain(
                mode=self.mode,
                llm=ChatOpenAI(
                    model=self.llm_kwargs.get("model"),
                    n=self.llm_kwargs.get("n"),
                    max_tokens=self.llm_kwargs.get("max_tokens"),
                    model_kwargs={"top_p": self.llm_kwargs.get("top_p")},
                    temperature=self.llm_kwargs.get("temperature"),
                    max_retries=100,
                ),
                examples=self.examples,
                verbose=self.verbose,
            )(
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "id": inputs["id"],
                    "plan": plan["plan"],
                }
            )
        else:
            ic = {"ic": ["" for _ in range(self.llm_kwargs.get("n"))]}

        if "constraint" in self.mode:
            from src.chain.constraint.constraint import ConstraintChain

            constraint = ConstraintChain(
                mode=self.mode,
                llm=ChatOpenAI(
                    model=self.llm_kwargs.get("model"),
                    n=1,
                    max_tokens=self.llm_kwargs.get("max_tokens"),
                    model_kwargs={"top_p": self.llm_kwargs.get("top_p")},
                    temperature=self.llm_kwargs.get("temperature"),
                    max_retries=100,
                ),
                examples=self.examples,
                verbose=self.verbose,
            )(
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "id": inputs["id"],
                    "plan": plan["plan"],
                    "ic": ic["ic"],
                }
            )
        else:
            constraint = {"constraint": ["" for _ in range(self.llm_kwargs.get("n"))]}

        if "filter" in self.mode:
            from src.chain.constraint.filter import FilterChain

            filtered = FilterChain(
                mode=self.mode,
                llm_kwargs=self.llm_kwargs,
                examples=self.examples,
                cache_dict=self.cache_dict,
                cache_dict_lock=self.cache_dict_lock,
                verbose=self.verbose,
            )(
                {
                    "prompt": inputs["prompt"],
                    "head": inputs["head"],
                    "id": inputs["id"],
                    "plan": [plan["plan"]],
                    "ic": [ic["ic"]],
                    "constraint": [constraint["constraint"]],
                }
            )
        else:
            filtered = {
                "reference_constraint": None,
                "filtered_plan": plan["plan"],
                "filtered_ic": ic["ic"],
                "filtered_constraint": constraint["constraint"],
            }

        from src.chain.constraint.code import CodeChain

        code = CodeChain(
            mode=self.mode,
            llm=ChatOpenAI(
                model=self.llm_kwargs.get("model"),
                n=1,
                max_tokens=self.llm_kwargs.get("max_tokens"),
                model_kwargs={"top_p": self.llm_kwargs.get("top_p")},
                temperature=self.llm_kwargs.get("temperature"),
                max_retries=100,
            ),
            examples=self.examples,
            verbose=self.verbose,
        )(
            {
                "prompt": inputs["prompt"],
                "head": inputs["head"],
                "id": inputs["id"],
                "plan": filtered["filtered_plan"],
                "ic": filtered["filtered_ic"],
                "constraint": filtered["filtered_constraint"],
            }
        )

        return {
            "plan": plan["plan"],
            "ic": ic["ic"],
            "constraint": constraint["constraint"],
            "reference_constraint": filtered["reference_constraint"],
            "filtered_plan": filtered["filtered_plan"],
            "filtered_ic": filtered["filtered_ic"],
            "filtered_constraint": filtered["filtered_constraint"],
            "code": code["code"],
        }
