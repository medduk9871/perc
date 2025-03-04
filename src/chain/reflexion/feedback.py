import sys
import ast
import astunparse
from typing import Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from src.chain.reflexion.templates.humaneval import CODE, TESTCASE


from src.evaluator import Evaluator


class FeedbackChain(Chain):
    dataset_type: str = "openai_humaneval"
    chain_model: str = "reflexion"
    verbose: bool = False

    evaluator: Evaluator = None

    @property
    def input_keys(self) -> List[str]:
        return ["candidate", "reference"]

    @property
    def output_keys(self) -> List[str]:
        return ["gen_tc_result", "feedback", "passed"]

    def __init__(self, **data):
        super().__init__(**data)

        self.evaluator = Evaluator(disable_tqdm=True)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        candidate = inputs["candidate"]
        reference = inputs["reference"]

        passed, result = self.evaluator.run_one(
            reference=reference,
            candidate=candidate,
            num_workers=1,
        )

        passed_tc = []
        failed_tc_result = []
        for i, r in enumerate(result):
            if r == "passed":
                passed_tc.append(reference[i])
            else:
                if r.strip() == "failed:":
                    try:
                        output = get_output(candidate, reference[i])
                        result[i] = f"failed: returned {output}"
                    except ValueError as e:
                        if e.args[0].startswith("Exceeds the limit"):
                            result[i] = f"failed"
                        else:
                            raise e

                comment = f"# {result[i] if len(result[i]) < 150 else result[i][:150] + '...'}"
                text = f"{reference[i]} {comment}"
                failed_tc_result.append(text)

        feedback = "Tests passing:\n"
        feedback += "\n".join(passed_tc)
        feedback += "\nTests failing:\n"
        feedback += "\n".join(failed_tc_result)

        return {
            "gen_tc_result": result,
            "feedback": feedback,
            "passed": passed,
        }


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left  # type: ignore
    except:
        call_str = ast_parsed.body[0].test  # type: ignore

    return astunparse.unparse(call_str).strip()


def get_output(func: str, assert_statement: str) -> str:
    try:
        exec(func, globals())
        func_call = get_call_str(assert_statement)
        output = eval(func_call, globals())
        return output
    except Exception as e:
        return str(e)
