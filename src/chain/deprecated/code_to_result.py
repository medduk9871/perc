from typing import Dict, List, Optional

from evaluate import load
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class Code2ResultChain(Chain):
    verbose: bool = False

    @property
    def input_keys(self) -> List[str]:
        return ["target_code", "target_test"]

    @property
    def output_keys(self) -> List[str]:
        return ["target_result"]

    def _call(
        self,
        inputs: Dict[str, List[str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        code_eval = load("code_eval")
        _, results = code_eval.compute(
            references=inputs["target_test"],
            predictions=inputs["target_code"],
        )

        result = list(map(lambda x: x[1]["result"], results[0]))

        return {"target_result": result}
