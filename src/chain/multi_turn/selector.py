from copy import deepcopy
from typing import Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class SelectorChain(Chain):
    key: str
    parts: Optional[List[str]]
    template: object = None
    verbose: bool = False

    @property
    def input_keys(self) -> List[str]:
        return [self.key]

    @property
    def output_keys(self) -> List[str]:
        return [self.key, "original_" + self.key, "parsed_" + self.key]

    def __init__(self, **data):
        super().__init__(**data)

    def _call(
        self,
        inputs: Dict[str, List[str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        rc = inputs[self.key]
        original_rc = deepcopy(rc)

        if isinstance(rc, str):
            parsed_rc = [self.template.parse(rc)]
        elif isinstance(rc, list):
            parsed_rc = [self.template.parse(r) for r in rc]
        else:
            raise TypeError("Input type must be str or list[str]")

        if self.parts is None:
            return {
                self.key: rc,
                "original_" + self.key: original_rc,
                "parsed_" + self.key: parsed_rc,
            }

        if isinstance(parsed_rc, str):
            rc = [self._select(parsed_rc)]
        elif isinstance(parsed_rc, list):
            rc = [self._select(r) for r in parsed_rc]
        else:
            raise TypeError("Input type must be str or list[str]")

        return {
            self.key: rc,
            "original_" + self.key: original_rc,
            "parsed_" + self.key: parsed_rc,
        }

    def _select(self, parsed_text: Dict[str, List[str]]) -> str:
        text = ""
        for part in self.parts:
            text += "\n".join(parsed_text[part]) + "\n"

        return text
