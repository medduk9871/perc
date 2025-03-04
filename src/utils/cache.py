from typing import Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class CacheChain(Chain):
    cache: dict
    chain: Chain
    verbose: bool = False

    @property
    def input_keys(self) -> List[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return self.chain.output_keys

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        if self.hit():
            return {k: self.cache[k] for k in self.chain.output_keys}
        else:
            return self.chain(inputs)

    def hit(self) -> bool:
        for k in self.chain.output_keys:
            if k not in self.cache:
                return False

        return True
    