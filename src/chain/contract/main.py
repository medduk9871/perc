from typing import Any, Dict, List, Optional

import networkx as nx
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain

from src.chain.contract.code import CodeChain
from src.chain.contract.contract import ContractChain
from src.chain.contract.plan import PlanChain
from src.chain.contract.testcase import TestcaseChain
from src.utils.graph import GraphChain


class MainChain(Chain):
    graph: Dict[str, List[str]] = {}
    llm_kwargs: Dict[str, Dict[str, Any]] = {}
    examples: Optional[Dict[str, List[Dict[str, str]]]] = None
    dataset_type: str = "openai_humaneval"
    verbose: bool = False

    chain: GraphChain = None

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

        plan_chain = PlanChain(
            dataset_type=self.dataset_type,
            llm_kwargs=self.llm_kwargs["plan"],
            examples=self.examples["plan"],
            verbose=self.verbose,
            parents=self.graph["plan"]
        )

        contract_chain = ContractChain(
            dataset_type=self.dataset_type,
            llm_kwargs=self.llm_kwargs["contract"],
            examples=self.examples["contract"],
            verbose=self.verbose,
            parents=self.graph["contract"]
        )

        code_chain = CodeChain(
            dataset_type=self.dataset_type,
            llm_kwargs=self.llm_kwargs["code"],
            examples=self.examples["code"],
            verbose=self.verbose,
            parents=self.graph["code"]
        )

        testcase_chain = TestcaseChain(
            dataset_type=self.dataset_type,
            llm_kwargs=self.llm_kwargs["testcase"],
            examples=self.examples["testcase"],
            verbose=self.verbose,
            parents=self.graph["testcase"]
        )

        chain_dic = {
            "plan": plan_chain,
            "contract": contract_chain,
            "code": code_chain,
            "testcase": testcase_chain,
        }

        graph = nx.DiGraph()
        for parent in self.graph.keys():
            graph.add_node(parent, chain=chain_dic[parent])
        for parent, children in self.graph.items():
            for child in children:
                graph.add_edge(child, parent)

        self.chain = GraphChain(graph=graph)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        return self.chain(inputs)
