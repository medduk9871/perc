from typing import Any, Dict, List, Optional

import networkx as nx
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain

from src.chain.multi_turn.code import CodeChain
from src.chain.multi_turn.plan import DraftPlanChain, FinalPlanChain
from src.chain.multi_turn.requirements import RequirementsChain, RequirementsSelectorChain
from src.chain.multi_turn.testcase import TestcaseChain, TestcaseSelectorChain
from src.utils.cache import CacheChain
from src.utils.graph import GraphChain


class MainChain(Chain):
    graph: Dict[str, List[str]] = {}
    llm_kwargs: Dict[str, Dict[str, Any]] = {}
    examples: Dict[str, Dict[str, List[Dict[str, str]]]] = None
    parts: Dict[str, List[str]] = {}
    dataset_type: str = "openai_humaneval"
    verbose: bool = False

    cache: dict = {}
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
        with open('./test_data.json', 'w') as f:
            import json
            json.dump(kwargs, f, indent=4)
        super().__init__(**kwargs)

        chain_dic = {}
        if "draft_plan" in self.graph:
            chain = DraftPlanChain(
                dataset_type=self.dataset_type,
                llm_kwargs=self.llm_kwargs["draft_plan"],
                examples=self.examples["draft_plan"],
                verbose=self.verbose,
                parents=self.graph["draft_plan"],
            )
            chain_dic["draft_plan"] = CacheChain(
                cache=self.cache,
                chain=chain,
                verbose=self.verbose,
            )

        if "requirements" in self.graph:
            chain = RequirementsChain(
                dataset_type=self.dataset_type,
                llm_kwargs=self.llm_kwargs["requirements"],
                examples=self.examples["requirements"],
                verbose=self.verbose,
                parents=self.graph["requirements"],
            )
            chain_dic["requirements"] = CacheChain(
                cache=self.cache,
                chain=chain,
                verbose=self.verbose,
            )

        if "requirements_selector" in self.graph:
            chain_dic["requirements_selector"] = RequirementsSelectorChain(
                dataset_type=self.dataset_type,
                parts=self.parts.get("requirements", None),
                verbose=self.verbose,
            )

        if "final_plan" in self.graph:
            chain = FinalPlanChain(
                dataset_type=self.dataset_type,
                llm_kwargs=self.llm_kwargs["final_plan"],
                examples=self.examples["final_plan"],
                verbose=self.verbose,
                parents=self.graph["final_plan"],
            )
            chain_dic["final_plan"] = CacheChain(
                cache=self.cache,
                chain=chain,
                verbose=self.verbose,
            )

        if "code" in self.graph:
            chain = CodeChain(
                dataset_type=self.dataset_type,
                llm_kwargs=self.llm_kwargs["code"],
                examples=self.examples["code"],
                verbose=self.verbose,
                parents=self.graph["code"],
            )
            chain_dic["code"] = CacheChain(
                cache=self.cache,
                chain=chain,
                verbose=self.verbose,
            )

        if "testcase" in self.graph:
            chain = TestcaseChain(
                dataset_type=self.dataset_type,
                llm_kwargs=self.llm_kwargs["testcase"],
                examples=self.examples["testcase"],
                verbose=self.verbose,
                parents=self.graph["testcase"],
            )
            chain_dic["testcase"] = CacheChain(
                cache=self.cache,
                chain=chain,
                verbose=self.verbose,
            )

        if "testcase_selector" in self.graph:
            chain_dic["testcase_selector"] = TestcaseSelectorChain(
                dataset_type=self.dataset_type,
                parts=self.parts.get("testcase", None),
                verbose=self.verbose,
            )

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
