from typing import Any, Dict, List, Optional

import networkx as nx
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from src.chain.reflexion.code import InitCodeChain
from src.chain.reflexion.reflexion import CurriculumReflexionChain, ReflexionChain
from src.chain.reflexion.testcase import CurriculumTestcaseChain, TestcaseChain

from src.utils.graph import GraphChain


class MainChain(Chain):
    graph: Dict[str, List[str]]
    llm_kwargs: Dict[str, Dict[str, Any]] = {}
    examples: Optional[Dict[str, List[Dict[str, Any]]]] = None
    dataset_type: str = "openai_humaneval"
    num_codes: int
    max_iters: int
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

    def __init__(self, **data):
        super().__init__(**data)

        chain_dic = {}

        if "testcase" in self.graph:
            chain_dic["testcase"] = TestcaseChain(
                dataset_type=self.dataset_type,
                llm_kwargs=self.llm_kwargs["testcase"],
                examples=self.examples["testcase"],
                verbose=self.verbose,
                parents=self.graph["testcase"],
            )

        if "init_code" in self.graph:
            chain_dic["init_code"] = InitCodeChain(
                dataset_type=self.dataset_type,
                llm_kwargs=self.llm_kwargs["init_code"],
                verbose=self.verbose,
                parents=self.graph["init_code"],
            )

        if "reflexion" in self.graph:
            chain_dic["reflexion"] = ReflexionChain(
                chain_model="reflexion",
                max_iters=self.max_iters,
                llm_kwargs=self.llm_kwargs,
                examples=self.examples,
                dataset_type=self.dataset_type,
                verbose=self.verbose,
            )

        if "curriculum_testcase" in self.graph:
            chain_dic["curriculum_testcase"] = CurriculumTestcaseChain(
                dataset_type=self.dataset_type,
                llm_kwargs=self.llm_kwargs["curriculum_testcase"],
                examples=self.examples["curriculum_testcase"],
                verbose=self.verbose,
                parents=self.graph["curriculum_testcase"],
            )
        
        if "curriculum_reflexion" in self.graph:
            chain_dic["curriculum_reflexion"] = CurriculumReflexionChain(
                chain_model="curriculum_reflexion",
                max_iters=self.max_iters,
                llm_kwargs=self.llm_kwargs,
                examples=self.examples,
                dataset_type=self.dataset_type,
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
        result_d = {}
        for _ in range(self.num_codes):
            result = self.chain(inputs)
            for key, value in result.items():
                if key in self.output_keys:
                    if key not in result_d:
                        result_d[key] = []
                    result_d[key].append(value)

        return result_d
