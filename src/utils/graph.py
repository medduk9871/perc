from typing import Any, Dict, List, Optional

import networkx as nx
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class GraphChain(Chain):
    graph: nx.DiGraph  # todo: test the graph is a dag

    @property
    def input_keys(self) -> List[str]:
        # find input nodes
        nodes = {n for n, d in self.graph.in_degree() if d == 0}

        # return input keys of input nodes
        return {k for n in nodes for k in self.graph.nodes[n]["chain"].input_keys}

    @property
    def output_keys(self) -> List[str]:
        # find nodes except input nodes
        nodes = {n for n, d in self.graph.in_degree() if d != 0}

        # return output keys of output nodes
        return {k for n in nodes for k in self.graph.nodes[n]["chain"].output_keys}

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        # traverse graph in bfs
        outputs = inputs
        for node in nx.topological_sort(self.graph):
            chain = self.graph.nodes[node]["chain"]
            new_outputs = chain(outputs)
            outputs.update(new_outputs)

        return outputs
