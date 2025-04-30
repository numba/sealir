import json
from typing import Dict

from model_explorer import (
    Adapter,
    AdapterMetadata,
    ModelExplorerGraphs,
    graph_builder,
)

from sealir.eqsat.egraph_utils import EGraphJsonDict


class EgraphJsonAdapter(Adapter):

    metadata = AdapterMetadata(
        id="egraph_json_adapter",
        name="EGraph adapter",
        description="Add EGraph support to Model Explorer",
        source_repo="",
        fileExts=["egraph_json"],
    )

    # This is required.
    def __init__(self):
        super().__init__()

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        jsdata: EGraphJsonDict
        with open(model_path, "r") as fin:
            jsdata = json.load(fin)

        graph = graph_builder.Graph(id="egraph")
        # Make nodes
        nodes = {}
        for node_k, node_info in jsdata["nodes"].items():
            nodes[node_k] = node = graph_builder.GraphNode(
                id=node_k,
                label=node_info["op"],
                namespace=node_info["eclass"],
            )
            node.attrs.append(
                graph_builder.KeyValue(
                    key="eclass",
                    value=node_info["eclass"],
                )
            )

        for node_k, node in nodes.items():
            children = jsdata["nodes"][node_k]["children"]
            for child_id, child in enumerate(children):
                nodes[child].incomingEdges.append(
                    graph_builder.IncomingEdge(
                        sourceNodeId=node_k, sourceNodeOutputId=str(child_id)
                    )
                )

        graph.nodes.extend(nodes.values())
        return {"graphs": [graph]}
