import pickle
from typing import Dict

from model_explorer import (
    Adapter,
    AdapterMetadata,
    ModelExplorerGraphs,
    graph_builder,
)

from sealir import ase, grammar, rvsdg


class RvsdgAdapter(Adapter):

    metadata = AdapterMetadata(
        id="rvsdg_adapter",
        name="RVSDG adapter",
        description="Add RVSDG support to Model Explorer",
        source_repo="",
        fileExts=["rvsdg"],
    )

    # This is required.
    def __init__(self):
        super().__init__()

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        with open(model_path, "rb") as fin:
            grm, tape, handle = pickle.load(fin)

        root = grm.downcast(ase.BasicSExpr(tape, handle))
        graph = graph_builder.Graph(id="egraph")

        # assign RVSDG region to namespaces by nesting them
        ns_map = self.map_regions_to_namespace(root)

        nodes = {}

        def format_tuple(vs: tuple, start: int):
            buf = []
            for i, v in enumerate(vs):
                text = (
                    f"{i}=[{start + i}]"
                    if isinstance(v, ase.SExpr)
                    else f"{i}={v!r}"
                )
                buf.append(text)
            return ", ".join(buf)

        def format_node_label(expr: ase.SExpr):
            if isinstance(expr, grammar.NamedSExpr):
                argnames = expr._slots
                args = []
                for i, k in enumerate(argnames):
                    v = getattr(expr, k)
                    if isinstance(v, ase.SExpr):
                        args.append(f"{k}=[{i}]")
                    elif isinstance(v, tuple):
                        args.append(f"{k}=({format_tuple(v, i)})")
                    else:
                        args.append(f"{k}={v!r}")

            else:
                args = [
                    f"[{i}]" if isinstance(arg, ase.SExpr) else repr(arg)
                    for i, arg in enumerate(expr._args)
                ]

            return f"{expr._head}({', '.join(args)})"

        class Grapher(ase.TreeVisitor):
            def visit(self, expr: ase.SExpr):
                node = graph_builder.GraphNode(
                    id=str(expr._handle),
                    label=format_node_label(expr),
                    namespace=ns_map[expr],
                )
                nodes[expr] = node

        ase.apply_bottomup(root, Grapher(), reachable="compute")

        for expr, node in nodes.items():
            for i, arg in enumerate(expr._args):
                if isinstance(arg, ase.SExpr):
                    node.incomingEdges.append(
                        graph_builder.IncomingEdge(
                            sourceNodeId=str(arg._handle),
                            targetNodeInputId=str(i),
                        )
                    )

        graph.nodes.extend(nodes.values())
        return {"graphs": [graph]}

    def map_regions_to_namespace(
        self, root: ase.SExpr
    ) -> dict[ase.SExpr, str]:

        memo = {}

        is_region = lambda x: isinstance(x, rvsdg.grammar.RegionEnd)
        for parents, node in ase.walk_descendants_depth_first_no_repeat(root):
            if not ase.is_metadata(node):
                regions = list(filter(is_region, parents))
                if is_region(node):
                    regions.append(node)
                ns = ["region_" + str(reg.begin._handle) for reg in regions]
                memo[node] = "/".join(ns)
        return memo
