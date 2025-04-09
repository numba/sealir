from __future__ import annotations

import heapq
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pprint import pformat, pprint

import networkx as nx
from egglog import EGraph

from .egraph_utils import EGraphJsonDict
from .rvsdg_extract_details import EGraphToRVSDG

MAX_COST = float("inf")


class CostModel:
    def get_cost_function(
        self,
        nodename: str,
        op: str,
        cost: float,
        nodes: dict[str, Node],
        child_costs: list[float],
    ) -> float:
        return cost + sum(child_costs)


def egraph_extraction(
    egraph: EGraph,
    rvsdg_sexpr,
    *,
    cost_model=None,
    converter_class=EGraphToRVSDG,
):
    gdct: EGraphJsonDict = json.loads(
        egraph._serialize(
            n_inline_leaves=0, split_primitive_outputs=False
        ).to_json()
    )
    [root] = get_graph_root(gdct)
    root_eclass = gdct["nodes"][root]["eclass"]

    cost_model = CostModel() if cost_model is None else cost_model
    extraction = Extraction(gdct, root_eclass, cost_model)
    cost, exgraph = extraction.choose()

    expr = convert_to_rvsdg(
        exgraph,
        gdct,
        rvsdg_sexpr,
        root,
        egraph,
        converter_class=converter_class,
    )
    return cost, expr


def convert_to_rvsdg(
    exgraph: nx.MultiDiGraph,
    gdct: EGraphJsonDict,
    rvsdg_sexpr,
    root: str,
    egraph: EGraph,
    *,
    converter_class,
):
    # Get declarations so we have named fields
    state = egraph._state
    decls = state.__egg_decls__

    # Do the conversion back into RVSDG
    conversion = converter_class(gdct, rvsdg_sexpr)
    node_iterator = list(nx.dfs_postorder_nodes(exgraph, source=root))

    def egg_fn_to_arg_names(egg_fn: str) -> tuple[str, ...]:
        for ref in state.egg_fn_to_callable_refs[egg_fn]:
            decl = decls.get_callable_decl(ref)
            return tuple(decl.signature.arg_names)
        else:
            raise ValueError(f"missing decl for {egg_fn!r}")

    def iterator(node_iter):
        for node in node_iter:
            children = [child for _, child in exgraph.out_edges(node)]
            # extract argument names
            kind, _, egg_fn = node.split("-")
            match kind:
                case "primitive":
                    pass
                case "function":
                    arg_names = egg_fn_to_arg_names(egg_fn)
                    children = dict(zip(arg_names, children, strict=True))
                case _:
                    raise NotImplementedError(f"kind is {kind!r}")
            yield node, children

    return conversion.run(iterator(node_iterator))


def get_graph_root(graph_json: EGraphJsonDict) -> set[str]:
    # Get GraphRoot
    roots = set()
    for k, v in graph_json["nodes"].items():
        if v["op"] == "GraphRoot":
            roots.add(k)
    return roots


@dataclass(frozen=True)
class Node:
    children: list[str]
    cost: float
    eclass: str
    op: str
    subsumed: bool


class Extraction:
    nodes: dict[str, Node]
    root_eclass: str
    cost_model: CostModel

    def __init__(self, graph_json: EGraphJsonDict, root_eclass, cost_model):
        self.root_eclass = root_eclass
        self.class_data = defaultdict(set)
        self.nodes = {k: Node(**v) for k, v in graph_json["nodes"].items()}
        for k, node in self.nodes.items():
            self.class_data[node.eclass].add(k)
        self.cost_model = cost_model

    def _compute_cost(self, max_iter=10000) -> dict[str, Bucket]:
        """Compute cost for the EGraph.

        Uses dynamic programming with iterative cost propagation

        Performance notes

        Time complexity is O(N * M) where:

        - N is the number of iterations specified by max_iter (default 10000)
        - M is the number of nodes in the graph

        For each iteration, the function:

        - Iterates through all nodes
        - For each node, computes costs based on its children
        - Updates selections dictionary

        The function can terminate early if a finite root score is computed,
        but in worst case it will run for the full N iterations.
        """
        nodes = self.nodes
        eclassmap: dict[str, set[str]] = defaultdict(set)

        for k, node in nodes.items():
            eclassmap[node.eclass].add(k)

        selections: dict[str, Bucket] = defaultdict(Bucket)

        def compute_cost(nodename: str, node: Node):
            if node.subsumed:
                return MAX_COST
            child_costs = []
            for child in node.children:
                child_node = nodes[child]
                choices = selections[child_node.eclass]
                if choices:
                    _, best = choices.best()
                    child_costs.append(best)
                else:
                    child_costs.append(MAX_COST)
            cost = self.cost_model.get_cost_function(
                nodename, node.op, node.cost, nodes, child_costs
            )
            return cost

        # Repeatedly compute cost and propagate while keeping the best variant
        # for each eclass. If root score is computed, return early.
        for round_i in range(max_iter):
            # propagate
            for k, node in nodes.items():
                cost = compute_cost(k, node)
                selections[node.eclass].put(cost, k)
            # root score is computed?
            if math.isfinite(selections[self.root_eclass].best()[1]):
                break

        return selections

    def choose(self) -> tuple[float, nx.MultiDiGraph]:
        selections = self._compute_cost()

        nodes = self.nodes
        chosen_root, rootcost = selections[self.root_eclass].best()
        sentry_cost(rootcost)

        # make selected graph
        G = nx.MultiDiGraph()
        todolist = [chosen_root]
        visited = set()
        while todolist:
            cur = todolist.pop()
            if cur in visited:
                continue
            visited.add(cur)

            for i, u in enumerate(nodes[cur].children):
                child_eclass = nodes[u].eclass
                child_key, cost = selections[child_eclass].best()
                sentry_cost(cost)
                G.add_edge(cur, child_key, label=str(i))
                todolist.append(child_key)

        return rootcost, G


def sentry_cost(cost: float) -> None:
    if math.isinf(cost):
        raise ValueError("invalid cost extracted")


class Bucket:
    def __init__(self):
        self.data = defaultdict(lambda: MAX_COST)

    def put(self, cost: float, key: str) -> None:
        self.data[key] = min(cost, self.data[key])

    def best(self) -> tuple[str, float]:
        return min(self.data.items(), key=lambda kv: kv[1])

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        cls = self.__class__.__name__
        args = pformat(sorted(self.data.items(), key=lambda x: x[1]))
        return f"{cls}({args})"
