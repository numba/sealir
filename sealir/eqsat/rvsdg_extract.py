from __future__ import annotations

import heapq
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pprint import pformat, pprint
from typing import NamedTuple

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
        ty: str,
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
    node_iterator = list(nx.dfs_postorder_nodes(exgraph, source=root))

    def egg_fn_to_arg_names(egg_fn: str) -> tuple[str, ...]:
        for ref in state.egg_fn_to_callable_refs[egg_fn]:
            decl = decls.get_callable_decl(ref)
            return tuple(decl.signature.arg_names)
        else:
            raise ValueError(f"missing decl for {egg_fn!r}")

    def iterator(node_iter):
        for node in node_iter:
            # Get children nodes in order
            children = [
                (data["label"], child)
                for _, child, data in exgraph.out_edges(node, data=True)
            ]
            if children:
                children.sort()
                _, children = zip(*children)
            # extract argument names
            kind, _, egg_fn = node.split("-")
            match kind:
                case "primitive":
                    pass
                case "function":
                    # TODO: put this into the converter_class
                    arg_names = egg_fn_to_arg_names(egg_fn)
                    children = dict(zip(arg_names, children, strict=True))
                case _:
                    raise NotImplementedError(f"kind is {kind!r}")
            yield node, children

    conversion = converter_class(gdct, rvsdg_sexpr, egg_fn_to_arg_names)
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
    node_types: dict[str, str]
    root_eclass: str
    cost_model: CostModel

    def __init__(self, graph_json: EGraphJsonDict, root_eclass, cost_model):
        self.root_eclass = root_eclass
        self.class_data = defaultdict(set)
        self.nodes = {k: Node(**v) for k, v in graph_json["nodes"].items()}
        self.node_types = {
            k: graph_json["class_data"][node.eclass]["type"]
            for k, node in self.nodes.items()
        }
        for k, node in self.nodes.items():
            self.class_data[node.eclass].add(k)
        self.cost_model = cost_model

    def _compute_cost(
        self, max_iter=10000, max_no_progress=500
    ) -> dict[str, Bucket]:
        """
        Uses dynamic programming with iterative cost propagation

        Args:
            max_iter (int, optional): Maximum number of iterations to compute
            costs. Defaults to 10000. max_no_progress (int, optional): Maximum
            iterations without cost improvement. Defaults to 500.

        Returns:
            dict[str, Bucket]: A mapping of equivalence classes to their lowest
            cost representations.


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
            ty = self.node_types[nodename]
            cost = self.cost_model.get_cost_function(
                nodename, node.op, ty, node.cost, nodes, child_costs
            )
            return cost

        # Repeatedly compute cost and propagate while keeping the best variant
        # for each eclass. If root score is computed, return early.
        no_progress = NoProgress(max_no_progress)
        for round_i in range(max_iter):
            # propagate
            for k, node in nodes.items():
                cost = compute_cost(k, node)
                selections[node.eclass].put(cost, k)
            # root score is computed?
            if math.isfinite(selections[self.root_eclass].best()[1]):
                break
            no_progress(selections)

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
                G.add_edge(cur, child_key, label=int(i))
                todolist.append(child_key)

        return rootcost, G


class ExtractionError(Exception):
    pass


@dataclass
class NoProgress:
    max_no_progress: int
    "Maximum iteration without progress"
    _last_iteration: int = 0
    _last_captured: dict[str, float] = field(default_factory=dict)

    def __call__(self, selections: dict[str, Bucket]) -> None:
        """
        Check if no progress has been made in the current iteration.

        Tracks iterations and compares the current state of selections with the
        last captured state. Raises ExtractionError if no improvement is
        detected, indicating a potential stagnation.

        Args:
            selections (dict[str, Bucket]): A dictionary of buckets
            representing different selections.

        """
        it = self._last_iteration
        self._last_iteration += 1
        if it % self.max_no_progress != 0:
            # skipped
            return
        state = {k: bkt.best().cost for k, bkt in selections.items()}
        if self._last_captured:
            if state == self._last_captured:
                raise ExtractionError(
                    "extraction stopped due to lack of progress for "
                    f"{self.max_no_progress} iterations"
                )
        self._last_captured = state


class _NameAndCostTuple(NamedTuple):
    name: str
    cost: float


def sentry_cost(cost: float) -> None:
    if math.isinf(cost):
        raise ValueError("invalid cost extracted")


class Bucket:
    def __init__(self):
        self.data = defaultdict(lambda: MAX_COST)

    def put(self, cost: float, key: str) -> None:
        self.data[key] = min(cost, self.data[key])

    def best(self) -> _NameAndCostTuple:
        best = min(self.data.items(), key=lambda kv: kv[1])
        return _NameAndCostTuple(*best)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        cls = self.__class__.__name__
        args = pformat(sorted(self.data.items(), key=lambda x: x[1]))
        return f"{cls}({args})"
