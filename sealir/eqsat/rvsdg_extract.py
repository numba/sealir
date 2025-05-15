from __future__ import annotations

import json
import math
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import starmap
from pprint import pformat
from typing import Any, Callable, NamedTuple, Sequence

import networkx as nx
from egglog import EGraph

from .egraph_utils import EGraphJsonDict
from .rvsdg_extract_details import EGraphToRVSDG

MAX_COST = float("inf")


@dataclass(frozen=True)
class CostFunc:
    equation: Callable
    constants: dict[str, str]

    def compute(self, *child_costs: float, **constants: float):
        return self.equation(*child_costs, **constants)


class CostModel:
    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ) -> CostFunc:
        return self.get_scaled(cost, multipliers=tuple([1.0] * len(children)))

    def get_simple(self, self_cost: float) -> CostFunc:
        """Get a simple cost function suitable for arithmetic expressions.

        This produce a more intuitive cost. Cost of children is not used.
        """
        return self.get_equation(lambda *args: self_cost, {})

    def get_scaled(
        self, self_cost: float, multipliers: Sequence[float]
    ) -> CostFunc:
        """Get a cost function that scales the cost of its children"""
        multipliers = tuple(multipliers)
        return self.get_equation(
            lambda *args: self_cost
            + sum(starmap(operator.mul, zip(args, multipliers, strict=True))),
            {},
        )

    def get_equation(
        self, equ: Callable, constants: dict[str, str]
    ) -> CostFunc:
        """Get a custom cost function.

        Parameters:

        equ: must accept `(*args, **kwargs)`, where the `*args` are the
             children costs and `**kwargs` are the constants.

        constants: is a mapping from keywords for `equ` to the child eclass.
        """
        return CostFunc(equ, constants)

    def eval_constant_node(
        self, nodename: str, op: str, ty: str
    ) -> int | float | None:
        if nodename.startswith("primitive-i64-"):
            return int(op)
        elif nodename.startswith("primitive-f64"):
            return float(op)
        return None


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


def render_extraction_graph(G: nx.MultiDiGraph, filename: str):
    """Render extraction-graph to SVG."""
    nx.drawing.nx_pydot.to_pydot(G).write_svg(filename + ".svg")


class Extraction:
    nodes: dict[str, Node]
    node_types: dict[str, str]
    root_eclass: str
    cost_model: CostModel

    _DEBUG = False

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
        self, max_iter=1000, max_no_progress=100
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

        # Create an eclass dependency graph
        G = nx.DiGraph()
        for eclass, enodes in eclassmap.items():
            G.add_node(eclass, shape="diamond")
            for k in enodes:
                enode = nodes[k]
                if not enode.subsumed:
                    G.add_node(k, shape="rect")
                    G.add_edge(eclass, k)
                    for child in enode.children:
                        G.add_edge(k, nodes[child].eclass)

        # Get edge weights multiplier
        edge_weights: dict[tuple[str, str], float]
        edge_weights = defaultdict(lambda: float(1))
        backedges = _find_backedges_dfs_edges(G, source=self.root_eclass)
        for edge in backedges:
            # Backedge is penalized
            edge_weights[edge] = 2

        for u, v in G.edges:
            edgeattrs = G[u][v]
            edgeattrs["cost"] = edgeattrs["label"] = edge_weights[u, v]

        Gdag = G.copy()
        Gdag.remove_edges_from(backedges)
        if self._DEBUG:
            render_extraction_graph(Gdag, "Gdag")

        # Get per-node cost function
        cm = self.cost_model
        nodecostmap: dict[str, CostFunc] = {}
        for k in G.nodes:
            if k not in eclassmap:
                node = nodes[k]
                children_eclasses = [nodes[c].eclass for c in node.children]
                nodecostmap[k] = cm.get_cost_function(
                    nodename=k,
                    op=node.op,
                    ty=self.node_types[k],
                    cost=node.cost,
                    children=children_eclasses,
                )
        if self._DEBUG:
            render_extraction_graph(G, "eclass")

        # Get constant eclasses
        constants = {}
        for eclass, members in eclassmap.items():
            for m in members:
                if (
                    val := cm.eval_constant_node(
                        m, self.nodes[m].op, self.node_types[m]
                    )
                ) is not None:
                    constants[eclass] = val
                    break

        topo_ordered = list(nx.topological_sort(Gdag))

        def propagate_cost():
            dagcost = DagCost(
                Gdag=Gdag,
                eclassmap=eclassmap,
                selections=selections,
                nodecostmap=nodecostmap,
                edge_weights=edge_weights,
                nodes=nodes,
                constants=constants,
            )

            for k in reversed(topo_ordered):
                if k not in eclassmap:
                    node = nodes[k]

                    cost_dag = dagcost.compute_cost(k)
                    cost = sum(cost_dag.values())
                    selections[node.eclass].put(cost, k)

        # Repeatedly compute cost and propagate while keeping the best variant
        # for each eclass. If root score is computed, return early.
        no_progress = NoProgress(max_no_progress)
        for round_i in range(max_iter):
            # propagate
            propagate_cost()
            # root score is computed?
            if math.isfinite(selections[self.root_eclass].best().cost):
                # TODO: implement change rate based termination
                break
            no_progress(selections)

        return selections

    def choose(self) -> tuple[float, nx.MultiDiGraph]:
        selections = self._compute_cost()

        nodes = self.nodes
        chosen_root, rootcost = selections[self.root_eclass].best()

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
                G.add_edge(cur, child_key, label=int(i))
                todolist.append(child_key)

        if self._DEBUG:
            render_extraction_graph(G, "chosen")
        return rootcost, G


def _find_backedges_dfs_edges(G, source) -> list[tuple[str, str]]:
    """
    Find backedges by comparing graph edges to DFS tree edges.

    Parameters:
        G: networkx.DiGraph, the directed graph

    Returns:
        List of tuples (u, v) representing backedges
    """
    # Get DFS tree edges
    dfs_tree_edges = set(nx.dfs_edges(G, source))
    # All graph edges
    all_edges = set(G.edges())
    # Backedges are edges not in the DFS tree but in cycles
    backedges = []
    for edge in all_edges - dfs_tree_edges:
        u, v = edge
        # Check if v is reachable from u in the DFS tree (indicating a cycle)
        if nx.has_path(G, v, u):
            backedges.append(edge)
    return backedges


@dataclass
class _DagCostStats:
    cache_miss: int = 0
    cache_hit: int = 0


@dataclass(frozen=True)
class DagCost:
    """
    Compute cost of a node in both node-based and edge-based cost computation.

    Node-based cost computation deduplicates repeated nodes in subgraphs by
    removing backedges.

    Edge-based cost computation includes self cost, which for simple operations
    is just its computation cost, but for control nodes it includes a factor of
    its children's costs, such as loops that should contain trip-count
    multiplied with the loop body's cost.
    """

    Gdag: nx.DiGraph
    nodes: dict[str, Node]
    eclassmap: dict[str, Any]
    selections: dict[str, Bucket]
    nodecostmap: dict[str, CostFunc]
    edge_weights: dict[tuple[str, str], float]
    "maps eclass to Bucket"
    constants: dict[str, Any]
    "maps eclass to constant value"
    _cache: dict[str, dict[str, float]] = field(default_factory=dict)
    _stats: _DagCostStats = field(default_factory=_DagCostStats)

    def compute_cost(self, nodename: str) -> dict[str, float]:
        if (cc := self._cache.get(nodename)) is None:
            self._stats.cache_miss += 1
            cc = self._cache[nodename] = self._compute_cost(nodename)
        else:
            self._stats.cache_hit += 1
        return cc

    def _compute_cost(self, nodename: str) -> dict[str, float]:
        assert (
            nodename not in self.eclassmap
        ), f"nodename ({nodename!r}) cannot be eclass"
        selections = self.selections
        costs = {}

        ## compute the local node cost
        nodes = self.nodes
        node = nodes[nodename]
        edge_weights = self.edge_weights
        child_edge_weights = [
            edge_weights[nodename, nodes[child].eclass]
            for child in node.children
        ]
        child_costs = []
        for child, edge_weight in zip(
            node.children, child_edge_weights, strict=True
        ):
            child_eclass = nodes[child].eclass
            if choices := selections[child_eclass]:
                cc = choices.best().cost * edge_weight
            else:
                cc = MAX_COST
            child_costs.append(cc)

        cf = self.nodecostmap[nodename]

        # Get constants for the cost function
        constants = {}
        if hasattr(cf, "constants"):
            for k, eclass in cf.constants.items():
                constants[k] = self.constants[eclass]

        local_cost = cf.compute(*child_costs, **constants)
        costs[nodename] = local_cost

        # Stop early if any of the cost is infinity.
        # This will stop cycles to current node.
        if any(map(math.isinf, costs.values())):
            return costs
        # compute children node cost
        for child in node.children:
            costs.update(self._compute_choice(nodes[child].eclass))
        return costs

    def _compute_choice(self, eclass: str) -> dict[str, float]:
        selections = self.selections

        choices = selections[eclass]
        if not choices:
            return {eclass: MAX_COST}
        best = choices.best()

        return self.compute_cost(best.name)


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
        state = {
            k: bkt.best().cost if bkt else MAX_COST
            for k, bkt in selections.items()
        }
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
