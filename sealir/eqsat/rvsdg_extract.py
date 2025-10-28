from __future__ import annotations

import json
import math
import operator
from collections import defaultdict, deque
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
    stats: dict[str, Any] | None = None,
):
    gdct: EGraphJsonDict = json.loads(
        egraph._serialize(
            n_inline_leaves=0, split_primitive_outputs=False
        ).to_json()
    )
    [root] = get_graph_root(gdct)
    root_eclass = gdct["nodes"][root]["eclass"]

    if stats is not None:
        stats["num_enodes"] = len(gdct["nodes"])
        stats["num_eclasses"] = len(gdct["class_data"])

    cost_model = CostModel() if cost_model is None else cost_model
    extraction = Extraction(gdct, root_eclass, cost_model)
    cost, exgraph = extraction.choose(stats=stats)

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
    common_root = "common_root"

    node_iterator = list(nx.dfs_postorder_nodes(exgraph, source=common_root))

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
            if node == common_root:
                yield node, children
            else:
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
        self.graph_json = graph_json
        self.root_eclass = "common_root"
        self.class_data = defaultdict(set)
        self.nodes = {k: Node(**v) for k, v in graph_json["nodes"].items()}
        self.node_types = {
            k: graph_json["class_data"][node.eclass]["type"]
            for k, node in self.nodes.items()
        }
        for k, node in self.nodes.items():
            self.class_data[node.eclass].add(k)
        self.cost_model = cost_model

    def _create_common_root(self, G: nx.DiGraph, eclassmap: dict[str, set[str]]) -> str:
        """Create a common root node and add it to the graph.

        Args:
            G: The eclass dependency graph
            eclassmap: Mapping from eclass names to sets of node names

        Returns:
            The name of the common root node
        """
        # Get all nodes with in-degree of 0
        common_root = "common_root"

        # Do the conversion back into RVSDG
        root_eclasses = [
            node
            for node, in_degree in G.in_degree()
            if in_degree == 0
            and self.graph_json["class_data"][node]["type"] != "Unit"
        ]
        G.add_node(common_root, shape="rect")
        for n in root_eclasses:
            G.add_edge(common_root, n)

        self.nodes[common_root] = Node(
            children=[next(iter(eclassmap[ec])) for ec in root_eclasses],
            cost=0.0,
            eclass="common_root",
            op="common_root",
            subsumed=False,
        )

        return common_root

    def _compute_cost(
        self,
        max_iter=1000,
        max_no_progress=100,
        epsilon=1e-6,
    ) -> tuple[dict[str, Bucket], int]:
        """
        Uses dynamic programming with iterative cost propagation

        Args:
            max_iter (int, optional):
                Maximum number of iterations to compute costs.
                Defaults to 10000.
            max_no_progress (int, optional):
                Maximum iterations without cost improvement.
                Defaults to 500.

        Returns:
            tuple[dict[str, Bucket], int]: A tuple containing:
                - A mapping of equivalence classes to their lowest cost
                  representations
                - The number of rounds (iterations) that were performed

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

        # Create common root node
        common_root = self._create_common_root(G, eclassmap)

        # Get per-node cost function
        cm = self.cost_model
        nodecostmap: dict[str, CostFunc] = {}
        for k in G.nodes:
            if k not in eclassmap:
                node = nodes[k]
                children_eclasses = [nodes[c].eclass for c in node.children]
                if k == "common_root":
                    nodecostmap[k] = cm.get_simple(0)
                else:
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

        # Use BFS layers to estimate topological sort
        topo_ordered = []
        for layer in nx.bfs_layers(G, [common_root]):
            topo_ordered += layer

        def propagate_cost(state_tracker):
            dagcost = SubgraphCost(
                eclassmap=eclassmap,
                selections=selections,
                nodecostmap=nodecostmap,
                nodes=nodes,
                constants=constants,
            )

            for k in reversed(topo_ordered):
                if k not in eclassmap:
                    node = nodes[k]
                    cost_dag = dagcost.compute_cost(k)
                    cost = sum(cost_dag.values())
                    selections[node.eclass].put(cost, k)

            state_tracker(selections)

        # Repeatedly compute cost and propagate while keeping the best variant
        # for each eclass. If root score is computed, return early.
        costchanged = ConvergenceTracker(epsilon=epsilon)
        state_tracker = ExtractionStateTracker()

        last_changed_i = 0
        for round_i in range(max_iter):
            # propagate
            propagate_cost(state_tracker)
            # check convergence condition
            if state_tracker.last:
                costchanged.update(state_tracker)
                if costchanged.converged():
                    # root score is computed?
                    if all(
                        math.isfinite(state_tracker.current[root])
                        for root in [common_root]
                    ):
                        break

                    # root score is missing?
                    if round_i - last_changed_i >= max_no_progress:
                        # no changes for max_no_progress iteration
                        raise ExtractionError(
                            "extraction stopped due to lack of progress for "
                            f"{max_no_progress} iterations"
                        )
                else:
                    last_changed_i = round_i

        return selections, round_i

    def choose(
        self, stats: dict[str, Any] | None = None
    ) -> tuple[float, nx.MultiDiGraph]:
        selections, round_i = self._compute_cost()
        if stats is not None:
            stats["extraction_iteration_count"] = round_i

        nodes = self.nodes
        assert self.root_eclass == "common_root"
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


@dataclass
class _SubgraphCostStats:
    cache_miss: int = 0
    cache_hit: int = 0


@dataclass(frozen=True)
class SubgraphCost:
    """
    Computes costs for nodes in a subgraph, optimizing for efficiency and
    handling complex dependencies.

    This class calculates the cost of nodes in a subgraph by combining local
    node costs with the costs of their child nodes, selected optimally from
    eclasses. It combines two cost computation strategies:

    - **Local cost computation**: Evaluates a node's cost using its cost
      function, which for simple operations reflects basic computation cost and
      for control nodes (e.g., loops) may incorporate factors like loop trip
      counts multiplied by the loop body's cost.
    - **Child-dependent cost computation**: Aggregates costs from child nodes,
      using the best available node from each child's e-class to ensure optimal
      selections. Each child and grand-child node is accounted for only once,
      preventing redundant cost contributions from shared nodes in the
      subgraph.

    To avoid redundant computations in cyclic subgraphs, the class uses caching
    and stops cost computation if costs become infinite (e.g., due to cycles).
    This effectively handles repeated nodes without recomputing their costs.

    """

    nodes: dict[str, Node]
    "Maps node names to Node objects."
    eclassmap: dict[str, Any]
    "Maps node names to their eclasses."
    selections: dict[str, Bucket]
    "Maps eclasses to Buckets containing node choices."
    nodecostmap: dict[str, CostFunc]
    "Maps node names to cost functions."
    constants: dict[str, Any]
    "Maps eclasses to constant values (eclass of primitive nodes)"
    _cache: dict[str, dict[str, float]] = field(default_factory=dict)
    "Stores computed node costs for reuse."
    _stats: _SubgraphCostStats = field(default_factory=_SubgraphCostStats)
    "Tracks cache hits and misses."
    _visited_dag_eclass: list[str] = field(default_factory=list)
    "Tracks Children DAG path to avoid recursion"

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

        # ---- Local cost computation ----
        nodes = self.nodes
        node = nodes[nodename]
        child_costs = []
        for child in node.children:
            child_eclass = nodes[child].eclass
            if choices := selections[child_eclass]:
                cc = choices.best().cost
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

        # ---- Child-dependent cost computation ----
        for child in node.children:
            costs.update(self._compute_choice(nodes[child].eclass))
        return costs

    def _compute_choice(self, eclass: str) -> dict[str, float]:
        if eclass in self._visited_dag_eclass:
            # Avoid recursion
            return {eclass: MAX_COST}
        self._visited_dag_eclass.append(eclass)
        try:
            selections = self.selections

            choices = selections[eclass]
            if not choices:
                return {eclass: MAX_COST}
            best = choices.best()

            return self.compute_cost(best.name)
        finally:
            self._visited_dag_eclass.pop()

class ExtractionError(Exception):
    pass


@dataclass(frozen=True)
class ExtractionStateTracker:
    """Keeps history of the last two selection states."""

    stack: deque[dict[str, float]] = field(
        default_factory=lambda: deque(maxlen=2)
    )

    def __call__(self, selections: dict[str, Bucket]) -> dict[str, float]:
        state = {
            k: bkt.best().cost if bkt else MAX_COST
            for k, bkt in selections.items()
        }
        if len(self.stack) >= 2:
            self.stack.popleft()
            # print("DROPPED")
        self.stack.append(state)
        return state

    @property
    def last(self) -> dict[str, float] | None:
        if len(self.stack) >= 2:
            return self.stack[0]
        return None

    @property
    def current(self) -> dict[str, float]:
        return self.stack[-1]


@dataclass
class ConvergenceTracker:
    """
    Convergence happens when the number of valid costs are unchanging and
    the maximum cost changes is below epsilon.
    """

    epsilon: float
    max_change: float = 0
    num_valid_last: int = 0
    num_valid_curr: int = 0

    def update(self, tracker: ExtractionStateTracker) -> None:
        last = tracker.last
        curr = tracker.current

        max_change = float(0)
        num_valid_last = 0
        num_valid_curr = 0
        assert last
        for eclass in curr:
            curr_cost = curr[eclass]
            prev_cost = last[eclass]
            if math.isfinite(curr_cost) and math.isfinite(prev_cost):
                max_change = max(max_change, abs(curr_cost - prev_cost))
            else:
                num_valid_last += math.isinf(prev_cost)
                num_valid_curr += math.isinf(curr_cost)
        self.max_change = max_change
        self.num_valid_last = num_valid_last
        self.num_valid_curr = num_valid_curr

    def converged(self) -> bool:
        return (
            self.num_valid_last == self.num_valid_curr
            and self.max_change < self.epsilon
        )


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
