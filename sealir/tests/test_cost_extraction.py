import json

from egglog import (
    EGraph,
    Expr,
    StringLike,
    function,
    i64,
    i64Like,
    rewrite,
    ruleset,
)

from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import (
    CostModel,
    EGraphJsonDict,
    Extraction,
    get_graph_root,
)


class Term(Expr):
    def __init__(self, name: StringLike): ...


@function
def Add(lhs: Term, rhs: Term) -> Term: ...


@function
def Mul(lhs: Term, rhs: Term) -> Term: ...


@function
def Pow(lhs: Term, i: i64Like) -> Term: ...


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        match op:
            case "Pow":
                cost = 10

        return self.get_simple(cost)


@ruleset
def simplify_pow(x: Term, i: i64):
    yield rewrite(Pow(x, i)).to(Mul(x, Pow(x, i - 1)), i > 0)
    yield rewrite(Pow(x, 0)).to(x)


def test_cost_duplicated_term():
    A = Term("A")
    B = Term("B")
    expr = Add(A, Add(B, A))
    egraph = EGraph()
    egraph.register(GraphRoot(expr))

    cost, exgraph = extraction(egraph, cost_model=MyCostModel())
    print("extracted:", flatten_multidigraph(exgraph))
    # t1 = function-0-Term___init__(primitive-String-2684354568)
    #  1 ------------^
    #  1 -------------------------------^
    # t2 = function-1-Term___init__(primitive-String-1610612745)
    #  1 ------------^
    #  1 -------------------------------^
    # GraphRoot(Add(Add(t1,t2),t1))
    #  1 -^
    #  1 --------^
    #  1 ------------^
    # = 7
    assert cost == 7


def test_simplify_power():
    A = Term("A")
    expr = Pow(A, 2)
    egraph = EGraph()
    egraph.register(GraphRoot(expr))

    cost, exgraph = extraction(egraph, cost_model=MyCostModel())
    assert cost == 14

    egraph.run(simplify_pow.saturate())
    cost, exgraph = extraction(egraph, cost_model=MyCostModel())

    print("extracted:", flatten_multidigraph(exgraph))
    # t1 = function-1-Term___init__(primitive-String-2684354568)
    #  1 ------------^
    #  1 ----------------------------------^
    # GraphRoot(Mul(Mul(t1), t1)))
    #  1 --^
    #  1 --------^
    #  1 ------------^
    # = 5
    assert cost == 5


def extraction(egraph, cost_model=None):
    # TODO move this into rvsdg_extract
    gdct: EGraphJsonDict = json.loads(
        egraph._serialize(
            n_inline_leaves=0, split_primitive_outputs=False
        ).to_json()
    )
    [root] = get_graph_root(gdct)
    root_eclass = gdct["nodes"][root]["eclass"]

    cost_model = cost_model or CostModel()
    extraction = Extraction(gdct, root_eclass, cost_model)
    cost, exgraph = extraction.choose()
    return cost, exgraph


def flatten_multidigraph(graph):
    def format_node(node):
        # Get all successors (children) of the node
        children = sorted(graph.successors(node))  # Sort for consistent output
        if not children:
            return f"{node}"
        children_str = ",".join(format_node(child) for child in children)
        return f"{node}({children_str})"

    # Get all nodes with no incoming edges (roots)
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]

    # Format each root and join results
    return ",".join(format_node(root) for root in sorted(roots))
