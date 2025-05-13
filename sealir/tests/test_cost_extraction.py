import json
import re

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
    ControlCostFunc,
    CostModel,
    EGraphJsonDict,
    Extraction,
    SimpleCostFunc,
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


@function
def Loop(body_expr: Term) -> Term: ...


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        match op, tuple(children):
            case "Pow", _:
                cost = 10
            case "Loop", (expr,):
                return self.get_control(self_cost=13, multipliers=[23])

        return self.get_simple(cost)


@ruleset
def simplify_pow(x: Term, i: i64):
    yield rewrite(Pow(x, i)).to(Mul(x, Pow(x, i - 1)), i > 1)
    yield rewrite(Pow(x, 1)).to(x)


def _extraction(egraph, cost_model=None):
    # TODO move this into rvsdg_extract
    gdct: EGraphJsonDict = json.loads(
        egraph._serialize(
            n_inline_leaves=0, split_primitive_outputs=False
        ).to_json()
    )
    [root] = get_graph_root(gdct)
    root_eclass = gdct["nodes"][root]["eclass"]

    cost_model = cost_model or CostModel()
    _extraction = Extraction(gdct, root_eclass, cost_model)
    cost, exgraph = _extraction.choose()
    return cost, exgraph


def _flatten_multidigraph(graph):
    def format_node(node):
        # Get all successors (children) of the node
        children = [v for u, v in graph.out_edges(node)]
        if not children:
            return node
        return tuple([node, *(format_node(child) for child in children)])

    # Get all nodes with no incoming edges (roots)
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    return [format_node(root) for root in roots]


def test_cost_duplicated_term():
    A = Term("A")
    B = Term("B")
    expr = Add(A, Add(B, A))
    egraph = EGraph()
    egraph.register(GraphRoot(expr))

    cost, exgraph = _extraction(egraph, cost_model=MyCostModel())
    [extracted] = _flatten_multidigraph(exgraph)
    print("extracted:", extracted)
    # t1 = function-0-Term___init__(primitive-String-2684354568)
    #  1 ------------^
    #  1 -------------------------------^
    # t2 = function-1-Term___init__(primitive-String-1610612745)
    #  1 ------------^
    #  1 -------------------------------^
    # GraphRoot(Add(t1,Add(t2,t1)))
    #  1 -^
    #  1 --------^
    #  1 ------------^
    # = 7
    assert cost == 7

    match extracted:
        case (graphroot, (add1, term1, (add2, term2, term3))):
            pass
        case _:
            assert False, f"failed to match: {extracted}"
    assert re.match(r"function-\d+-Term___init__", term1[0])
    assert re.match(r"function-\d+-Term___init__", term2[0])
    assert re.match(r"function-\d+-Term___init__", term3[0])
    assert term1 == term3
    assert term2 != term3
    assert add1 != add2
    assert add1 != add2
    assert graphroot.endswith("GraphRoot")
    assert add1.endswith("Add")
    assert add2.endswith("Add")


def test_simplify_pow_2():

    A = Term("A")
    expr = Pow(A, 2)
    egraph = EGraph()
    egraph.register(GraphRoot(expr))

    cost, exgraph = _extraction(egraph, cost_model=MyCostModel())
    assert cost == 14

    egraph.run(simplify_pow.saturate())
    cost, exgraph = _extraction(egraph, cost_model=MyCostModel())
    [extracted] = _flatten_multidigraph(exgraph)

    print("extracted:", extracted)
    # t1 = function-1-Term___init__(primitive-String-2684354568)
    #  1 ------------^
    #  1 ----------------------------------^
    # GraphRoot(Mul(t1, t1)))
    #  1 --^
    #  1 --------^
    # = 4
    assert cost == 4
    match extracted:
        case (graphroot, (mul1, term1, term2)):
            pass
        case _:
            assert False, f"failed to match: {extracted}"
    assert re.match(r"function-\d+-Term___init__", term1[0])
    assert re.match(r"function-\d+-Term___init__", term2[0])
    assert term1 == term2
    assert graphroot.endswith("GraphRoot")
    assert mul1.endswith("Mul")


def test_simplify_pow_3():
    A = Term("A")
    expr = Pow(A, 3)
    egraph = EGraph()
    egraph.register(GraphRoot(expr))

    cost, exgraph = _extraction(egraph, cost_model=MyCostModel())
    assert cost == 14

    egraph.run(simplify_pow.saturate())
    cost, exgraph = _extraction(egraph, cost_model=MyCostModel())
    [extracted] = _flatten_multidigraph(exgraph)

    print("extracted:", extracted)
    # t1 = function-1-Term___init__(primitive-String-2684354568)
    #  1 ------------^
    #  1 ----------------------------------^
    # GraphRoot(Mul(t1, Mul(t1, t1)))
    #  1 --^
    #  1 --------^
    #  1 ----------------^
    # = 5
    assert cost == 5
    match extracted:
        case (graphroot, (mul1, term1, (mul2, term2, term3))):
            pass
        case _:
            assert False, f"failed to match: {extracted}"
    assert re.match(r"function-\d+-Term___init__", term1[0])
    assert term1 == term2
    assert term1 == term3
    assert graphroot.endswith("GraphRoot")
    assert mul1.endswith("Mul")
    assert mul2.endswith("Mul")


def test_simple_cost_func():
    scf = SimpleCostFunc(self_cost=10)
    cost = scf.compute([7, 8, 9])
    assert cost == 10
    assert CostModel().get_simple(self_cost=10) == scf


def test_control_cost_func():
    ccf = ControlCostFunc(self_cost=10, multipliers=(2, 3, 4))
    cost = ccf.compute([7, 8, 9])
    assert cost == 10 + (2 * 7) + (3 * 8) + (4 * 9)
    assert CostModel().get_control(10, [2, 3, 4]) == ccf


def test_loop_multiplier():
    A = Term("A")
    expr = Loop(Pow(A, 3))
    egraph = EGraph()
    egraph.register(GraphRoot(expr))
    egraph.run(simplify_pow.saturate())
    cost, exgraph = _extraction(egraph, cost_model=MyCostModel())
    [extracted] = _flatten_multidigraph(exgraph)
    print("extracted", extracted)
    # Pow(A, 3) = 4 (see test_simplify_pow_3)
    # Loop(A, 3) = (4 * 23 + 13) + 4
    #               ^^^^^^^^^^^ cost from multiplier and self_cost
    #                            ^^^^ cost from DAG of children
    # GraphRoot(...) = 1
    assert cost == (4 * 23 + 13) + 4 + 1

    match extracted:
        case (graphroot, (loop, (mul1, term1, (mul2, term2, term3)))):
            pass
        case _:
            assert False, f"failed to match: {extracted}"
    assert re.match(r"function-\d+-Term___init__", term1[0])
    assert term1 == term2
    assert term1 == term3
    assert graphroot.endswith("GraphRoot")
    assert mul1.endswith("Mul")
    assert mul2.endswith("Mul")
    assert loop.endswith("Loop")
