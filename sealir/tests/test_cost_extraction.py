# mypy: disable-error-code="empty-body"
from egglog import EGraph, StringLike, function, i64, i64Like, rewrite, ruleset

from sealir.eqsat.rvsdg_eqsat import GraphRoot, Term
from sealir.eqsat.rvsdg_extract import CostModel, egraph_extraction


@function
def named_term(name: StringLike) -> Term: ...


@function
def Add(lhs: Term, rhs: Term) -> Term: ...


@function
def Mul(lhs: Term, rhs: Term) -> Term: ...


@function
def Pow(lhs: Term, i: i64Like) -> Term: ...


@function
def Loop(body_expr: Term) -> Term: ...


@function
def Repeat(body_expr: Term, ntime: i64Like) -> Term: ...


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        match op, tuple(children):
            case "Pow", _:
                cost = 10
            case "Loop", (expr,):
                return self.get_scaled(self_cost=13, multipliers=[23])
            case "Repeat", (expr, ntimes):

                def equ(expr, _, ntimes):
                    return expr * ntimes

                return self.get_equation(equ, constants=dict(ntimes=ntimes))
        return self.get_simple(cost)


@ruleset
def simplify_pow(x: Term, i: i64):
    yield rewrite(Pow(x, i)).to(Mul(x, Pow(x, i - 1)), i > 1)
    yield rewrite(Pow(x, 1)).to(x)


def _flatten_multidigraph(graph):
    def format_node(node):
        # Get all successors (children) of the node
        children = [v for u, v in graph.out_edges(node)]
        if not children:
            return node
        return tuple([node, *(format_node(child) for child in children)])

    roots = [k for k in graph.nodes if k.endswith("GraphRoot")]
    return [format_node(x) for x in roots]


def _extraction(egraph):
    extraction = egraph_extraction(egraph, cost_model=MyCostModel())
    # Use the new explicit method for extracting with auto-detected root
    result = extraction.extract_graph_root()
    exgraph = result.graph
    [extracted] = _flatten_multidigraph(exgraph)
    return result.cost, extracted


def test_cost_duplicated_term():
    A = named_term("A")
    B = named_term("B")
    expr = Add(A, Add(B, A))
    egraph = EGraph()
    egraph.register(GraphRoot(expr))

    cost, extracted = _extraction(egraph)
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
    assert term1[0].endswith("named_term")
    assert term2[0].endswith("named_term")
    assert term3[0].endswith("named_term")
    assert term1 == term3
    assert term2 != term3
    assert add1 != add2
    assert add1 != add2
    assert graphroot.endswith("GraphRoot")
    assert add1.endswith("Add")
    assert add2.endswith("Add")


def test_simplify_pow_2():

    A = named_term("A")
    expr = Pow(A, 2)
    egraph = EGraph()
    egraph.register(GraphRoot(expr))

    extraction = egraph_extraction(egraph, cost_model=MyCostModel())
    # Use the new explicit method for extracting from default root
    result = extraction.extract_graph_root()
    assert result.cost == 14

    egraph.run(simplify_pow.saturate())

    extraction = egraph_extraction(egraph, cost_model=MyCostModel())
    result = extraction.extract_graph_root()
    cost = result.cost

    [extracted] = _flatten_multidigraph(result.graph)

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
    assert term1[0].endswith("named_term")
    assert term2[0].endswith("named_term")
    assert term1 == term2
    assert graphroot.endswith("GraphRoot")
    assert mul1.endswith("Mul")


def test_simplify_pow_3():
    A = named_term("A")
    expr = Pow(A, 3)
    egraph = EGraph()
    egraph.register(GraphRoot(expr))

    extraction = egraph_extraction(egraph, cost_model=MyCostModel())
    result = extraction.extract_graph_root()
    exgraph = result.graph
    assert result.cost == 14

    egraph.run(simplify_pow.saturate())
    extraction = egraph_extraction(egraph, cost_model=MyCostModel())
    result = extraction.extract_graph_root()
    exgraph = result.graph
    cost = result.cost
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
    assert term1[0].endswith("named_term")
    assert term1 == term2
    assert term1 == term3
    assert graphroot.endswith("GraphRoot")
    assert mul1.endswith("Mul")
    assert mul2.endswith("Mul")


def test_simple_cost_func():
    scf = CostModel().get_simple(self_cost=10)
    cost = scf.compute(7, 8, 9)
    # cost is just self_cost
    assert cost == 10


def test_scaled_cost_func():
    ccf = CostModel().get_scaled(10, [2, 3, 4])
    cost = ccf.compute(7, 8, 9)
    assert cost == 10 + (2 * 7) + (3 * 8) + (4 * 9)


def test_loop_multiplier():
    A = named_term("A")
    expr = Loop(Pow(A, 3))
    egraph = EGraph()
    egraph.register(GraphRoot(expr))
    egraph.run(simplify_pow.saturate())
    cost, extracted = _extraction(egraph)
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
    assert term1[0].endswith("named_term")
    assert term1 == term2
    assert term1 == term3
    assert graphroot.endswith("GraphRoot")
    assert mul1.endswith("Mul")
    assert mul2.endswith("Mul")
    assert loop.endswith("Loop")


def test_const_factor():
    A = named_term("A")
    expr = Repeat(A, 13)
    egraph = EGraph()
    egraph.register(GraphRoot(expr))
    cost, extracted = _extraction(egraph)
    dagcost = 2 + 1 + 1
    #         ^ Term("A")
    #             ^ literal 13
    #                 ^ GraphRoot
    repeatcost = 2 * 13
    #            ^ Term("A")
    #                 ^ ntimes
    assert cost == repeatcost + dagcost
    match extracted:
        case (graphroot, (repeat, term, literal)):
            pass
        case _:
            assert False, f"failed to match: {extracted}"
    assert term[0].endswith("named_term")
    assert literal.startswith("primitive-i64-")
    assert graphroot.endswith("GraphRoot")
    assert repeat.endswith("Repeat")
