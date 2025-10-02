from egglog import (
    EGraph,
    Expr,
    StringLike,
    function,
    i64,
    i64Like,
    rewrite,
    ruleset,
    default_cost_model,
    get_callable_args,
    get_callable_fn,
    BaseExpr,
    greedy_dag_cost_model,
)

from sealir.eqsat.rvsdg_eqsat import GraphRoot


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


@function
def Repeat(body_expr: Term, ntime: i64Like) -> Term: ...


@greedy_dag_cost_model
def cost_model(egraph: EGraph, expr: BaseExpr, child_costs: list[int]) -> int:
    callable = get_callable_fn(expr)
    if callable == Pow:
        return sum(child_costs, start=10)
    if callable == Loop:
        return 13 + child_costs[0] * 24
    match get_callable_args(expr, Repeat):
        case (_, i64(n)):
            return sum(child_costs, start=child_costs[0] * n)
    return default_cost_model(egraph, expr, child_costs)


@ruleset
def simplify_pow(x: Term, i: i64):
    yield rewrite(Pow(x, i)).to(Mul(x, Pow(x, i - 1)), i > 1)
    yield rewrite(Pow(x, 1)).to(x)


def test_cost_duplicated_term():
    A = Term("A")
    B = Term("B")
    expr = GraphRoot(Add(A, Add(B, A)))
    egraph = EGraph()
    egraph.register(expr)
    res, cost = egraph.extract(expr, cost_model=cost_model, include_cost=True)
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
    assert cost.total == 7
    assert res == expr


def test_simplify_pow_2():
    A = Term("A")
    expr = GraphRoot(Pow(A, 2))
    egraph = EGraph()
    egraph.register(expr)

    res, cost = egraph.extract(expr, cost_model=cost_model, include_cost=True)
    assert res == expr
    assert cost.total == 14

    egraph.run(simplify_pow.saturate())
    res, cost = egraph.extract(expr, cost_model=cost_model, include_cost=True)

    # t1 = function-1-Term___init__(primitive-String-2684354568)
    #  1 ------------^
    #  1 ----------------------------------^
    # GraphRoot(Mul(t1, t1)))
    #  1 --^
    #  1 --------^
    # = 4
    assert res == GraphRoot(Mul(A, A))
    assert cost.total == 4


def test_simplify_pow_3():
    A = Term("A")
    expr = GraphRoot(Pow(A, 3))
    egraph = EGraph()
    egraph.register(expr)

    res, cost = egraph.extract(expr, cost_model=cost_model, include_cost=True)
    assert res == expr
    assert cost.total == 14

    egraph.run(simplify_pow.saturate())
    res, cost = egraph.extract(expr, cost_model=cost_model, include_cost=True)

    # t1 = function-1-Term___init__(primitive-String-2684354568)
    #  1 ------------^
    #  1 ----------------------------------^
    # GraphRoot(Mul(t1, Mul(t1, t1)))
    #  1 --^
    #  1 --------^
    #  1 ----------------^
    # = 5
    assert res == GraphRoot(Mul(A, Mul(A, A)))
    assert cost.total == 5


def test_loop_multiplier():
    A = Term("A")
    expr = GraphRoot(Loop(Pow(A, 3)))
    egraph = EGraph()
    egraph.register(expr)
    egraph.run(simplify_pow.saturate())
    res, cost = egraph.extract(expr, cost_model=cost_model, include_cost=True)
    # Pow(A, 3) = 4 (see test_simplify_pow_3)
    # Loop(A, 3) = (4 * 23 + 13) + 4
    #               ^^^^^^^^^^^ cost from multiplier and self_cost
    #                            ^^^^ cost from DAG of children
    # GraphRoot(...) = 1
    assert cost.total == (4 * 23 + 13) + 4 + 1
    assert res == GraphRoot(Loop(Mul(A, Mul(A, A))))


def test_const_factor():
    A = Term("A")
    expr = GraphRoot(Repeat(A, 13))
    egraph = EGraph()
    egraph.register(GraphRoot(expr))
    extracted, cost = egraph.extract(expr, cost_model=cost_model, include_cost=True)
    assert extracted == expr
    dagcost = 2 + 1 + 1
    #         ^ Term("A")
    #             ^ literal 13
    #                 ^ GraphRoot
    repeatcost = 2 * 13
    #            ^ Term("A")
    #                 ^ ntimes
    assert cost.total == repeatcost + dagcost
