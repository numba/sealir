import inspect
from functools import partial
from pprint import pprint
from typing import cast

from sealir import rvsdg, rvsdg_conns

DOT_VIEW = False


def test_for_loop():
    def udt(a, b, c):
        for i in range(a, b):
            c += a + b
            c += c
        return c

    edges = run(udt)
    uda = rvsdg_conns.UseDefAnalysis(edges)

    def callee_is_global_load(node: rvsdg_conns.GraphNode, name: str) -> bool:
        callee_pos = rvsdg.Py_Call._field_position("callee")
        callee_node = uda.node_inputs[node]
        callee_expr = cast(
            rvsdg_conns.ExprNode, callee_node.get_input_port(callee_pos)
        )
        match callee_expr.expr:
            case rvsdg.Py_GlobalLoad(name=k) if k == name:
                return True
        return False

    # Find descendent of range() call
    def match_range_call(node: rvsdg_conns.GraphNode) -> bool:
        # get call to range()
        match node:
            case rvsdg_conns.ExprNode(expr=rvsdg.Py_Call(callee=callee)):
                pass
            case _:
                return False

        return callee_is_global_load(node, name="range")

    def get_iter_node(prefixes, x):
        [range_node] = prefixes
        if uda.is_sole_user(x, of=range_node, port=1):
            match x:
                case rvsdg_conns.ExprNode(expr=rvsdg.Py_Call()):
                    if callee_is_global_load(x, name="iter"):
                        return True

    def get_next_node(prefixes, x):
        [_, iter_node] = prefixes
        if uda.is_sole_user(x, of=iter_node, port=1):
            match x:
                case rvsdg_conns.ExprNode(expr=rvsdg.Py_Call()):
                    if callee_is_global_load(x, name="next"):
                        return True

    [(range_node, iter_node, next_node)] = list(
        uda.search_use_chain(
            lambda _, x: match_range_call(x), get_iter_node, get_next_node
        )
    )

    callee_pos = rvsdg.Py_Call._field_position("callee")
    assert isinstance(range_node.expr, rvsdg.Py_Call)
    assert (
        uda.node_inputs[range_node].get_input_port(callee_pos).expr.name
        == "range"
    )
    assert isinstance(iter_node.expr, rvsdg.Py_Call)
    assert (
        uda.node_inputs[iter_node].get_input_port(callee_pos).expr.name
        == "iter"
    )
    assert isinstance(next_node.expr, rvsdg.Py_Call)
    assert (
        uda.node_inputs[next_node].get_input_port(callee_pos).expr.name
        == "next"
    )


def test_if_else():
    def udt(a, b):
        if a < b:
            c = b + 1
        else:
            c = a + 1
        return c

    run(udt)


def run(udt):
    node = rvsdg.restructure_source(udt)
    sig = inspect.signature(udt)
    edges = rvsdg_conns.build_value_state_connection(
        node, sig.parameters.keys()
    )

    dot = rvsdg_conns.render_dot(edges)
    if DOT_VIEW:
        dot.view()
    return edges
