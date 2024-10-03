import inspect
from types import SimpleNamespace
from typing import cast

from sealir import ase, grammar, rvsdg, rvsdg_conns

DOT_VIEW = True


def test_for_loop():
    def udt(a, b, c):
        for i in range(a, b):
            c += a + b
            c += c
        return c

    edges = run(udt)
    uda = rvsdg_conns.UseDefAnalysis(edges)

    def callee_is_global_load(node: rvsdg_conns.GraphNode, name: str) -> bool:
        callee_pos = grammar.field_position(rvsdg.Py_Call, "callee")
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
        [*_, iter_node] = prefixes
        if uda.is_sole_user(x, of=iter_node, port=1):
            match x:
                case rvsdg_conns.ExprNode(expr=rvsdg.Py_Call()):
                    if callee_is_global_load(x, name="next"):
                        return True

    def get_endloop_compare(prefixes, x):
        [*_, next_node] = prefixes
        for user in uda.get_users_of(of=next_node, port=1):
            if user == x:
                match user:
                    case rvsdg_conns.ExprNode(expr=rvsdg.Py_Compare()):
                        lhs_idx = grammar.field_position(
                            rvsdg.Py_Compare, "lhs"
                        )
                        rhs_idx = grammar.field_position(
                            rvsdg.Py_Compare, "rhs"
                        )
                        lhs_node = uda.node_inputs[user].get_input_port(
                            lhs_idx
                        )
                        rhs_node = uda.node_inputs[user].get_input_port(
                            rhs_idx
                        )

                        # The following is needed because plain variable is for
                        # capturing but `x.y` is a matched value
                        tmp = SimpleNamespace()
                        tmp.next_node = next_node

                        sentinel = (
                            uda.node_inputs[next_node]
                            .get_input_port(
                                grammar.field_position(rvsdg.Py_Call, "args")
                                + 1
                            )
                            .expr
                        )

                        def capture_info(node):
                            match node:
                                case tmp.next_node:
                                    return (0, node)
                                case rvsdg_conns.ExprNode(
                                    expr=rvsdg.Py_Str() as string
                                ):
                                    return (1, string)
                            return (-1, None)

                        captured = sorted(
                            map(capture_info, (lhs_node, rhs_node))
                        )
                        match captured:
                            case ((0, _), (1, string)) if ase.matches(
                                string, sentinel
                            ):
                                return True
                            case _:
                                continue

    [(range_node, iter_node, next_node, endloop_compare)] = list(
        uda.search_use_chain(
            lambda _, x: match_range_call(x),
            get_iter_node,
            get_next_node,
            get_endloop_compare,
        )
    )

    callee_pos = grammar.field_position(rvsdg.Py_Call, "callee")
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
    assert isinstance(endloop_compare.expr, rvsdg.Py_Compare)
    print(range_node, iter_node, next_node, endloop_compare)


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
