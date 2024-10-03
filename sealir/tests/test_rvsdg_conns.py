import inspect

from sealir import grammar, rvsdg, rvsdg_conns
from sealir.py.canonical import PythonAnalysis

DOT_VIEW = True


def test_for_loop():
    def udt(a, b, c):
        for i in range(a, b):
            c += a + b
            c += c
        return c

    edges = run(udt)
    uda = rvsdg_conns.UseDefAnalysis(edges)

    pa = PythonAnalysis(uda)
    [loop_info] = pa.find_for_loops()

    callee_pos = grammar.field_position(rvsdg.Py_Call, "callee")
    assert isinstance(loop_info.range_node.expr, rvsdg.Py_Call)

    def callee(node) -> str:
        return uda.node_inputs[node].get_input_port(callee_pos).expr.name

    assert callee(loop_info.range_node) == "range"
    assert isinstance(loop_info.iter_node.expr, rvsdg.Py_Call)
    assert callee(loop_info.iter_node) == "iter"
    assert isinstance(loop_info.next_node.expr, rvsdg.Py_Call)
    assert callee(loop_info.next_node) == "next"
    assert isinstance(loop_info.endloop_compare.expr, rvsdg.Py_Compare)
    print(loop_info)


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
