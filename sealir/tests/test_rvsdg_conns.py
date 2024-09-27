import inspect

from sealir import rvsdg, rvsdg_conns

DOT_VIEW = True


def test_for_loop():
    def udt(a, b, c):
        for i in range(a, b):
            c += a + b
            c += c
        return c

    run(udt)


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
