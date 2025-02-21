from __future__ import annotations

from sealir import rvsdg
from sealir.rvsdg.evaluating import evaluate


def test_if_else():

    def udt(c):
        a = c - 1
        if a < c:
            b = a + 2
        else:
            pass
        return b + 3

    rvsdg_ir, dbginfo = rvsdg.restructure_source(udt)
    print(dbginfo.show_sources())
    args = (10,)
    kwargs = {}
    res = evaluate(rvsdg_ir, args, kwargs, dbginfo=dbginfo)
    print("res =", res)
    assert res == udt(*args, **kwargs)


def test_for_loop():

    def udt(n):
        c = 0
        for i in range(n):
            c += i
        return c

    rvsdg_ir, dbginfo = rvsdg.restructure_source(udt)
    print(dbginfo.show_sources())
    args = (10,)
    kwargs = {}
    res = evaluate(rvsdg_ir, args, kwargs, dbginfo=dbginfo)
    print("res =", res)
    assert res == udt(*args, **kwargs)


if __name__ == "__main__":
    # test_for_loop()
    test_if_else()
