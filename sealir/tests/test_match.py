from __future__ import annotations

from textwrap import dedent

from sealir import ase
from sealir.ase import SimpleExpr as Expr


def test_match():
    """Tests the matching functionality of the ase module.

    This test suite verifies that the pattern matching functionality of the
    ase module works as expected. It checks various cases of matching
    expressions with different depths and structures.
    """
    with ase.Tape() as tp:
        a = tp.expr("num", 123)
        b = tp.expr("num", 321)
        c = tp.expr("add", a, a)
        d = tp.expr("sub", c, b)
        e = tp.expr("mul", b, d)

    match e:
        case Expr("mul", (x, y)):
            assert x == b
            assert y == d
        case _:
            assert False

    match c:
        case Expr("add", (x, y)):
            assert x == y
            assert isinstance(x, Expr)
            match x:
                case Expr("num", (123,)):
                    ...
                case _:
                    assert False
        case _:
            assert False

    match c:
        case Expr("add", (Expr("num", (x,)), Expr("num", (y,)))):
            assert x == y
            assert x == 123
        case _:
            assert False

    match d:
        case Expr("sub", (Expr("add", (x, y)), Expr("num", (321,)))):
            assert isinstance(x, Expr)
            assert isinstance(y, Expr)
            assert x == a
            assert y == a
        case _:
            assert False

    match d:
        case Expr(
            "sub", (Expr("add", (x, Expr("num", (y,)))), Expr("num", (321,)))
        ):
            assert isinstance(y, int)
            match x:
                case Expr("num", (z,)):
                    assert z == y
                case _:
                    assert False
        case _:
            assert False
