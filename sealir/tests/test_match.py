from __future__ import annotations

from textwrap import dedent

from sealir import ase


def test_match():
    """Tests the matching functionality of the ase module.

    This test suite verifies that the pattern matching functionality of the
    ase module works as expected. It checks various cases of matching
    expressions with different depths and structures.
    """
    with ase.Tree():
        a = ase.expr("num", 123)
        b = ase.expr("num", 321)
        c = ase.expr("add", a, a)
        d = ase.expr("sub", c, b)
        e = ase.expr("mul", b, d)

    match e.as_tuple():
        case ('mul', x, y):
            assert x == b
            assert y == d
        case _:
            assert False

    match c.as_tuple():
        case ('add', x, y):
            assert x == y
            assert isinstance(x, ase.Expr)
            match x.as_tuple():
                case ('num', 123):
                    ...
                case _:
                    assert False
        case _:
            assert False

    match c.as_tuple(depth=2):
        case ('add', ("num", x), ("num", y)):
            assert x == y
            assert x == 123
        case _:
            assert False

    match d.as_tuple(depth=2):
        case ('sub', ("add", x, y), ("num", 321)):
            assert isinstance(x, ase.Expr)
            assert isinstance(y, ase.Expr)
            assert x == a
            assert y == a
        case _:
            assert False

    match d.as_tuple(depth=3):
        case ('sub', ("add", x, ('num', y)), ("num", 321)):
            assert isinstance(x, tuple)
            assert isinstance(y, int)
            match x:
                case ("num", z):
                    assert z == y
                case _:
                    assert False
        case _:
            assert False