from collections import ChainMap

import pytest

from sealir import ase, lam
from sealir.llvm_pyapi_backend import llvm_codegen
from sealir.rvsdg import Grammar, restructure_source


def test_return_arg0():
    def udt(n: int, m: int) -> int:
        return n

    args = (12, 32)
    run(udt, args)


def test_return_arg1():
    def udt(n: int, m: int) -> int:
        return m

    args = (12, 32)
    run(udt, args)


def test_simple_add():
    def udt(n: int, m: int) -> int:
        a = n + m
        return a

    args = (12, 32)
    run(udt, args)


def test_chained_binop():
    def udt(n: int, m: int) -> int:
        a = n + m * 10
        return a

    args = (12, 32)
    run(udt, args)


def test_inplace_add_1():
    def udt(n: int, m: int) -> int:
        a = n + m
        a += n
        return a

    args = (12, 32)
    run(udt, args)


def test_multi_assign():
    def udt(n: int, m: int) -> tuple[int, int]:
        a = b = n + m
        return a, b

    args = (12, 32)
    run(udt, args)


def test_if_else_1():
    def udt(n: int, m: int) -> int:
        # basic min
        if n < m:
            out = n
        else:
            out = m
        return out

    args = (12, 32)
    run(udt, args)

    args = (32, 12)
    run(udt, args)


def test_if_else_2():
    def udt(n: int, m: int) -> int:
        if n < m:
            x = n
            y = m
        else:
            x = m
            y = n
        return x, y

    args = (12, 32)
    run(udt, args)

    args = (32, 12)
    run(udt, args)


def test_if_else_3():
    def udt(n: int, m: int) -> int:
        if m > n:
            a = b = m
        else:
            a = b = n
        return a, b

    args = (12, 32)
    run(udt, args)
    args = (32, 12)
    run(udt, args)


def test_if_else_4():
    def udt(n: int, m: int) -> int:
        a = m + n
        c = a
        if m > n:
            a = b = n + m
        else:
            a = b = n * m
        c += a
        c *= b
        return c

    args = (12, 32)
    run(udt, args)

    args = (32, 12)
    run(udt, args)


def test_while_1():
    def udt(n: int, m: int) -> tuple[int, int]:
        i = 0
        c = 0
        while i < n:
            c += i * m
            i += 1
        return i, c

    args = (5, 3)
    run(udt, args)

    args = (0, 3)
    run(udt, args)


def test_range_iterator_1():
    def udt(n: int) -> tuple[int, int]:
        it = iter(range(n))
        a = next(it)
        b = next(it)
        return a, b

    args = (5,)
    run(udt, args)


def run(func, args, *, localscope=None):
    expected = func(*args)

    lam_node = restructure_source(func)

    assert localscope is None

    cg = llvm_codegen(lam_node)
    res = cg(*args)
    assert res == expected
