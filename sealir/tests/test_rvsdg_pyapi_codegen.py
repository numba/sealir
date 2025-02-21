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


def test_for_loop_reduce_add_1d():
    def udt(n: int) -> int:
        c = 0
        for i in range(n):
            c += i
        return c

    args = (5,)
    run(udt, args)

    args = (0,)
    run(udt, args)


def test_for_loop_reduce_add_2d():
    def udt(n: int) -> int:
        c = 0
        for i in range(n):
            for j in range(i):
                c += i + j
        return c

    args = (5,)
    run(udt, args)

    args = (0,)
    run(udt, args)


def test_for_loop_reduce_add_2d_w_break():
    def udt(n: int) -> int:
        c = 0
        for i in range(n):
            for j in range(i):
                c += i * j
                if c > 20:
                    break
        return c

    args = (5,)
    run(udt, args)

    args = (0,)
    run(udt, args)


def test_for_if_1():
    def udt(n):
        t = 0
        for i in range(n):
            if i > 2:
                # `s` is first defined in the loop conditionally
                s = 123
            t += i

        return t, s

    args = (5,)
    run(udt, args)


def run(func, args, *, localscope=None):
    from sealir.rvsdg.restructuring import format_rvsdg

    expected = func(*args)

    func, dbg = restructure_source(func)

    print(format_rvsdg(func))

    assert localscope is None

    cg = llvm_codegen(func)
    res = cg(*args)
    assert res == expected
