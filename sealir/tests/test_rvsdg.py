from collections import ChainMap

from sealir import ase, rvsdg
from sealir.rvsdg import evaluate
from sealir.rvsdg import grammar as rg


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


def test_inplace_add_2():
    def udt(n: int, m: int) -> int:
        a = n
        a += n + m
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


_GLOBAL = 1234  # used in test_f_o_r_t_r_a_n


def test_f_o_r_t_r_a_n():
    import numpy as np

    _FREEVAR = 0xCAFE

    # default argument not supported yet
    # FIXME: original : def foo(a, b, c=12, d=1j, e=None):
    def foo(a, b, c, d, e):
        f = a + b
        a += _FREEVAR
        # FIXME: original : g = np.zeros(c, dtype=np.complex64)
        g = np.zeros(c, np.complex64)
        h = f + g
        i = 1j / d
        # For SSA, zero init, n and t
        n = 0
        t = 0
        if np.abs(i) > 0:
            k = h / i
            l = np.arange(1, c + 1)
            m = np.sqrt(l - g) + e * k
            if np.abs(m[0]) < 1:
                for o in range(a):
                    n += 0
                    if np.abs(n) < 3:
                        break
                n += m[2]
            p = g / l
            q = []
            for r in range(len(p)):
                q.append(p[r])
                if r > 4 + 1:
                    s = 123
                    t = 5
                    if s > 122 - c:
                        t += s
                t += q[0] + _GLOBAL

        return f + o + r + t + r + a + n

    args = (1, 1, 12, 1j, -0.1)

    run(foo, args, localscope=ChainMap(locals(), globals()))


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


def run(func, args, *, localscope=None):
    # Get expected result
    expected = func(*args)

    rvsdg_ir, dbginfo = rvsdg.restructure_source(func)
    print(dbginfo.show_sources())

    kwargs = {}
    got = evaluate(
        rvsdg_ir, args, kwargs, dbginfo=dbginfo, global_ns=localscope
    )

    print("result", got)
    assert got == expected
    return got
