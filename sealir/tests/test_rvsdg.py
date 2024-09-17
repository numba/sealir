from collections import ChainMap

from sealir.lam import LamBuilder
from sealir.rvsdg import (
    EvalCtx,
    EvalLamState,
    lambda_evaluation,
    restructure_source,
)


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
    def udt(n: int, m: int) -> int:
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
    def udt(n: int, m: int) -> int:
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
    def udt(n: int) -> int:
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
            # FIXME  original : if np.abs(m[0]) < 1:
            if np.abs(m[0]) < 5:
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

    def transformed_foo(a, b, c, d, e):
        f = a + b
        a += _FREEVAR
        g = np.zeros(c, np.complex64)
        h = f + g
        i = 1j / d
        n = 0
        t = 0
        if np.abs(i) > 0:
            k = h / i
            l = np.arange(1, c + 1)
            m = np.sqrt(l - g) + e * k
            if np.abs(m[0]) < 5:
                __scfg_iterator_7__ = iter(range(a))
                o = None
                __scfg_loop_cont__ = True
                while __scfg_loop_cont__:
                    __scfg_iter_last_7__ = o
                    o = next(__scfg_iterator_7__, "__scfg_sentinel__")
                    if o != "__scfg_sentinel__":
                        n += 0
                        if np.abs(n) < 3:
                            __scfg_exit_var_1__ = 0
                            __scfg_backedge_var_1__ = 1
                        else:
                            __scfg_backedge_var_1__ = 0
                            __scfg_exit_var_1__ = -1
                    else:
                        __scfg_exit_var_1__ = 1
                        __scfg_backedge_var_1__ = 1
                    __scfg_loop_cont__ = not __scfg_backedge_var_1__
                if __scfg_exit_var_1__ in (0,):
                    pass
                else:
                    o = __scfg_iter_last_7__
                n += m[2]
            else:
                pass
            p = g / l
            q = []
            __scfg_iterator_14__ = iter(range(len(p)))
            r = None
            __scfg_loop_cont__ = True
            while __scfg_loop_cont__:
                __scfg_iter_last_14__ = r
                r = next(__scfg_iterator_14__, "__scfg_sentinel__")
                if r != "__scfg_sentinel__":
                    q.append(p[r])
                    if r > 4 + 1:
                        s = 123
                        t = 5
                        if s > 122 - c:
                            t += s
                        else:
                            pass
                    else:
                        pass
                    t += q[0] + _GLOBAL
                    __scfg_backedge_var_0__ = 0
                else:
                    __scfg_backedge_var_0__ = 1
                __scfg_loop_cont__ = not __scfg_backedge_var_0__
            r = __scfg_iter_last_14__
        else:
            pass
        return f + o + r + t + r + a + n

    args = (1, 1, 12, 1j, 1)
    a = foo(*args)
    b = transformed_foo(*args)
    assert a == b

    run(foo, args, localscope=ChainMap(locals(), globals()))


def run(func, args, *, localscope=None):
    expected = func(*args)

    lam = restructure_source(func)

    # Prepare run
    lb = LamBuilder(lam.tape)

    if localscope is None:
        ctx = EvalCtx.from_arguments(*args)
    else:
        ctx = EvalCtx.from_arguments_and_locals(args, localscope)

    with lam.tape:
        app_root = lb.app(lam, *ctx.make_arg_node())

    # out = lb.format(app_root)
    # print(out)

    memo = app_root.traverse(lambda_evaluation, EvalLamState(context=ctx))
    res = memo[app_root]
    print("result", res)
    got = res[1]

    assert got == expected
    return got
