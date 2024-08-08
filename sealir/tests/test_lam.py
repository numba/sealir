from __future__ import annotations

from textwrap import dedent

from sealir.lam import LamBuilder


def test_lam():
    lambar = LamBuilder()

    @lambar.lam_func
    def lam1(a, b):
        c = lambar.expr("add", a, b)
        d = lambar.expr("sub", a, a)
        return lambar.expr("tuple", c, d)

    out = lambar.app(lam1, lambar.expr("num", 1), lambar.expr("num", 2))
    print(out.str())
    beta_out2 = lambar.beta_reduction(out)
    print(beta_out2.str())
    assert (
        beta_out2.str()
        == "(expr 'tuple' (expr 'add' (expr 'num' 1) (expr 'num' 2)) (expr 'sub' (expr 'num' 1) (expr 'num' 1)))"
    )


def test_lam_loop():
    lambar = LamBuilder()

    # Implements
    #
    # c =0
    # for i in range(n):
    #    c += i
    # return c

    @lambar.lam_func
    def loop_body(n, i, c):
        # i < n
        loop_cont = lambar.expr("lt", i, n)
        # c += i
        c = lambar.expr("add", c, i)
        # i += 1
        i = lambar.expr("add", i, lambar.expr("num", 1))
        # return (loop_cont, n, i, c)
        ret = lambar.expr("tuple", loop_cont, lambar.expr("tuple", n, i, c))
        return ret

    @lambar.lam_func
    def if_body(n, i):
        c = lambar.expr("num", 0)
        return lambar.expr("scf.loop", loop_body, n, i, c)

    @lambar.lam_func
    def func_body(n):
        i = lambar.expr("num", 0)
        lt = lambar.expr("lt", i, n)
        ifthen = lambar.expr("scf.if", lt, if_body, n, i)
        return ifthen

    toplevel = lambar.expr("func", func_body, "foo", 1)

    print("Format")
    print(toplevel.str())
    print(lambar.format(toplevel))

    print(lambar._tree.dump())


def test_lam_abstract():
    """Abstraction pass introduces `app (lam)` to ensure all variables are
    single use.
    """
    lambar = LamBuilder()

    # Implements
    #
    # func ( x )
    #    a = x * 2
    #    b = a + x
    #    c = a - 1
    #    d = b * c
    #    return d

    @lambar.lam_func
    def func_body(x):
        a = lambar.expr("mul", x, lambar.expr("num", 2))
        b = lambar.expr("add", a, x)
        c = lambar.expr("sub", a, lambar.expr("num", 1))
        d = lambar.expr("mul", b, c)
        return d

    @lambar.lam_func
    def expected_func_body(x):
        a = lambar.expr("mul", x, lambar.expr("num", 2))

        @lambar.lam_func
        def remaining(a):
            b = lambar.expr("add", a, x)
            c = lambar.expr("sub", a, lambar.expr("num", 1))
            d = lambar.expr("mul", b, c)
            return d

        return lambar.app(remaining, a)

    func_body = lambar.run_abstraction_pass(func_body)
    print(lambar.format(func_body))
    assert expected_func_body.str() == func_body.str()
    print(func_body.str())


def test_lam_abstract_deeper():
    """Abstraction pass introduces `app (lam)` to ensure all variables are
    single use.
    """
    lambar = LamBuilder()

    # Implements
    #
    # func ( x )
    #    a = x * 2
    #    b = a + x
    #    c = a - b
    #    d = b * c
    #    return d

    @lambar.lam_func
    def func_body(x):
        a = lambar.expr("mul", x, lambar.expr("num", 2))
        b = lambar.expr("add", a, x)
        c = lambar.expr("sub", a, b)
        d = lambar.expr("mul", b, c)
        return d

    @lambar.lam_func
    def expected_func_body(x):
        a = lambar.expr("mul", x, lambar.expr("num", 2))

        @lambar.lam_func
        def inner(a):
            b = lambar.expr("add", a, x)

            @lambar.lam_func
            def remaining(b):
                c = lambar.expr("sub", a, b)
                d = lambar.expr("mul", b, c)
                return d

            return lambar.app(remaining, b)

        return lambar.app(inner, a)

    func_body = lambar.run_abstraction_pass(func_body)
    print(lambar.format(func_body))
    assert expected_func_body.str() == func_body.str()
    print(func_body.str())


def test_lam_identity():
    lambar = LamBuilder()

    @lambar.lam_func
    def func_body(x):
        return x

    # This is a special case for the formatter to only contain a simple expr
    out = lambar.format(func_body)
    expected_str = dedent("""
        let $1 = λ {
          (arg 0)
        }
    """)
    assert out.strip() == expected_str.strip()
    assert func_body.str() == "(lam (arg 0))"
