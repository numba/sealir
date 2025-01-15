from __future__ import annotations

from textwrap import dedent

from sealir import ase, grammar
from sealir.lam import (
    App,
    LamGrammar,
    app_func,
    beta_reduction,
    format,
    lam_func,
    run_abstraction_pass,
)


class _Val(grammar.Rule):
    pass


class Num(_Val):
    val: int


class _BinOp(_Val):
    lhs: ase.SExpr
    rhs: ase.SExpr


class Add(_BinOp): ...


class Sub(_BinOp): ...


class Mul(_BinOp): ...


class Lt(_BinOp): ...


class Loop(_Val):
    body: ase.SExpr
    args: tuple[ase.SExpr, ...]


class If(_Val):
    cond: ase.SExpr
    then: ase.SExpr
    args: tuple[ase.SExpr, ...]


class Func(_Val):
    body: ase.SExpr
    name: str
    args: tuple[ase.SExpr, ...]


class Tuple(_Val):
    args: tuple[ase.SExpr, ...]


class UsecaseLamGrammar(grammar.Grammar):
    start = LamGrammar.start | _Val


def test_curry():
    with UsecaseLamGrammar(ase.Tape()) as grm:

        @lam_func(grm)
        def lam1(a, b):
            return a

        out = app_func(grm, lam1, grm.write(Num(1)), grm.write(Num(2)))
        print(ase.pretty_str(out))
        assert (
            ase.pretty_str(out)
            == "(App (Num 2) (App (Num 1) (Lam (Lam (Arg 1)))))"
        )

        @lam_func(grm)
        def lam2(a, b):
            return b

        out = app_func(grm, lam2, grm.write(Num(1)), grm.write(Num(2)))
        print(ase.pretty_str(out))
        assert (
            ase.pretty_str(out)
            == "(App (Num 2) (App (Num 1) (Lam (Lam (Arg 0)))))"
        )


def test_lam():
    with UsecaseLamGrammar(ase.Tape()) as grm:

        @lam_func(grm)
        def lam1(a, b):
            c = grm.write(Add(lhs=a, rhs=b))
            d = grm.write(Sub(lhs=a, rhs=a))
            return grm.write(Tuple((c, d)))

        out = app_func(grm, lam1, grm.write(Num(1)), grm.write(Num(2)))

    print(ase.pretty_str(lam1))
    print(ase.pretty_str(out))

    with grm:
        beta_out2 = beta_reduction(out)

    print(ase.pretty_str(beta_out2))
    assert (
        ase.pretty_str(beta_out2)
        == "(Tuple (Add (Num 1) (Num 2)) (Sub (Num 1) (Num 1)))"
    )


def test_lam_loop():
    with UsecaseLamGrammar(ase.Tape()) as grm:

        # Implements
        #
        # c =0
        # for i in range(n):
        #    c += i
        # return c

        @lam_func(grm)
        def loop_body(n, i, c):
            # i < n
            loop_cont = grm.write(Lt(lhs=i, rhs=n))
            # c += i
            c = grm.write(Add(lhs=c, rhs=i))
            # i += 1
            i = grm.write(Add(lhs=i, rhs=grm.write(Num(1))))
            # return (loop_cont, n, i, c)
            ret = grm.write(Tuple((loop_cont, grm.write(Tuple((n, i, c))))))
            return ret

        @lam_func(grm)
        def if_body(n, i):
            c = grm.write(Num(0))
            return grm.write(Loop(body=loop_body, args=(n, i, c)))

        @lam_func(grm)
        def func_body(n):
            i = grm.write(Num(0))
            lt = grm.write(Lt(lhs=i, rhs=n))
            ifthen = grm.write(If(cond=lt, then=if_body, args=(n, i)))
            return ifthen

        toplevel = grm.write(Func(body=func_body, name="foo", args=(1,)))

    print("Format")
    print(ase.pretty_str(toplevel))
    print(format(toplevel))

    print(grm._tape.dump())


def test_lam_abstract():
    """Abstraction pass introduces `app (lam)` to ensure all variables are
    single use.
    """

    # Implements
    #
    # func ( x )
    #    a = x * 2
    #    b = a + x
    #    c = a - 1
    #    d = b * c
    #    return d

    with UsecaseLamGrammar(ase.Tape()) as grm:

        @lam_func(grm)
        def func_body(x):
            a = grm.write(Mul(lhs=x, rhs=grm.write(Num(2))))
            b = grm.write(Add(lhs=a, rhs=x))
            c = grm.write(Sub(lhs=a, rhs=grm.write(Num(1))))
            d = grm.write(Mul(lhs=b, rhs=c))
            return d

        @lam_func(grm)
        def expected_func_body(x):
            a = grm.write(Mul(lhs=x, rhs=grm.write(Num(2))))

            @lam_func(grm)
            def remaining(a):
                b = grm.write(Add(lhs=a, rhs=x))
                c = grm.write(Sub(lhs=a, rhs=grm.write(Num(1))))
                d = grm.write(Mul(lhs=b, rhs=c))
                return d

            return app_func(grm, remaining, a)

        func_body = run_abstraction_pass(grm, func_body)
    print(format(func_body))
    assert ase.pretty_str(expected_func_body) == ase.pretty_str(func_body)
    print(ase.pretty_str(func_body))


def test_lam_abstract_deeper():
    """Abstraction pass introduces `app (lam)` to ensure all variables are
    single use.
    """

    # Implements
    #
    # func ( x )
    #    a = x * 2
    #    b = a + x
    #    c = a - b
    #    d = b * c
    #    return d
    with UsecaseLamGrammar(ase.Tape()) as grm:

        @lam_func(grm)
        def func_body(x):
            a = grm.write(Mul(lhs=x, rhs=grm.write(Num(2))))
            b = grm.write(Add(lhs=a, rhs=x))
            c = grm.write(Sub(lhs=a, rhs=b))
            d = grm.write(Mul(lhs=b, rhs=c))
            return d

        @lam_func(grm)
        def expected_func_body(x):
            a = grm.write(Mul(lhs=x, rhs=grm.write(Num(2))))

            @lam_func(grm)
            def inner(a):
                b = grm.write(Add(lhs=a, rhs=x))

                @lam_func(grm)
                def remaining(b):
                    c = grm.write(Sub(lhs=a, rhs=b))
                    d = grm.write(Mul(lhs=b, rhs=c))
                    return d

                return app_func(grm, remaining, b)

            return app_func(grm, inner, a)

        func_body = run_abstraction_pass(grm, func_body)
    print(format(func_body))
    print(format(expected_func_body))
    assert ase.pretty_str(expected_func_body) == ase.pretty_str(func_body)
    print(ase.pretty_str(func_body))


def test_lam_identity():
    with UsecaseLamGrammar(ase.Tape()) as grm:

        @lam_func(grm)
        def func_body(x):
            return x

    # This is a special case for the formatter to only contain a simple expr
    out = format(func_body)
    expected_str = dedent(
        """
        let $1 = λ {
          (Arg 0)
        }
    """
    )
    assert out.strip() == expected_str.strip()
    assert ase.pretty_str(func_body) == "(Lam (Arg 0))"
