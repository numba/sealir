from collections.abc import Generator

from sealir import ase, grammar


@grammar.rule
class Val:
    pass


@grammar.rule
class Num(Val):
    value: int


@grammar.rule
class BinOp(Val):
    lhs: ase.BaseExpr
    rhs: ase.BaseExpr


@grammar.rule
class Add(BinOp): ...


@grammar.rule
class Sub(BinOp): ...


@grammar.rule
class Mul(BinOp): ...


class CalcGrammar(grammar.Grammar):
    start = Val


def test_calculator():

    assert Num.__match_args__ == ("value",)
    assert Add.__match_args__ == ("lhs", "rhs")
    assert Sub.__match_args__ == ("lhs", "rhs")
    assert Mul.__match_args__ == ("lhs", "rhs")

    with ase.Tape() as tp:
        grm = CalcGrammar(tp)
        a = grm.write(Num(value=123))
        b = grm.write(Num(value=321))
        c = grm.write(Add(lhs=a, rhs=a))
        d = grm.write(Sub(lhs=c, rhs=b))
        e = grm.write(Mul(lhs=b, rhs=d))

    assert c.lhs.value == a.value

    class Calc(ase.TreeVisitor):
        def __init__(self):
            self.memo = {}

        def visit(self, expr: ase.BaseExpr):
            memo = self.memo
            match grm.match(expr):
                case Num(value=val):
                    result = val
                case Add(lhs=lhs, rhs=rhs):
                    result = memo[lhs] + memo[rhs]
                case Sub(lhs=lhs, rhs=rhs):
                    result = memo[lhs] - memo[rhs]
                case Mul(lhs=lhs, rhs=rhs):
                    result = memo[lhs] * memo[rhs]
                case _:
                    raise AssertionError("unknown op")
            memo[expr] = result

    calc = Calc()
    ase.apply_bottomup(e, calc)
    result = calc.memo[e]

    def expected():
        a = 123
        b = 321
        c = a + a
        d = c - b
        e = b * d
        return e

    assert expected() == result


def test_calculator_traverse():
    with ase.Tape() as tp:
        grm = CalcGrammar(tp)
        a = grm.write(Num(value=123))
        b = grm.write(Num(value=321))
        c = grm.write(Add(lhs=a, rhs=a))
        d = grm.write(Sub(lhs=c, rhs=b))
        e = grm.write(Mul(lhs=b, rhs=d))

    def calc(
        sexpr: ase.BaseExpr, state: ase.TraverseState
    ) -> Generator[ase.BaseExpr, int, int]:
        match sexpr:
            case Num(value=int(value)):
                return value
            case Add(lhs=lhs, rhs=rhs):
                return (yield lhs) + (yield rhs)
            case Sub(lhs=lhs, rhs=rhs):
                return (yield lhs) - (yield rhs)
            case Mul(lhs=lhs, rhs=rhs):
                return (yield lhs) * (yield rhs)
            case _:
                raise AssertionError(sexpr)

    memo = ase.traverse(e, calc)
    print(memo)
