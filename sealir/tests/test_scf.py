from sealir import ase, scf, lam, grammar

class _Val(grammar.Rule):
    pass


class Int(_Val):
    val: int


class _BinOp(_Val):
    lhs: ase.BaseExpr
    rhs: ase.BaseExpr


class Add(_BinOp): ...


class Sub(_BinOp): ...


class Mul(_BinOp): ...


class Lt(_BinOp): ...


class Loop(_Val):
    body: ase.BaseExpr
    arg: ase.BaseExpr


class IfElse(_Val):
    cond: ase.BaseExpr
    arg: ase.BaseExpr
    then: ase.BaseExpr
    orelse: ase.BaseExpr


class Func(_Val):
    body: ase.BaseExpr
    name: str
    args: tuple[ase.BaseExpr, ...]


class Tuple(_Val):
    args: tuple[ase.BaseExpr, ...]


class TestGrammar(grammar.Grammar):
    start = lam.LamGrammar.start | _Val


def make_sum_reduce_loop(grm: grammar.Grammar):
    @lam.lam_func(grm)
    def func_body(i, x, y):
        n = grm.write(Mul(x, y))
        cond = grm.write(Lt(i, n))
        c = grm.write(Int(0))

        @scf.region(grm)
        def loop_start(i, n, c):

            @scf.region(grm)
            def loop_body(i, n, c):
                c = grm.write(Add(i, c))
                i = grm.write(Add(i, grm.write(Int(1))))
                datatup = grm.write(Tuple((i, n, c)))
                cond = grm.write(Lt(i, n))
                tup = grm.write(Tuple((cond, datatup)))
                return tup

            tup = grm.write(Tuple((i, n, c)))
            return grm.write(Loop(body=loop_body, arg=tup))

        @scf.region(grm)
        def noop(tup):
            return tup

        tup = grm.write(Tuple((i, n, c)))
        out = grm.write(IfElse(cond=cond, arg=tup, then=loop_start, orelse=noop))
        c = grm.write(lam.Unpack(out, 2))
        return c

    func_body = lam.run_abstraction_pass(grm, func_body)
    return func_body


def test_scf_sum_reduce_loop():
    with TestGrammar(ase.Tape()) as grm:
        func_body = make_sum_reduce_loop(grm)
    print(lam.format(func_body))
    print(ase.pretty_str(func_body))
    # lambar = lambar.simplify()
    # lambar.render_dot().view()
