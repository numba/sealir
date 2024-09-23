from sealir import ase
from sealir.rewriter import TreeRewriter


def test_rewrite():

    class RewriteCalcMachine(TreeRewriter[ase.SExpr]):
        def rewrite_add(self, orig, lhs, rhs):
            tp = orig._tape
            [x] = lhs._args
            [y] = rhs._args
            return tp.expr("num", x + y)

        def rewrite_sub(self, orig, lhs, rhs):
            tp = orig._tape
            [x] = lhs._args
            [y] = rhs._args
            return tp.expr("num", x - y)

        def rewrite_mul(self, orig, lhs, rhs):
            tp = orig._tape
            [x] = lhs._args
            [y] = rhs._args
            return tp.expr("num", x * y)

    with ase.Tape() as tp:
        a = tp.expr("num", 123)
        b = tp.expr("num", 321)
        c = tp.expr("add", a, a)
        d = tp.expr("sub", c, b)
        e = tp.expr("mul", b, d)

    calc = RewriteCalcMachine()
    with tp:
        ase.apply_bottomup(e, calc)
    print(e._tape.dump())
    reduced = calc.memo[e]
    [result] = reduced._args

    def expected():
        a = 123
        b = 321
        c = a + a
        d = c - b
        e = b * d
        return e

    assert expected() == result

    mintree = ase.Tape()
    new_reduced = ase.copy_tree_into(reduced, mintree)
    print(mintree.dump())
    assert ase.pretty_str(new_reduced) == ase.pretty_str(reduced)
    assert len(mintree) == 1
