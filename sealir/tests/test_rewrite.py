from sealir import ase
from sealir.rewriter import TreeRewriter


def test_rewrite():

    class RewriteCalcMachine(TreeRewriter[ase.Expr]):
        def rewrite_add(self, orig, lhs, rhs):
            tp = orig.tape
            [x] = lhs.args
            [y] = rhs.args
            return tp.expr("num", x + y)

        def rewrite_sub(self, orig, lhs, rhs):
            tp = orig.tape
            [x] = lhs.args
            [y] = rhs.args
            return tp.expr("num", x - y)

        def rewrite_mul(self, orig, lhs, rhs):
            tp = orig.tape
            [x] = lhs.args
            [y] = rhs.args
            return tp.expr("num", x * y)

    with ase.Tape() as tp:
        a = tp.expr("num", 123)
        b = tp.expr("num", 321)
        c = tp.expr("add", a, a)
        d = tp.expr("sub", c, b)
        e = tp.expr("mul", b, d)

    calc = RewriteCalcMachine()
    with tp:
        e.apply_bottomup(calc)
    print(e.tape.dump())
    reduced = calc.memo[e]
    [result] = reduced.args

    def expected():
        a = 123
        b = 321
        c = a + a
        d = c - b
        e = b * d
        return e

    assert expected() == result

    mintree = ase.Tape()
    new_reduced = reduced.copy_tree_into(mintree)
    print(mintree.dump())
    assert new_reduced.str() == reduced.str()
    assert len(mintree) == 1
