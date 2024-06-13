from sealir import ase
from sealir.rewriter import TreeRewriter


def test_rewrite():

    class RewriteCalcMachine(TreeRewriter[ase.Expr]):
        def rewrite_add(self, lhs, rhs):
            [x] = lhs.args
            [y] = rhs.args
            return ase.expr("num", x + y)

        def rewrite_sub(self, lhs, rhs):
            [x] = lhs.args
            [y] = rhs.args
            return ase.expr("num", x - y)

        def rewrite_mul(self, lhs, rhs):
            [x] = lhs.args
            [y] = rhs.args
            return ase.expr("num", x * y)

    with ase.Tree():
        a = ase.expr("num", 123)
        b = ase.expr("num", 321)
        c = ase.expr("add", a, a)
        d = ase.expr("sub", c, b)
        e = ase.expr("mul", b, d)

    calc = RewriteCalcMachine()
    e.apply_bottomup(calc)
    print(e.tree.dump())
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

    mintree = ase.Tree()
    new_reduced = reduced.copy_tree_into(mintree)
    print(mintree.dump())
    assert new_reduced.str() == reduced.str()
    assert len(mintree) == 1
