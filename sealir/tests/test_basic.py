from sealir import ase


def test_bottom():
    tape = ase.Tape()
    assert tape._read_token(0) is None


def test_basic():
    with ase.Tape() as tape:
        a = ase.expr("num", 1)
        b = ase.expr("num", 2)
        c = ase.expr("add", a, b)
        d = ase.expr("sub", a, a)

    assert c.str() == "(add (num 1) (num 2))"
    assert c.head == "add"
    assert c.args[0].head == "num"
    assert c.args[0].args == (1,)
    assert c.args[1].head == "num"
    assert c.args[1].args == (2,)

    parent_of_a = list(a.walk_parents())
    assert parent_of_a[0] == c
    assert parent_of_a[0] != a
    assert parent_of_a[0] != b
    assert parent_of_a[1] == d

    for p in parent_of_a:
        assert p.contains(a)


def test_copy_tree():
    with ase.Tape() as tape:
        ase.expr("num", 0)
        a = ase.expr("num", 1)
        b = ase.expr("num", 2)
        ase.expr("num", 3)
        ase.expr("add", a, b)
        d = ase.expr("sub", a, a)
        e = ase.expr("mul", b, d)

    new_tree = ase.Tape()
    new_e = e.copy_tree_into(new_tree)

    assert len(new_tree._heap) < len(tape._heap)
    assert len(new_tree._tokens) < len(tape._tokens)

    assert new_e != e
    assert new_e.str() == e.str()


def test_apply_bottomup():
    with ase.Tape():
        a = ase.expr("num", 1)
        b = ase.expr("num", 2)
        c = ase.expr("sub", a, a)
        d = ase.expr("add", c, b)
        e = ase.expr("mul", b, c)
        f = ase.expr("div", e, d)

    buffer = []

    class BufferVisitor(ase.TreeVisitor):
        def visit(self, expr: ase.Expr):
            buffer.append(expr)

    bv = BufferVisitor()
    e.apply_bottomup(bv)

    # It is expected the visitor will see every S-expr in the Tape.
    # Regardless of whether it is reachable from the root S-expr.
    # But it will not go further then that (exclude `f`)
    assert buffer == [a, b, c, d, e]


def test_calculator():
    with ase.Tape():
        a = ase.expr("num", 123)
        b = ase.expr("num", 321)
        c = ase.expr("add", a, a)
        d = ase.expr("sub", c, b)
        e = ase.expr("mul", b, d)

    class Calc(ase.TreeVisitor):
        def __init__(self):
            self.memo = {}

        def visit(self, expr: ase.Expr):
            if expr.head == "num":
                self.memo[expr] = expr.args[0]
            elif expr.head == "add":
                self.memo[expr] = (
                    self.memo[expr.args[0]] + self.memo[expr.args[1]]
                )
            elif expr.head == "sub":
                self.memo[expr] = (
                    self.memo[expr.args[0]] - self.memo[expr.args[1]]
                )
            elif expr.head == "mul":
                self.memo[expr] = (
                    self.memo[expr.args[0]] * self.memo[expr.args[1]]
                )
            else:
                raise AssertionError("unknown op")

    calc = Calc()
    e.apply_bottomup(calc)
    result = calc.memo[e]

    def expected():
        a = 123
        b = 321
        c = a + a
        d = c - b
        e = b * d
        return e

    assert expected() == result
