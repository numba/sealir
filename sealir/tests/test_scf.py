from sealir.lam import LamBuilder
from sealir import ase
from sealir import scf


def test_scf_sum_reduce_loop():
    lambar = LamBuilder()

    @lambar.lam_func
    def func_body(i, x, y):
        n = lambar.expr("mul", x, y)
        cond = lambar.expr("lt", i, n)
        c = lambar.expr("num", 0)

        @scf.region(lambar)
        def loop_start(i, n, c):

            @scf.region(lambar)
            def loop_body(i, n, c):
                c = lambar.expr("add", i, c)
                i = lambar.expr("add", i, lambar.expr("num", 1))
                datatup = lambar.expr("tuple", i, n, c)
                cond = lambar.expr("lt", i, n)
                tup = lambar.expr("tuple", cond, datatup)
                return tup

            tup = lambar.expr("tuple", i, n, c)
            return lambar.expr("scf.dowhile", loop_body, tup)

        @scf.region(lambar)
        def noop(tup):
            return tup

        tup = lambar.expr("tuple", i, n, c)
        out = lambar.expr("scf.switch", cond, tup, loop_start, noop)
        c = lambar.expr("tuple.getitem", out, lambar.expr("num", 2))
        return c

    func_body = lambar.run_abstraction_pass(func_body)
    print(lambar.format(func_body))
    print(func_body.str())

    # lambar = lambar.simplify()
    # lambar.render_dot().view()
