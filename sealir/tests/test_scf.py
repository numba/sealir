from sealir import ase, scf
from sealir.lam import LamBuilder


def make_sum_reduce_loop(lambar):
    @lambar.lam_func
    def func_body(i, x, y):
        n = lambar.expr("mul", x, y)
        cond = lambar.expr("lt", i, n)
        c = lambar.expr("int", 0)

        @scf.region(lambar)
        def loop_start(i, n, c):

            @scf.region(lambar)
            def loop_body(i, n, c):
                c = lambar.expr("add", i, c)
                i = lambar.expr("add", i, lambar.expr("int", 1))
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
        c = lambar.expr("tuple.getitem", out, lambar.expr("int", 2))
        return c

    func_body = lambar.run_abstraction_pass(func_body)
    return func_body


def test_scf_sum_reduce_loop():
    lambar = LamBuilder()
    func_body = make_sum_reduce_loop(lambar)
    print(lambar.format(func_body))
    print(ase.pretty_str(func_body))
    # lambar = lambar.simplify()
    # lambar.render_dot().view()
