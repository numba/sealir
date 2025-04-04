from sealir import ase, rvsdg
from sealir.eqsat import py_eqsat, rvsdg_eqsat
from sealir.rvsdg import grammar as rg
from sealir.tests.test_rvsdg_egraph_roundtrip import (
    frontend,
    llvm_codegen,
    middle_end,
)


def compiler_pipeline(fn, *, ruleset, verbose=False):
    rvsdg_expr, dbginfo = frontend(fn)

    def define_egraph(egraph, func):
        root = rvsdg_eqsat.GraphRoot(func)
        egraph.let("root", root)

        if verbose:
            print(egraph.extract(root))

        egraph.run(ruleset.saturate())

        if verbose:
            print(egraph.extract(root))

    cost, extracted = middle_end(rvsdg_expr, define_egraph)
    if verbose:
        print("Extracted from EGraph".center(80, "="))
        print("cost =", cost)
        print(rvsdg.format_rvsdg(extracted))

    jt = llvm_codegen(extracted)
    return jt, extracted


def run_test(fn, jt, args):
    res = jt(*args)
    assert res == fn(*args)


def test_const_fold_ifelse():
    """Testing constant folding of if-else when the condition is a constant"""

    ruleset = rvsdg_eqsat.make_rules()

    def check_output_is_argument(extracted, argname: str):
        """Check that the return value is the argument of name `argname`."""
        outportmap = dict((p.name, p.value) for p in extracted.body.ports)
        argportidx = extracted.body.begin.inports.index(argname)
        match outportmap["!ret"]:
            case rg.Unpack(val, idx):
                print(val)
                assert val == extracted.body.begin
                assert idx == argportidx
            case _:
                assert False

    def get_if_else_node(extracted):
        """Get all rvsdg IfElse nodes"""

        def is_if_else(expr):
            match expr:
                case rg.IfElse():
                    return True
            return False

        walker = ase.walk_descendants_depth_first_no_repeat(extracted)
        if_elses = [cur for _, cur in walker if is_if_else(cur)]
        return if_elses

    def ifelse_fold_select_false(a, b):
        c = 0
        if c:
            return a
        else:
            return b

    jt, extracted = compiler_pipeline(
        ifelse_fold_select_false, ruleset=ruleset, verbose=True
    )
    args = (12, 34)
    run_test(ifelse_fold_select_false, jt, args)
    check_output_is_argument(extracted, "b")
    # prove that constant folding of the branch condition eliminated the if-else
    # node
    assert len(get_if_else_node(extracted)) == 0

    def ifelse_fold_select_true(a, b):
        c = 1
        if c:
            return a
        else:
            return b

    jt, extracted = compiler_pipeline(
        ifelse_fold_select_true, ruleset=ruleset, verbose=True
    )
    args = (12, 34)
    run_test(ifelse_fold_select_true, jt, args)
    # prove that constant folding of the branch condition eliminated the if-else
    # node
    assert len(get_if_else_node(extracted)) == 0
    check_output_is_argument(extracted, "a")


def test_forloop_lifting():
    ruleset = rvsdg_eqsat.make_rules() | py_eqsat.make_rules()

    def sum_ints(n):
        c = 0
        for i in range(n):
            c += i
        return c + i

    jt, extracted = compiler_pipeline(sum_ints, ruleset=ruleset, verbose=True)
    run_test(sum_ints, jt, args=(12,))

    # Prove that the lifting to PyForLoop worked.
    walker = ase.walk_descendants_depth_first_no_repeat(extracted)
    forloops = [cur for _, cur in walker if isinstance(cur, rg.PyForLoop)]
    assert len(forloops) == 1
