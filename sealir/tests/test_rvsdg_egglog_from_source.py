from egglog import EGraph

from sealir import ase, rvsdg
from sealir.eqsat import py_eqsat, rvsdg_eqsat
from sealir.eqsat.rvsdg_convert import egraph_conversion
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


def test_for_range_lifting():
    from egglog import Vec, function, i64, rule, ruleset, union

    from sealir.eqsat.py_eqsat import (
        Py_Call,
        Py_ForLoop,
        Py_LoadGlobal,
        Term,
        TermList,
    )
    from sealir.eqsat.rvsdg_eqsat import DynInt, Term, wildcard

    @function
    def My_ForRange(
        range_args: TermList,
        indvar_arg_idx: DynInt,
        iterlast_arg_idx: DynInt,
        body: Term,
        operands: TermList,
    ) -> Term: ...

    _wc = wildcard

    @ruleset
    def ruleset_for_range_lift(
        iter_arg_idx: i64,
        body: Term,
        forloop: Term,
        operands: Vec[Term],
        iter_res: Term,
        iter_args: Vec[Term],
        range_res: Term,
        range_args: TermList,
        indvar_arg_idx: DynInt,
        iterlast_arg_idx: DynInt,
    ):
        # Lift Py_ForLoop into My_ForRange
        yield rule(
            forloop
            == Py_ForLoop(
                iter_arg_idx=DynInt(iter_arg_idx),
                indvar_arg_idx=indvar_arg_idx,
                iterlast_arg_idx=iterlast_arg_idx,
                body=body,
                operands=TermList(operands),
            ),
            iter_res == operands[iter_arg_idx],
            iter_res
            == Py_Call(
                func=Py_LoadGlobal(io=_wc(Term), name="iter"),
                io=_wc(Term),
                args=TermList(iter_args),
            ).getPort(1),
            range_res == iter_args[0],
            range_res
            == Py_Call(
                func=Py_LoadGlobal(io=_wc(Term), name="range"),
                io=_wc(Term),
                args=range_args,
            ).getPort(1),
        ).then(
            union(forloop).with_(
                My_ForRange(
                    range_args=range_args,
                    indvar_arg_idx=indvar_arg_idx,
                    iterlast_arg_idx=iterlast_arg_idx,
                    body=body,
                    operands=TermList(operands),
                )
            ),
            union(forloop).with_(Term.LiteralStr("dbg_for_range")),
        )

    rs_schedule = (
        rvsdg_eqsat.make_rules()
        | py_eqsat.make_rules()
        | ruleset_for_range_lift
    )

    def sum_ints(n):
        c = 0
        for i in range(n):
            c += i
        return c + i

    fn = sum_ints

    rvsdg_expr, dbginfo = frontend(fn)
    print(rvsdg.format_rvsdg(rvsdg_expr))

    # Convert to egraph
    memo = egraph_conversion(rvsdg_expr)

    func = memo[rvsdg_expr]

    egraph = EGraph()

    root = rvsdg_eqsat.GraphRoot(func)
    egraph.let("root", root)

    egraph.run(rs_schedule.saturate())
    egraph.check(Term.LiteralStr("dbg_for_range"))
