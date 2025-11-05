from egglog import EGraph

from sealir import rvsdg
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import (
    EGraphToRVSDG,
    egraph_extraction,
    get_graph_root,
)
from sealir.llvm_pyapi_backend import llvm_codegen


def frontend(fn):
    """
    Frontend code is all encapsulated in sealir.rvsdg.restructure_source
    """
    rvsdg_expr, dbginfo = rvsdg.restructure_source(fn)

    return rvsdg_expr, dbginfo


def middle_end(rvsdg_expr, apply_to_egraph, cost_model=None, stats=None):
    """The middle end encode the RVSDG into a EGraph to apply rewrite rules.
    After that, it is extracted back into RVSDG.
    """
    # Convert to egraph
    memo = egraph_conversion(rvsdg_expr)

    func = memo[rvsdg_expr]

    egraph = EGraph()
    apply_to_egraph(egraph, func)

    # Extraction
    extraction = egraph_extraction(egraph, cost_model=cost_model, stats=stats)
    result = extraction.extract_graph_root()
    expr = result.extract_sexpr(rvsdg_expr, EGraphToRVSDG)
    return result.cost, expr


def compiler_pipeline(fn, args, *, verbose=False):
    rvsdg_expr, dbginfo = frontend(fn)

    # Middle end
    def display_egraph(egraph: EGraph, func):
        # For now, the middle end is just an identity function that exercise
        # the encoding into and out of egraph.
        root = GraphRoot(func)
        egraph.let("root", root)
        if verbose:
            # For inspecting the egraph
            egraph.display()  # opens a webpage when run

        return root

    stats = {}
    cost, extracted = middle_end(rvsdg_expr, display_egraph, stats=stats)

    print("Extracted from EGraph".center(80, "="))
    print("cost =", cost)
    print(rvsdg.format_rvsdg(extracted))
    print("stats:", stats)

    jt = llvm_codegen(rvsdg_expr)
    res = jt(*args)

    print("JIT: output".center(80, "="))
    print(res)

    assert res == fn(*args)


def test_sum_ints():
    def sum_ints(n):
        c = 0
        for i in range(n):
            c += i
        return c

    compiler_pipeline(sum_ints, (12,), verbose=False)


def test_max_two():
    def max_if_else(x, y):
        if x > y:
            return x
        else:
            return y

    compiler_pipeline(max_if_else, (1, 2), verbose=False)
    compiler_pipeline(max_if_else, (3, 2), verbose=False)
