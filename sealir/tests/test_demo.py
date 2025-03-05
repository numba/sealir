# mypy: disable-error-code="empty-body"

from __future__ import annotations

import os
import time

import numpy as np
from egglog import EGraph, String, eq, var

from sealir import ase, rvsdg
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import (
    DoPartialEval,
    Env,
    Eval,
    GraphRoot,
    PartialEvaluated,
    Term,
    Value,
    make_rules,
    valuelist,
)
from sealir.eqsat.rvsdg_extract import egraph_extraction
from sealir.llvm_pyapi_backend import llvm_codegen


def read_env(v: str):
    if v:
        return int(v)
    else:
        return 0


DEBUG = read_env(os.environ.get("DEBUG", ""))


def run(
    root, *extra_statements, checks=[], assume=None, debug_points=None
) -> EGraph:
    """
    Example assume
    --------------

    > def assume(egraph: EGraph):
    >     @egraph.register
    >     def facts(val: Value):
    >         from egglog import eq, rule, union
    >         yield rule(
    >             eq(val).to(VBinOp("Lt", Value.ConstI64(0), Value.ConstI64(1)))
    >         ).then(union(val).with_(Value.BoolTrue()))

    """

    egraph = EGraph()  # save_egglog_string=True)
    egraph.let("root", root)
    for i, stmt in enumerate(extra_statements):
        egraph.let(f"stmt{i}", stmt)
    # egraph.display()
    ruleset = make_rules()

    if debug_points:
        for k, v in debug_points.items():
            egraph.let(f"debug_point_{k}", v)

    if assume is not None:
        assume(egraph)

    ts = time.time()
    saturate(egraph, ruleset)
    te = time.time()
    print("saturation time", te - ts)

    ts = time.time()
    out = egraph.simplify(root, 1)
    te = time.time()
    print("extraction time", te - ts)
    # print(egraph.as_egglog_string)
    print("simplified output".center(80, "-"))
    print(out)
    print("=" * 80)
    # egraph.display()
    if checks:
        try:
            egraph.check(*checks)
        except Exception:
            if debug_points:
                for k, v in debug_points.items():
                    print(f"debug {k}".center(80, "-"))
                    for each in egraph.extract_multiple(v, 5):
                        print(each)
                        print("-=-")

            raise
    return egraph


def saturate(egraph: EGraph, ruleset):
    reports = []
    if DEBUG:
        # Borrowed from egraph.saturate(schedule)

        from pprint import pprint

        from egglog.visualizer_widget import VisualizerWidget

        def to_json() -> str:
            return egraph._serialize().to_json()

        egraphs = [to_json()]
        while True:
            report = egraph.run(ruleset)
            reports.append(report)
            egraphs.append(to_json())
            # pprint({k: v for k, v in report.num_matches_per_rule.items()
            #         if v > 0})
            if not report.updated:
                break
        VisualizerWidget(egraphs=egraphs).display_or_open()

    else:
        report = egraph.run(ruleset.saturate())
        reports.append(report)
    return reports


def test_geglu_tanh_approx():

    def float32(num):
        return np.float32(num)

    dt = float32

    def tanh(x):
        return np.tanh(x)

    def sqrt(x):
        return np.sqrt(x)

    pi = np.pi

    def udt(a):
        result = (
            dt(0.5)
            * a
            * (dt(1) + tanh(sqrt(dt(2) / dt(pi)) * (a + dt(0.044715) * a**3)))
        )
        return result

    rvsdg_expr, dbginfo = rvsdg.restructure_source(udt)
    print(rvsdg.format_rvsdg(rvsdg_expr))

    memo = egraph_conversion(rvsdg_expr)

    egfunc = memo[rvsdg_expr]

    # Eval with Env
    env = Env.nil()
    env = env.nest(
        # Argument list
        valuelist(Value.IOState(), Value.Param(0))
    )
    root = GraphRoot(egfunc)

    extra_statements = [
        DoPartialEval(env, egfunc),
    ]

    checks = [
        # eq(root).to(
        #     GraphRoot(
        #         Term.Func(
        #             str(rvsdg_expr._handle),
        #             var("fname", String),
        #             PartialEvaluated(Value.ConstI64(134)),
        #         )
        #     )
        # ),
    ]

    egraph = run(root, *extra_statements, checks=checks)
    # Extraction
    cost, extracted = egraph_extraction(
        egraph,
        rvsdg_expr,
    )
    print(ase.as_tuple(extracted, depth=5))
    print("cost =", cost)
    print(rvsdg.format_rvsdg(extracted))

    ns = {
        "dt": dt,
        "pi": pi,
        "sqrt": sqrt,
        "tanh": tanh,
    }

    cg = llvm_codegen(extracted, ns)

    arg = 0.315
    print(cg(0.315))
    print(udt(arg))
