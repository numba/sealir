# mypy: disable-error-code="empty-body"

from __future__ import annotations

import os

from egglog import EGraph, eq

from sealir import ase, rvsdg
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import (
    Env,
    Eval,
    GraphRoot,
    PartialEvaluated,
    Value,
    make_rules,
    valuelist,
)
from sealir.eqsat.rvsdg_extract import egraph_extraction


def read_env(v: str):
    if v:
        return int(v)
    else:
        return 0


DEBUG = read_env(os.environ.get("DEBUG", ""))


def run(root, *, checks=[], assume=None, debug_points=None) -> EGraph:
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

    ruleset = make_rules(extraction=True)

    if debug_points:
        for k, v in debug_points.items():
            egraph.let(f"debug_point_{k}", v)

    if assume is not None:
        assume(egraph)

    saturate(egraph, ruleset)
    out = egraph.extract(root)
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


def test_max_if_else_from_source():
    def udt(a, b):
        if a < b:
            c = b
        else:
            c = a
        return c

    rvsdg_expr, dbginfo = rvsdg.restructure_source(udt)
    print(rvsdg.format_rvsdg(rvsdg_expr))

    memo = egraph_conversion(rvsdg_expr)

    egraphed = memo[rvsdg_expr]

    # Eval with Env
    env = Env.nil()
    env = env.nest(
        # Argument list
        valuelist(Value.IOState(), Value.ConstI64(2), Value.ConstI64(134))
    )
    root = GraphRoot(PartialEvaluated(Eval(env, egraphed)))

    checks = [
        eq(root).to(GraphRoot(PartialEvaluated(Value.ConstI64(134)))),
    ]
    egraph = run(root, checks=checks)

    # Extraction
    extracted = egraph_extraction(
        egraph,
    )
    print(extracted)
