# mypy: disable-error-code="empty-body"

from __future__ import annotations

import os

from egglog import EGraph, eq

from sealir.eqsat.rvsdg_eqsat import (
    Debug,
    Env,
    Eval,
    LoopIndVar,
    LVAnalysis,
    RegionDef,
    Term,
    Value,
    VBinOp,
    VSum,
    make_rules,
    termlist,
    valuelist,
)


def read_env(v: str):
    if v:
        return int(v)
    else:
        return 0


DEBUG = read_env(os.environ.get("DEBUG", ""))


def run(root, *, checks=[], assume=None, debug_points=None):
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

    ruleset = make_rules()

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


def region_builder(nin: int):
    def wrapped(fn):
        region = RegionDef(fn.__name__, nin)
        ins = region.begin()
        outs = fn(region, ins)
        assert isinstance(outs, (tuple, list))
        return region.end(termlist(*outs))

    return wrapped


def test_straight_line_basic():
    @region_builder(5)
    def main(region, ins):
        return list(map(ins.get, range(5)))

    env = Env.nil()
    env = env.nest(valuelist(*map(Value.Param, range(5))))
    root = Eval(env, main)

    checks = [
        eq(root).to(valuelist(*map(Value.Param, range(5))).toValue()),
    ]
    run(root, checks=checks)


def test_max_if_else():
    fn = RegionDef("main", 2)

    @region_builder(2)
    def main(region, ins):
        ctx_body = fn.begin()
        a = ctx_body.get(0)
        b = ctx_body.get(1)

        lt = Term.Lt(a, b)

        # Then
        @region_builder(2)
        def if_then(region, ins):
            a = ins.get(0)
            b = ins.get(1)
            return [b]

        # Else
        @region_builder(2)
        def or_else(region, ins):
            a = ins.get(0)
            b = ins.get(1)
            return [a]

        # Do Branch
        ifthen = Term.Branch(lt, termlist(a, b), if_then, or_else)

        # Return
        return [ifthen.getPort(0)]

    # Eval with Env
    env = Env.nil()
    env = env.nest(valuelist(Value.ConstI64(0), Value.ConstI64(1)))
    root = Eval(env, main)

    checks = [
        eq(root).to(Value.ConstI64(1)),
    ]
    run(root, checks=checks)


def test_loop_analysis():
    debug_points = {}

    @region_builder(2)
    def loop(region, ins):
        a, b = ins.get(0), ins.get(1)

        debug_points["a"] = LVAnalysis(Debug.ValueOf(a))

        na = Term.Add(a, Term.LiteralI64(1))
        cond = Term.Lt(na, b)
        return [cond, na, b]

    @region_builder(2)
    def main(region, ins):
        return [Term.Loop(termlist(ins.get(0), ins.get(1)), loop)]

    # Eval with Env
    env = Env.nil()
    env = env.nest(valuelist(Value.Param(0), Value.Param(1)))
    root = Eval(env, main)

    run(
        root,
        checks=[
            eq(debug_points["a"]).to(
                LoopIndVar(
                    "Add", Value.Param(0), Value.Param(1), Value.ConstI64(1)
                )
            ),
        ],
        debug_points=debug_points,
    )


def test_sum_loop():
    # Equivalent source:
    #     In [1]: def thesum(init, n):
    #    ...:     c = init
    #    ...:     for i in range(n):
    #    ...:         c += i
    #    ...:     return c
    # Target:
    #   c = sum(range(n))

    debug_points = {}

    @region_builder(2)
    def main(region, ins):
        init = ins.get(0)
        n = ins.get(1)
        c = init
        i = Term.LiteralI64(0)

        @region_builder(2)
        def body(region, ins):
            i = ins.get(0)
            n = ins.get(1)
            c = ins.get(2)

            debug_points["i"] = Debug.ValueOf(i)

            c = Term.Add(c, i)
            i = Term.Add(i, Term.LiteralI64(1))
            lt = Term.Lt(i, n)
            return [lt, i, n, c]

        # Do Loop
        loop = Term.Loop(termlist(i, n, c), body)

        # Return
        return [loop.getPort(1), loop.getPort(2)]  # sum(range())

    # Eval with Env
    env = Env.nil()
    env = env.nest(valuelist(Value.Param(0), Value.Param(1)))
    root = Eval(env, main)

    checks = [
        eq(root).to(
            valuelist(
                Value.Param(1),
                VBinOp(
                    "Add",
                    Value.Param(0),
                    VSum(
                        Value.ConstI64(0),
                        Value.Param(1),
                        Value.ConstI64(1),
                    ),
                ),
            ).toValue()
        ),
    ]
    run(root, checks=checks, debug_points=debug_points)
