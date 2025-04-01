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
    Region,
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
    limit = 1000
    if DEBUG:
        # Borrowed from egraph.saturate(schedule)

        from pprint import pprint

        from egglog.visualizer_widget import VisualizerWidget

        def to_json() -> str:
            return egraph._serialize().to_json()

        egraphs = [to_json()]
        c = 0
        while c < limit:
            report = egraph.run(ruleset)
            reports.append(report)
            egraphs.append(to_json())
            # pprint({k: v for k, v in report.num_matches_per_rule.items()
            #         if v > 0})
            c += 1
            if not report.updated:
                break
        VisualizerWidget(egraphs=egraphs).display_or_open()

    else:
        report = egraph.run(ruleset.saturate())
        reports.append(report)
    return reports


def region_builder(*args, arity: int | None = None):
    if args:
        assert arity is None
    else:
        assert arity is not None
        args = tuple([Term.Param(i) for i in range(arity)])

    def wrapped(fn):
        nin = len(args)
        in_names = " ".join(map(str, range(nin)))
        region = Region(fn.__name__, in_names, termlist(*args))
        outs = fn(region, region.begin())
        assert isinstance(outs, (tuple, list))
        out_names = " ".join(map(str, range(nin)))
        return Term.RegionEnd(region, out_names, termlist(*outs))

    return wrapped


def test_straight_line_basic():
    @region_builder(arity=5)
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
    @region_builder(arity=2)
    def main(region, ins):
        a = ins.get(0)
        b = ins.get(1)

        lt = Term.Lt(a, b)

        # Then
        @region_builder(a, b)
        def if_then(region, ins):
            a = ins.get(0)
            b = ins.get(1)
            return [b]

        # Else
        @region_builder(a, b)
        def or_else(region, ins):
            a = ins.get(0)
            b = ins.get(1)
            return [a]

        # Do Branch
        ifelse = Term.IfElse(lt, if_then, or_else)

        # Return
        return [ifelse.getPort(0)]

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

    @region_builder(arity=2)
    def main(region, ins):
        a = ins.get(0)
        b = ins.get(1)

        @region_builder(a, b)
        def loop(region, ins):
            a, b = ins.get(0), ins.get(1)

            debug_points["a"] = LVAnalysis(Debug.ValueOf(a))

            na = Term.Add(a, Term.LiteralI64(1))
            cond = Term.Lt(na, b)
            return [cond, na, b]

        return [Term.Loop(loop, loopvar="0")]

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
    #   c = sum(range(n)) + init

    debug_points = {}

    @region_builder(arity=2)
    def main(region, ins):
        init = ins.get(0)
        n = ins.get(1)
        c = init
        i = Term.LiteralI64(0)

        @region_builder(i, n, c)
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
        loop = Term.Loop(body, loopvar="0")

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
