from __future__ import annotations

import os

from egglog import (
    Expr,
    i64,
    i64Like,
    Vec,
    String,
    StringLike,
    EGraph,
    function,
)

DEBUG = bool(os.environ.get("DEBUG", ""))


class RegionDef(Expr):
    def __init__(self, uid: String, nin: i64Like): ...

    def begin(self) -> InputPorts: ...

    def end(self, outvals: Vec[Term]) -> Term: ...


class InputPorts(Expr):

    def get(self, idx: i64Like) -> Term: ...


class Env(Expr):
    @classmethod
    def nil(cls) -> Env: ...

    def nest(self, ports: Value) -> Env: ...


class Value(Expr):
    @classmethod
    def Param(cls, i: i64Like) -> Value: ...

    @classmethod
    def BoolTrue(cls) -> Value: ...

    @classmethod
    def BoolFalse(cls) -> Value: ...

    @classmethod
    def List(cls, vals: Vec[Value]) -> Value: ...


class Term(Expr):
    @classmethod
    def Branch(
        cls, cond: Term, inputs: Vec[Term], then: Term, orelse: Term
    ) -> Term: ...

    @classmethod
    def Lt(cls, a: Term, b: Term) -> Term: ...

    def getport(self, idx: i64Like) -> Term: ...


@function
def Eval(env: Env, term: Term) -> Value: ...


@function
def EvalMap(env: Env, terms: Vec[Term]) -> Value: ...


@function
def VCons(head: Value, tail: Value) -> Value: ...


@function
def VConsEmpty() -> Value: ...


@function
def VGetPort(val: Value, idx: i64) -> Value: ...


@function
def VBranch(cond: Value, then: Value, orelse: Value) -> Value: ...


@function
def VBinOp(opname: StringLike, lhs: Value, rhs: Value) -> Value: ...


@function
def EnvEnter(env: Env, ins: Vec[Term]) -> Env: ...


def run(root, *, checks=[], assume=None):
    from egglog import eq, ne, rule, rewrite, set_, union

    egraph = EGraph()
    egraph.let("root", root)

    @egraph.register
    def _propagate_RegionDef_from_the_end(
        nin: i64,
        vec_terms: Vec[Term],
        env: Env,
        regionname: String,
    ):
        yield rewrite(
            Eval(env, RegionDef(regionname, nin).end(vec_terms)),
        ).to(EvalMap(env, vec_terms))

    @egraph.register
    def _VBranch(
        env: Env,
        cond: Term,
        input_terms: Vec[Term],
        then_term: Term,
        else_term: Term,
        va: Value,
        vb: Value,
    ):
        # VBranch
        yield rewrite(
            Eval(env, Term.Branch(cond, input_terms, then_term, else_term))
        ).to(
            VBranch(
                Eval(env, cond),
                Eval(EnvEnter(env, input_terms), then_term),
                Eval(EnvEnter(env, input_terms), else_term),
            )
        )
        # Simplify
        yield rewrite(VBranch(Value.BoolTrue(), va, vb)).to(va)
        yield rewrite(VBranch(Value.BoolFalse(), va, vb)).to(vb)

    @egraph.register
    def _eval_ins_get(i: i64, ins: InputPorts, vec_vals: Vec[Value], env: Env):
        yield rewrite(Eval(env.nest(Value.List(vec_vals)), ins.get(i))).to(
            vec_vals[i],
            # given
            i < vec_vals.length(),
        )

    @egraph.register
    def _VGetPort(i: i64, vec_vals: Vec[Value], env: Env, term: Term):
        # VGetPort
        yield rewrite(Eval(env, term.getport(i))).to(
            VGetPort(Eval(env, term), i)
        )
        yield rewrite(VGetPort(Value.List(vec_vals), i)).to(
            vec_vals[i],
            # given
            i < vec_vals.length(),
        )

    @egraph.register
    def _EvalMap_to_VCons_ValueList(
        vec_terms: Vec[Term],
        vec_vals: Vec[Value],
        env: Env,
        val: Value,
    ):
        # EvalMap to VCons and Value.List
        yield rewrite(EvalMap(env, vec_terms)).to(
            VCons(Eval(env, vec_terms[0]), EvalMap(env, vec_terms.remove(0))),
            # given
            vec_terms.length() > i64(0),
        )

        yield rewrite(EvalMap(env, vec_terms)).to(
            VConsEmpty(),
            # given
            eq(vec_terms.length()).to(i64(0)),
        )
        yield rewrite(
            VCons(val, VConsEmpty()),
        ).to(
            Value.List(Vec(val)),
        )
        yield rewrite(
            VCons(val, Value.List(vec_vals)),
        ).to(Value.List(Vec(val).append(vec_vals)))

    @egraph.register
    def _EnvEnter_EvalMap(vec_terms: Vec[Term], env: Env):
        # EnvEnter
        yield rewrite(EnvEnter(env, vec_terms)).to(
            Env.nil().nest(EvalMap(env, vec_terms))
        )

    @egraph.register
    def _VBinOp_Lt(env: Env, ta: Term, tb: Term):
        yield rewrite(Eval(env, Term.Lt(ta, tb))).to(
            VBinOp("Lt", Eval(env, ta), Eval(env, tb))
        )

    if assume is not None:
        assume(egraph)

    saturate(egraph)
    out = egraph.simplify(root, 1)
    print('simplified output'.center(80, '-'))
    print(out)
    print('=' * 80)
    if checks:
        egraph.check(*checks)
    return out


def region_builder(nin: int):
    def wrapped(fn):
        region = RegionDef(fn.__name__, nin)
        ins = region.begin()
        outs = fn(region, ins)
        assert isinstance(outs, (tuple, list))
        return region.end(Vec(*outs))

    return wrapped


def saturate(egraph, limit=1_000):
    if DEBUG:
        egraph.saturate(limit=limit, n_inline_leaves=2)
    else:
        # workaround egraph.saturate() is always opening the viz.
        i = 0
        while egraph.run(1).updated and i < limit:
            i += 1


def test_straight_line_basic():
    from egglog import eq

    @region_builder(5)
    def main(region, ins):
        return list(map(ins.get, range(5)))

    env = Env.nil()
    env = env.nest(Value.List(Vec(*map(Value.Param, range(5)))))
    root = Eval(env, main)

    checks = [
        eq(root).to(Value.List(Vec(*map(Value.Param, range(5))))),
    ]
    run(root, checks=checks)


def test_max_if_else():
    from egglog import eq

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
        ifthen = Term.Branch(lt, Vec(a, b), if_then, or_else)

        # Return
        return [ifthen.getport(0)]

    # Eval with Env
    env = Env.nil()
    env = env.nest(Value.List(Vec(Value.Param(0), Value.Param(1))))
    root = Eval(env, main)

    def assume(egraph: EGraph):
        @egraph.register
        def facts(val: Value):
            from egglog import eq, rule, union

            yield rule(
                eq(val).to(VBinOp("Lt", Value.Param(0), Value.Param(1)))
            ).then(union(val).with_(Value.BoolTrue()))


    checks = [
        eq(root).to(Value.List(Vec(Value.Param(1)))),
    ]
    run(root, checks=checks, assume=assume)

