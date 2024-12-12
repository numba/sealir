# mypy: disable-error-code="empty-body"

from __future__ import annotations

import os
from typing import Callable
from functools import partial

from egglog import (
    Expr,
    i64,
    i64Like,
    Bool,
    BoolLike,
    Vec,
    Set,
    String,
    StringLike,
    EGraph,
    function,
)
from egglog import eq, ne, rule, rewrite, set_, union, ruleset

DEBUG = bool(os.environ.get("DEBUG", ""))


class RegionDef(Expr):
    def __init__(self, uid: String, nin: i64Like): ...

    def begin(self) -> InputPorts: ...

    def end(self, outvals: TermList) -> Term: ...


class InputPorts(Expr):

    def get(self, idx: i64Like) -> Term: ...


class Env(Expr):
    @classmethod
    def nil(cls) -> Env: ...

    def nest(self, ports: ValueList) -> Env: ...


class Value(Expr):
    @classmethod
    def Param(cls, i: i64Like) -> Value: ...

    @classmethod
    def BoolTrue(cls) -> Value: ...

    @classmethod
    def BoolFalse(cls) -> Value: ...

    @classmethod
    def ConstI64(cls, val: i64Like) -> Value: ...

    def toList(self) -> ValueList: ...

    def __or__(self, rhs: Value) -> Value: ...


class ValueList(Expr):
    def __init__(self, vals: Vec[Value]): ...

    def append(self, vs: ValueList) -> ValueList: ...

    def toValue(self) -> Value: ...

    def toSet(self) -> Set[Value]: ...

    def map(self, fn: Callable[[Value], Value]) -> ValueList: ...

    @classmethod
    def Merge(cls,
              merge_fn: Callable[[Value, Value], Value],
              vas: ValueList,
              vbs: ValueList) -> ValueList: ...



class Term(Expr):
    @classmethod
    def Branch(
        cls, cond: Term, inputs: TermList, then: Term, orelse: Term
    ) -> Term: ...
    @classmethod
    def Loop(cls, inputs: TermList, body: Term) -> Term: ...

    @classmethod
    def Lt(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def Add(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def LiteralI64(cls, val: i64) -> Term: ...

    def getPort(self, idx: i64Like) -> Term: ...


class TermList(Expr):
    def __init__(self, terms: Vec[Term]): ...

    # def map(self, fn: Callable[[Term], Term]) -> TermList: ...

    def mapValue(self: TermList, fn: Callable[[Term], Value]) -> ValueList: ...


def termlist(*args: Term) -> TermList:
    return TermList(Vec(*args))


@function
def Eval(env: Env, term: Term) -> Value: ...


@function
def EvalMap(env: Env, terms: TermList) -> ValueList: ...

@function
def VFix(v: Value) -> Value: ...

@function
def VGetPort(vs: ValueList, idx: i64) -> Value: ...


@function
def VBranch(cond: Value, then: Value, orelse: Value) -> Value: ...

@function(unextractable=True)
def _LoopTemp(outs: ValueList) -> Value: ...

@function
def _LoopBack(phis: ValueList, body: Term) -> ValueList: ...

@function
def _LoopDropCond(outs: Value) -> ValueList: ...

@function
def _LoopOutputs(v: Value) -> ValueList: ...

@function
def _LoopPhiOf(phi: Value, loopback: Value) -> Value: ...
@function
def _MapLoopPhiOf(phis: ValueList, loopbacks: ValueList) -> ValueList: ...

@function
def VLoop(phis: ValueList, body: Value) -> Value: ...


@function
def VBinOp(opname: StringLike, lhs: Value, rhs: Value) -> Value: ...


@function
def VMergeList(lhs: Value, rhs: Value) -> Value: ...


@function
def EnvEnter(env: Env, ins: TermList) -> Env: ...


@function
def EnvLoop(vs: ValueList) -> Env: ...


@function
def PhiMap(left: ValueList, right: ValueList) -> ValueList: ...

@function
def VPhi(a: Value, b: Value) -> Value: ...


class LVA(Expr):
    """Loop Variable Analysis"""
    ...

@function
def LVAnalysis(v: Value) -> LVA: ...

@function
def LoopIncremented(op: StringLike, phi: Value, init: Value, step: Value) -> LVA: ...
@function
def LoopIndVar(op: StringLike, start: Value, stop: Value, step: Value) -> LVA: ...
@function
def LoopAccumIndVar(op: StringLike, start: Value, stop: Value, step: Value) -> LVA: ...

@function
def IsLoopInVariant(v: Value) -> Bool: ...

@function
def VSum(vstart: Value, vstop: Value, vstep: Value) -> Value: ...

@ruleset
def _propagate_RegionDef_from_the_end(
    nin: i64,
    vec_terms: Vec[Term],
    env: Env,
    regionname: String,
):
    yield rewrite(
        Eval(env, RegionDef(regionname, nin).end(TermList(vec_terms))),
    ).to(EvalMap(env, TermList(vec_terms)).toValue())


@ruleset
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
        Eval(
            env, Term.Branch(cond, TermList(input_terms), then_term, else_term)
        )
    ).to(
        VBranch(
            Eval(env, cond),
            Eval(EnvEnter(env, TermList(input_terms)), then_term),
            Eval(EnvEnter(env, TermList(input_terms)), else_term),
        )
    )
    # Simplify
    yield rewrite(VBranch(Value.BoolTrue(), va, vb)).to(va)
    yield rewrite(VBranch(Value.BoolFalse(), va, vb)).to(vb)


@ruleset
def _VLoop(
    env: Env,
    input_terms: TermList,
    body_term: Term,
    vec_va: Vec[Value],
    vec_vb: Vec[Value],
    vl: ValueList,
    vl2: ValueList,
    phis: ValueList,
    va: Value,
    vb: Value,
):
    # VLoop
    yield rewrite(
        Eval(env, Term.Loop(input_terms, body_term)),
    ).to(
        _LoopTemp(_LoopBack(EvalMap(env, input_terms), body_term))
    )

    yield rule(
        eq(vl).to(_LoopBack(phis, body_term)),
    ).then(
        union(_LoopTemp(vl)).with_(
            VLoop(_pm:=PhiMap(phis, vl),
                  _inner:=Eval(EnvLoop(_pm), body_term))
        ),
        union(vl).with_(
            _LoopDropCond(_inner)
        )
    )

    yield rewrite(VLoop(vl, va)).to(_LoopOutputs(va).toValue())

    # EnvLoop
    yield rewrite(
        EnvLoop(phis)
    ).to(
        Env.nil().nest(phis)
    )
    # _LoopDropCond
    yield rewrite(
        _LoopDropCond(ValueList(vec_va).toValue())
    ).to(
        # Drop the loop condition
        ValueList(vec_va.remove(0))
    )

    # _LoopOutputs
    yield rewrite(
        _LoopOutputs(ValueList(vec_va).toValue())
    ).to(
        # Drop the loop condition
        ValueList(vec_va.remove(0)).map(VFix)
    )

    # PhiMap
    yield rewrite(
        PhiMap(vl, vl2)
    ).to(
        ValueList.Merge(VPhi, vl, vl2)
    )

    # Phi
    yield rewrite(VPhi(va, vb)).to(va|vb)


@ruleset
def _eval_ins_get(i: i64, ins: InputPorts, vec_vals: Vec[Value], env: Env):
    yield rewrite(Eval(env.nest(ValueList(vec_vals)), ins.get(i))).to(
        vec_vals[i],
        # given
        i < vec_vals.length(),
    )


@ruleset
def _VGetPort(i: i64, vec_vals: Vec[Value], env: Env, term: Term):
    # VGetPort
    yield rewrite(Eval(env, term.getPort(i))).to(
        VGetPort(Eval(env, term).toList(), i)
    )
    yield rewrite(VGetPort(ValueList(vec_vals), i)).to(
        vec_vals[i],
        # given
        i < vec_vals.length(),
    )

@ruleset
def _Value_rules(a: Value, b: Value):
    # __or__ associativity
    yield rewrite( a | b ).to( b | a)

    # merge phis
    yield rewrite( a | a ).to( a )
    # merge loop back phis
    yield rule(
        eq(a).to(a | b)
    ).then(
        union(b).with_(a),
        set_(IsLoopInVariant(a)).to(Bool(True)),
    )




@ruleset
def _ValueList_rules(
    vs1: Vec[Value],
    vs2: Vec[Value],
    vl: ValueList,
    vl2: ValueList,
    fn: Callable[[Value], Value],
    merge_fn: Callable[[Value, Value], Value],
):
    yield rewrite(ValueList(vs1).append(ValueList(vs2))).to(
        ValueList(vs1.append(vs2))
    )
    # Simplify ValueList
    yield rewrite(vl.toValue().toList()).to(vl)
    yield rewrite(
        ValueList(vs1).toValue(),
    ).to(
        vs1[0],
        # given
        eq(i64(1)).to(vs1.length()),
    )
    # map
    yield rewrite(ValueList(vs1).map(fn)).to(
        valuelist(),
        # given
        eq(i64(0)).to(vs1.length()),
    )
    yield rewrite(ValueList(vs1).map(fn)).to(
        valuelist(fn(vs1[0])).append(ValueList(vs1.remove(0)).map(fn)),
        # given
        vs1.length() > i64(0),
    )
    # Merge
    yield rewrite(
        ValueList.Merge(merge_fn, ValueList(vs1), ValueList(vs2))
    ).to(
        ValueList(Vec(merge_fn(vs1[0], vs2[0]))).append(
            ValueList.Merge(merge_fn, ValueList(vs1.remove(0)), ValueList(vs2.remove(0)))
        ),
        # given
        vs1.length() > i64(0),
        vs2.length() > i64(0),
    )
    yield rewrite(
        ValueList.Merge(merge_fn, ValueList(Vec[Value].empty()), ValueList(Vec[Value].empty()))
    ).to(
        ValueList(Vec[Value].empty())
    )
    # toSet
    yield rewrite(
        ValueList(vs1).toSet()
    ).to(
        Set(vs1[0]) | ValueList(vs1.remove(0)).toSet(),
        # given
        vs1.length() > i64(0),
    )

    yield rewrite(
        ValueList(Vec[Value]()).toSet()
    ).to(
        Set[Value].empty()
    )


@ruleset
def _EvalMap_to_ValueList(
    vec_terms: Vec[Term],
    env: Env,
    map_fn: Callable[[Term], Value],
):
    # EvalMap to Value.List
    yield rewrite(EvalMap(env, TermList(vec_terms))).to(
        TermList(vec_terms).mapValue(partial(Eval, env)),
    )
    yield rewrite(TermList(vec_terms).mapValue(map_fn)).to(
        valuelist(),
        # given
        eq(i64(0)).to(vec_terms.length()),
    )
    yield rewrite(TermList(vec_terms).mapValue(map_fn)).to(
        valuelist(map_fn(vec_terms[0])).append(
            TermList(vec_terms.remove(0)).mapValue(map_fn)
        ),
        # given
        vec_terms.length() > i64(0),
    )


@ruleset
def _EnvEnter_EvalMap(terms: TermList, env: Env):
    # EnvEnter
    yield rewrite(EnvEnter(env, terms)).to(
        Env.nil().nest(EvalMap(env, terms))
    )

@ruleset
def _VBinOp_assoc(op: String, va: Value, vb: Value):
    yield rewrite(VBinOp(op, va, vb)).to(VBinOp(op, vb, va))

@ruleset
def _VBinOp_Lt(env: Env, ta: Term, tb: Term, i: i64, j: i64):
    yield rewrite(Eval(env, Term.Lt(ta, tb))).to(
        VBinOp("Lt", Eval(env, ta), Eval(env, tb))
    )
    # Constant I64
    yield rewrite(VBinOp("Lt", Value.ConstI64(i), Value.ConstI64(j))).to(
        Value.BoolTrue(),
        # given
        i < j,
    )
    yield rewrite(VBinOp("Lt", Value.ConstI64(i), Value.ConstI64(j))).to(
        Value.BoolFalse(),
        # given
        i >= j,
    )


@ruleset
def _VBinOp_Add(env: Env, ta: Term, tb: Term, i: i64, j: i64):
    yield rewrite(Eval(env, Term.Add(ta, tb))).to(
        VBinOp("Add", Eval(env, ta), Eval(env, tb))
    )


@ruleset
def _LoopAnalysis(
    env: Env,
    ta: Term,
    tb: Term,
    i: i64,
    j: i64,
    op: String,
    op2: String,
    vres: Value,
    va: Value,
    vb: Value,
    vc: Value,
    vstart: Value,
    vstop: Value,
    vstep: Value,
    vby: Value,
    vcond: Value,
    vn: Value,
    vphis: Vec[Value],
    phis: ValueList,
    outs: ValueList,
    vs: Vec[Value],
    _a: Value,
    _b: Value,
    _c: Value,


):
    # Match i += consti64 as LoopIncremented()
    yield rule(
        eq(vc).to(va | vb),
        eq(va).to(VBinOp(op, vc, vby)),
    ).then(
        union(LVAnalysis(va)).with_(LoopIncremented(op, vc, vb, vby))
    )
    # Match LoopIncrement() < n as LoopIndVar if cond used by VLoop
    yield rule(
        VLoop(ValueList(vphis), ValueList(vs).toValue()),
        eq(vcond).to(vs[0]),  # the condition
        eq(vcond).to(VBinOp("Lt", vc, vn)),
        vphis.contains(va),
        eq(LVAnalysis(vc)).to(LoopIncremented(op, va, vb, vby)),
        eq(i64(1) + vphis.length()).to(vs.length()), # wellformed
    ).then(
        union(LVAnalysis(va)).with_(LoopIndVar(op, vb, vn, vby))
    )
    # Match accumulator c += vstep
    yield rule(
        eq(LVAnalysis(va)).to(LoopIncremented(op, vb, vc, vby)),
        eq(LVAnalysis(vby)).to(LoopIndVar(op2, vstart, vstop, vstep))
    ).then(
        union(LVAnalysis(vb)).with_(LoopAccumIndVar(op, vstart, vstop, vstep))
    )

    # Help find the phi node
    yield rule(
        VLoop(ValueList(vphis), ValueList(vs).toValue()),
        vs.length() > 1,
    ).then(
        _MapLoopPhiOf(ValueList(vphis), ValueList(vs.remove(0))),
    )
    yield rewrite(_MapLoopPhiOf(ValueList(vphis), ValueList(vs))).to(
        valuelist(_LoopPhiOf(vphis[0], vs[0])).append(_MapLoopPhiOf(ValueList(vphis.remove(0)), ValueList(vs.remove(0)))),
        # given
        vphis.length() > 0
    )
    yield rewrite(_MapLoopPhiOf(valuelist(), valuelist())).to(valuelist())

    # Match VFix(v) and LVAnalysis(v) is LoopAccumIndVar
    yield rewrite(
        VFix(va)
    ).to(
        VSum(vstart, vstop, vstep),
        # given
        _LoopPhiOf(vb, va),
        eq(LVAnalysis(vb)).to(LoopAccumIndVar("Add", vstart, vstop, vstep))
    )



@ruleset
def _Eval_Term_Literals(
    env: Env,
    i: i64,
):
    yield rewrite(Eval(env, Term.LiteralI64(i))).to(Value.ConstI64(i))

def valuelist(*args: Value) -> ValueList:
    if not args:
        return ValueList(Vec[Value]())
    return ValueList(Vec(*args))


def make_rules():
    return (
        _Value_rules
        | _Eval_Term_Literals
        | _propagate_RegionDef_from_the_end
        | _VBranch
        | _VLoop
        | _eval_ins_get
        | _VGetPort
        | _EvalMap_to_ValueList
        | _ValueList_rules
        | _EnvEnter_EvalMap
        | _VBinOp_assoc
        | _VBinOp_Lt
        | _VBinOp_Add
        | _LoopAnalysis
    )


def run(root, *, checks=[], assume=None):
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

    egraph = EGraph() #save_egglog_string=True)
    egraph.let("root", root)

    ruleset = make_rules()

    if assume is not None:
        assume(egraph)

    saturate(egraph, ruleset)
    out = egraph.simplify(root, ruleset.saturate())
    # print(egraph.as_egglog_string)
    print("simplified output".center(80, "-"))
    print(out)
    print("=" * 80)
    if checks:
        egraph.check(*checks)
    return out


def region_builder(nin: int):
    def wrapped(fn):
        region = RegionDef(fn.__name__, nin)
        ins = region.begin()
        outs = fn(region, ins)
        assert isinstance(outs, (tuple, list))
        return region.end(termlist(*outs))

    return wrapped


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


def test_sum_loop():
    # Equivalent source:
    #     In [1]: def thesum(init, n):
    #    ...:     c = init
    #    ...:     for i in range(n):
    #    ...:         c += i
    #    ...:     return c
    # Target:
    #   c = sum(range(n))
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

            c = Term.Add(c, i)
            i = Term.Add(i, Term.LiteralI64(1))
            lt = Term.Lt(i, n)
            return [lt, i, n, c]

        # Do Loop
        loop = Term.Loop(termlist(i, n, c), body)

        # Return
        return [loop.getPort(2)]  # sum(range())

    # Eval with Env
    env = Env.nil()
    # env = env.nest(valuelist(Value.ConstI64(0), Value.ConstI64(10)))
    env = env.nest(valuelist(Value.Param(0), Value.Param(1)))
    root = Eval(env, main)

    # FIXME:
    # This is printing
    #   VSum(Value.ConstI64(0), Value.Param(1), Value.ConstI64(1))
    # But missing adjustment to initial offset
    checks = [
        # eq(root).to(valuelist(Value.ConstI64(1)).toValue()),
    ]
    run(root, checks=checks)


if __name__ == "__main__":
    test_sum_loop()

