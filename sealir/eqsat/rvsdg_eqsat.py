# mypy: disable-error-code="empty-body"

from __future__ import annotations

from functools import partial
from typing import Callable

from egglog import (
    Bool,
    Expr,
    Set,
    String,
    StringLike,
    Vec,
    eq,
    function,
    i64,
    i64Like,
    method,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
)


class Region(Expr):
    def __init__(self, uid: StringLike, ins: StringLike, ports: TermList): ...

    def begin(self) -> InputPorts: ...


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
    def IOState(cls) -> Value: ...

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
    def Merge(
        cls,
        merge_fn: Callable[[Value, Value], Value],
        vas: ValueList,
        vbs: ValueList,
    ) -> ValueList: ...


class Term(Expr):
    @classmethod
    def Func(cls, uid: StringLike, fname: StringLike, body: Term) -> Term: ...

    @classmethod
    def Branch(cls, cond: Term, then: Term, orelse: Term) -> Term: ...
    @classmethod
    def Loop(cls, body: Term) -> Term: ...

    @classmethod
    def Lt(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def LtIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def Add(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def AddIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def LiteralI64(cls, val: i64) -> Term: ...
    @classmethod
    def IO(cls) -> Term: ...

    @classmethod
    def Param(cls, idx: i64Like) -> Term: ...

    @classmethod
    def RegionEnd(
        self, region: Region, outs: StringLike, ports: TermList
    ) -> Term: ...

    def getPort(self, idx: i64Like) -> Term: ...


class TermList(Expr):
    def __init__(self, terms: Vec[Term]): ...

    # def map(self, fn: Callable[[Term], Term]) -> TermList: ...

    def mapValue(self: TermList, fn: Callable[[Term], Value]) -> ValueList: ...


class Debug(Expr):

    @method(unextractable=True)
    @classmethod
    def ValueOf(cls, term: Term) -> Value: ...


def termlist(*args: Term) -> TermList:
    return TermList(Vec(*args))


@function
def has_pure_type(v: Value) -> Bool:
    """Type of value is pure such that all operations on it has no side effects."""
    ...


@function
def Eval(env: Env, term: Term) -> Value: ...


@function
def EvalMap(env: Env, terms: TermList) -> ValueList: ...


@function
def VFix(v: Value) -> Value: ...


@function
def VGetPort(vs: ValueList, idx: i64) -> Value: ...


@function
def VFunc(body: Value) -> Value: ...


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
def VBinOpIO(
    opname: StringLike, io: Value, lhs: Value, rhs: Value
) -> Value: ...


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
def LVAnalysis(phi: Value) -> LVA:
    """LVAnalysis always apply to the phi node"""
    ...


@function
def LoopIncremented(
    op: StringLike, phi: Value, init: Value, step: Value, res: Value
) -> LVA: ...
@function
def LoopIndVar(
    op: StringLike, start: Value, stop: Value, step: Value
) -> LVA: ...
@function
def LoopAccumIndVar(
    op: StringLike, init: Value, start: Value, stop: Value, step: Value
) -> LVA: ...


@function
def IsLoopInVariant(v: Value) -> Bool: ...


@function
def VSum(vstart: Value, vstop: Value, vstep: Value) -> Value: ...


@function
def PartialEvaluated(value: Value) -> Term: ...


@function
def DoPartialEval(env: Env, func: Term) -> Value: ...


@function
def GraphRoot(t: Term) -> Term: ...


# ------------------------------ RuleSets ------------------------------


@ruleset
def _propagate_RegionDef_from_the_end(
    vec_terms: Vec[Term],
    env: Env,
    regionname: String,
    ins: String,
    outs: String,
    args: TermList,
):
    yield rewrite(
        Eval(
            env,
            Term.RegionEnd(
                Region(regionname, ins, args), outs, TermList(vec_terms)
            ),
        ),
    ).to(EvalMap(env, TermList(vec_terms)).toValue())
    # ).to(EvalMap(EnvEnter(env, args), TermList(vec_terms)).toValue())


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
    yield rewrite(Eval(env, Term.Branch(cond, then_term, else_term))).to(
        VBranch(
            Eval(env, cond),
            Eval(env, then_term),
            Eval(env, else_term),
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
    i: i64,
    regionname: String,
    ins: String,
    outs: String,
    out_terms: TermList,
):
    # VLoop
    yield rewrite(
        Eval(env, Term.Loop(body_term)),
    ).to(
        _LoopTemp(_LoopBack(EvalMap(env, input_terms), body_term)),
        # given
        eq(body_term).to(
            Term.RegionEnd(
                Region(regionname, ins, input_terms), outs, out_terms
            )
        ),
    )

    yield rule(
        eq(vl).to(_LoopBack(phis, body_term)),
    ).then(
        union(_LoopTemp(vl)).with_(
            VLoop(
                _pm := PhiMap(phis, vl),
                _inner := Eval(EnvLoop(_pm), body_term),
            )
        ),
        union(vl).with_(_LoopDropCond(_inner)),
    )

    yield rewrite(VLoop(vl, va)).to(_LoopOutputs(va).toValue())

    # EnvLoop
    yield rewrite(EnvLoop(phis)).to(Env.nil().nest(phis))
    # _LoopDropCond
    yield rewrite(_LoopDropCond(ValueList(vec_va).toValue())).to(
        # Drop the loop condition
        ValueList(vec_va.remove(0))
    )

    # _LoopOutputs
    yield rewrite(_LoopOutputs(ValueList(vec_va).toValue())).to(
        # Drop the loop condition
        ValueList(vec_va.remove(0)).map(VFix)
    )

    # VFix of Param
    yield rewrite(VFix(Value.Param(i))).to(Value.Param(i))

    # PhiMap
    yield rewrite(PhiMap(vl, vl2)).to(ValueList.Merge(VPhi, vl, vl2))

    # Phi
    yield rewrite(VPhi(va, vb), subsume=True).to(va | vb)


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
    yield rewrite(a | b).to(b | a)

    # merge phis
    yield rewrite(a | a).to(a)
    # merge loop back phis
    yield rule(eq(a).to(a | b)).then(
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
    yield rewrite(
        ValueList(vs1).append(ValueList(vs2)),
        subsume=True,
    ).to(ValueList(vs1.append(vs2)))
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
    yield rewrite(
        ValueList(vs1).map(fn),
        subsume=True,
    ).to(
        valuelist(fn(vs1[0])).append(ValueList(vs1.remove(0)).map(fn)),
        # given
        vs1.length() > i64(0),
    )
    # Merge
    yield rewrite(
        ValueList.Merge(merge_fn, ValueList(vs1), ValueList(vs2)),
        subsume=True,
    ).to(
        ValueList(Vec(merge_fn(vs1[0], vs2[0]))).append(
            ValueList.Merge(
                merge_fn, ValueList(vs1.remove(0)), ValueList(vs2.remove(0))
            )
        ),
        # given
        vs1.length() > i64(0),
        vs2.length() > i64(0),
    )
    yield rewrite(
        ValueList.Merge(
            merge_fn,
            ValueList(Vec[Value].empty()),
            ValueList(Vec[Value].empty()),
        ),
        subsume=True,
    ).to(ValueList(Vec[Value].empty()))
    # toSet
    yield rewrite(ValueList(vs1).toSet()).to(
        Set(vs1[0]) | ValueList(vs1.remove(0)).toSet(),
        # given
        vs1.length() > i64(0),
    )

    yield rewrite(ValueList(Vec[Value]()).toSet()).to(Set[Value].empty())


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
def _Debug_Eval(term: Term, env: Env, val: Value):
    yield rule(
        Debug.ValueOf(term),
        eq(val).to(Eval(env, term)),
    ).then(union(Debug.ValueOf(term)).with_(val))


@ruleset
def _EnvEnter_EvalMap(terms: TermList, env: Env):
    # EnvEnter
    yield rewrite(EnvEnter(env, terms)).to(Env.nil().nest(EvalMap(env, terms)))


@ruleset
def _VBinOp_communtativity(va: Value, vb: Value):
    for op in ["Add"]:
        yield rewrite(VBinOp(op, va, vb)).to(VBinOp(op, vb, va))


@ruleset
def _VBinOp_Lt(
    env: Env, ta: Term, tb: Term, io: Term, i: i64, j: i64, op: String
):
    yield rewrite(Eval(env, Term.Lt(ta, tb))).to(
        VBinOp("Lt", Eval(env, ta), Eval(env, tb))
    )
    yield rewrite(Eval(env, Term.LtIO(io, ta, tb))).to(
        VBinOpIO("Lt", Eval(env, io), Eval(env, ta), Eval(env, tb))
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
def _VBinOp_Pure(
    ta: Value,
    tb: Value,
    io: Value,
    op: String,
    i: i64,
):
    # Constant I64 is pure
    yield rewrite(VBinOpIO(op, io, ta, tb)).to(
        ValueList(Vec(Value.IOState(), VBinOp(op, ta, tb))).toValue(),
        # given
        has_pure_type(ta),
        has_pure_type(tb),
    )

    yield rule(
        eq(ta).to(Value.ConstI64(i)),
    ).then(set_(has_pure_type(ta)).to(Bool(True)))


@ruleset
def _VBinOp_Add(env: Env, ta: Term, tb: Term, i: i64, j: i64, io: Term):
    yield rewrite(Eval(env, Term.Add(ta, tb))).to(
        VBinOp("Add", Eval(env, ta), Eval(env, tb))
    )

    yield rewrite(Eval(env, Term.Add(ta, tb))).to(
        VBinOp("Add", Eval(env, ta), Eval(env, tb))
    )
    yield rewrite(Eval(env, Term.AddIO(io, ta, tb))).to(
        VBinOpIO("Add", Eval(env, io), Eval(env, ta), Eval(env, tb))
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
    vinit: Value,
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
    ).then(union(LVAnalysis(vc)).with_(LoopIncremented(op, vc, vb, vby, va)))
    # Match LoopIncrement() < n as LoopIndVar if cond used by VLoop
    yield rule(
        VLoop(ValueList(vphis), ValueList(vs).toValue()),
        eq(vcond).to(vs[0]),  # the condition
        eq(vcond).to(VBinOp("Lt", vc, vn)),
        vphis.contains(va),
        eq(LVAnalysis(va)).to(LoopIncremented(op, va, vb, vby, vc)),
        eq(i64(1) + vphis.length()).to(vs.length()),  # wellformed
    ).then(union(LVAnalysis(va)).with_(LoopIndVar(op, vb, vn, vby)))
    # Match accumulator c += vstep
    yield rule(
        eq(LVAnalysis(vb)).to(LoopIncremented(op, vb, vc, vby, va)),
        eq(LVAnalysis(vby)).to(LoopIndVar(op2, vstart, vstop, vstep)),
    ).then(
        union(LVAnalysis(vb)).with_(
            LoopAccumIndVar(op, init=vc, start=vstart, stop=vstop, step=vstep)
        )
    )

    # Help find the phi node
    yield rule(
        VLoop(ValueList(vphis), ValueList(vs).toValue()),
        vs.length() > 1,
    ).then(
        _MapLoopPhiOf(ValueList(vphis), ValueList(vs.remove(0))),
    )
    yield rewrite(
        _MapLoopPhiOf(ValueList(vphis), ValueList(vs)),
        subsume=True,
    ).to(
        valuelist(_LoopPhiOf(vphis[0], vs[0])).append(
            _MapLoopPhiOf(ValueList(vphis.remove(0)), ValueList(vs.remove(0)))
        ),
        # given
        vphis.length() > 0,
    )
    yield rewrite(
        _MapLoopPhiOf(valuelist(), valuelist()),
        subsume=True,
    ).to(valuelist())

    # Match VFix(v) and LVAnalysis(v) is LoopAccumIndVar
    yield rewrite(VFix(va)).to(
        VBinOp("Add", vinit, VSum(vstart, vstop, vstep)),
        # given
        _LoopPhiOf(vb, va),
        eq(LVAnalysis(vb)).to(
            LoopAccumIndVar("Add", vinit, vstart, vstop, vstep)
        ),
    )


@ruleset
def _Eval_Term_Literals(
    env: Env,
    i: i64,
):
    yield rewrite(Eval(env, Term.LiteralI64(i))).to(Value.ConstI64(i))


@ruleset
def _PartialEval_rules(
    env: Env,
    term: Term,
    body: Term,
    value: Value,
    uid: String,
    fname: String,
):
    yield rewrite(DoPartialEval(env, Term.Func(uid, fname, body))).to(
        VFunc(Eval(env, body))
    )
    yield rewrite(term).to(
        PartialEvaluated(value),
        # given
        eq(value).to(Eval(env, term)),
    )


def valuelist(*args: Value) -> ValueList:
    if not args:
        return ValueList(Vec[Value]())
    return ValueList(Vec(*args))


def make_rules(*, communtative=False):
    rules = (
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
        | _VBinOp_Pure
        | _VBinOp_Lt
        | _VBinOp_Add
        | _Debug_Eval
        | _LoopAnalysis
    )
    rules |= _PartialEval_rules
    if communtative:
        rules |= _VBinOp_communtativity
    return rules
