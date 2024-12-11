# mypy: disable-error-code="empty-body"

from __future__ import annotations

import os
from typing import Callable
from functools import partial

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
    def ConstI64(cls, val: i64) -> Value: ...

    def toList(self) -> ValueList: ...

    def __or__(self, rhs: Value) -> Value: ...


class ValueList(Expr):
    def __init__(self, vals: Vec[Value]): ...

    def append(self, vs: ValueList) -> ValueList: ...

    def toValue(self) -> Value: ...


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


def make_TermList(*args: Term) -> TermList:
    return TermList(Vec(*args))


@function
def Eval(env: Env, term: Term) -> Value: ...


@function
def EvalMap(env: Env, terms: TermList) -> ValueList: ...


@function
def VGetPort(vs: ValueList, idx: i64) -> Value: ...


@function
def VBranch(cond: Value, then: Value, orelse: Value) -> Value: ...


@function
def VLoop(body: Value) -> Value: ...


@function
def VBinOp(opname: StringLike, lhs: Value, rhs: Value) -> Value: ...


@function
def VMergeList(lhs: Value, rhs: Value) -> Value: ...


@function
def EnvEnter(env: Env, ins: TermList) -> Env: ...


@function
def EnvLoop(env: Env, loop: Term) -> Env: ...


@function
def FoldBinOp(op: StringLike, lhs: Value, rhs: Value) -> Value: ...


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


# @egraph.register
# def _VLoop(
#     env: Env,
#     input_terms: Vec[Term],
#     body_term: Term,
#     vec_va: Vec[Value],
#     vec_vb: Vec[Value],
# ):
#     # VLoop
#     loop = Term.Loop(input_terms, body_term)
#     yield rewrite(
#         Eval(env, loop)
#     ).to(
#         VLoop(
#             Eval(
#                 EnvLoop(env, loop),
#                 body_term,
#             )
#         ),
#     )
#     # EnvLoop
#     yield rewrite(
#         EnvLoop(env, loop),
#     ).to(
#         EnvEnter(env, VListFresh(input_terms)),
#         # Env.nil().nest(
#         #     VCons()
#         #     VMergeList(
#         #         EvalMap(env, input_terms),
#         #         Unlist(DropListFirst(loop)),
#         #     )
#         # )
#     )
#     """
#     a[0] = a0 | aN
#     b[0] = b0 | bN

#     Eval(Env(a[0], b[0]), term)


#     """
#     # VMergeList
#     yield rewrite(
#         VMergeList(Value.List(vec_va), Value.List(vec_vb))
#     ).to(
#         VCons(vec_va[0] | vec_vb[0], VMergeList(Value.List(vec_va.remove(0)),
#                                                 Value.List(vec_vb.remove(0)))),
#         # given
#         vec_va.length() > i64(0),
#         vec_vb.length() > i64(0),
#     )
#     yield rewrite(
#         VMergeList(Value.List(vec_va), Value.List(vec_vb))
#     ).to(
#         VConsEmpty(),
#         # given
#         eq(i64(0)).to(vec_va.length()),
#         eq(i64(0)).to(vec_vb.length()),
#     )


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
def _ValueList_rules(
    vs1: Vec[Value],
    vs2: Vec[Value],
    vl: ValueList,
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


@ruleset
def _EvalMap_to_ValueList(
    vec_terms: Vec[Term],
    vec_vals: Vec[Value],
    env: Env,
    val: Value,
    term: Term,
    map_fn: Callable[[Term], Value],
    term_list: TermList,
):
    # EvalMap to Value.List
    yield rewrite(EvalMap(env, TermList(vec_terms))).to(
        TermList(vec_terms).mapValue(partial(Eval, env)),
    )
    yield rewrite(TermList(vec_terms).mapValue(map_fn)).to(
        new_values(),
        # given
        eq(i64(0)).to(vec_terms.length()),
    )
    yield rewrite(TermList(vec_terms).mapValue(map_fn)).to(
        new_values(map_fn(vec_terms[0])).append(
            TermList(vec_terms.remove(0)).mapValue(map_fn)
        ),
        # given
        vec_terms.length() > i64(0),
    )


@ruleset
def _EnvEnter_EvalMap(vec_terms: Vec[Term], env: Env):
    # EnvEnter
    yield rewrite(EnvEnter(env, TermList(vec_terms))).to(
        Env.nil().nest(EvalMap(env, TermList(vec_terms)))
    )


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
def _VBinOp_LoopFold(
    env: Env,
    ta: Term,
    tb: Term,
    i: i64,
    j: i64,
    op: String,
    vres: Value,
    va: Value,
    vb: Value,
):
    yield rule(eq(vres).to(VBinOp(op, va, vb))).then(FoldBinOp(op, va, vb))
    """
    a = a0 | a1

    a1 = a + 1

    """


def new_values(*args: Value) -> ValueList:
    if not args:
        return ValueList(Vec[Value]())
    return ValueList(Vec(*args))


def make_rules():
    return (
        _propagate_RegionDef_from_the_end
        | _VBranch
        | _eval_ins_get
        | _VGetPort
        | _EvalMap_to_ValueList
        | _ValueList_rules
        | _EnvEnter_EvalMap
        | _VBinOp_Lt
        | _VBinOp_Add
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

    egraph = EGraph()
    egraph.let("root", root)

    schedule = make_rules().saturate()

    if assume is not None:
        assume(egraph)

    saturate(egraph, schedule)
    out = egraph.simplify(root, schedule)
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
        return region.end(make_TermList(*outs))

    return wrapped


def saturate(egraph: EGraph, schedule):
    if DEBUG:
        egraph.saturate(schedule, n_inline_leaves=2)
    else:
        egraph.run(schedule)


def test_straight_line_basic():
    from egglog import eq

    @region_builder(5)
    def main(region, ins):
        return list(map(ins.get, range(5)))

    env = Env.nil()
    env = env.nest(new_values(*map(Value.Param, range(5))))
    root = Eval(env, main)

    checks = [
        eq(root).to(new_values(*map(Value.Param, range(5))).toValue()),
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
        ifthen = Term.Branch(lt, make_TermList(a, b), if_then, or_else)

        # Return
        return [ifthen.getPort(0)]

    # Eval with Env
    env = Env.nil()
    env = env.nest(new_values(Value.ConstI64(0), Value.ConstI64(1)))
    root = Eval(env, main)

    checks = [
        eq(root).to(Value.ConstI64(1)),
    ]
    run(root, checks=checks)


# def test_sum_loop():
#     from egglog import eq

#     fn = RegionDef("main", 2)

#     @region_builder(2)
#     def main(region, ins):
#         ctx_body = fn.begin()
#         init = ctx_body.get(0)
#         n = ctx_body.get(1)
#         c = init
#         i = Term.LiteralI64(0)

#         @region_builder(2)
#         def body(region, ins):
#             i = ins.get(0)
#             n = ins.get(1)
#             c = ins.get(2)

#             c = Term.Add(c, i)
#             i = Term.Add(i, Term.LiteralI64(1))
#             lt = Term.Lt(i, n)
#             return [lt, i, n, c]

#         # Do Loop
#         loop = Term.Loop(Vec(i, n, c), body)

#         # Return
#         return [loop.getPort(2)]

#     # Eval with Env
#     env = Env.nil()
#     env = env.nest(Value.List(Vec(Value.ConstI64(0), Value.ConstI64(10))))
#     root = Eval(env, main)

#     checks = [
#         eq(root).to(Value.List(Vec(Value.ConstI64(1)))),
#     ]
#     run(root, checks=checks)


if __name__ == "__main__":
    test_max_if_else()


# test_straight_line_basic()
