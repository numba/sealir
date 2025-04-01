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
    Unit,
    Vec,
    eq,
    f64,
    f64Like,
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


class InPorts(Expr):
    def __init__(self, names: Vec[String]): ...


def inports(*names: str):
    return InPorts(names=Vec[String](*names))


class Region(Expr):
    def __init__(self, uid: StringLike, inports: InPorts): ...

    def get(self, idx: i64Like) -> Term: ...


class Port(Expr):
    def __init__(self, name: StringLike, term: Term): ...

    @property
    def value(self) -> Term: ...


class Term(Expr):
    @classmethod
    def Func(cls, uid: StringLike, fname: StringLike, body: Term) -> Term: ...

    @classmethod
    def Apply(cls, region: Term, operands: TermList) -> Term: ...

    @classmethod
    def IfElse(
        cls, cond: Term, then: Term, orelse: Term, operands: TermList
    ) -> Term: ...

    @classmethod
    def Loop(
        cls, body: Term, loopvar: StringLike, operands: TermList
    ) -> Term: ...

    @classmethod
    def NotIO(cls, io: Term, term: Term) -> Term: ...

    @classmethod
    def Lt(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def LtIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def Gt(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def GtIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def Add(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def AddIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def InplaceAdd(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def InplaceAddIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def Mul(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def MulIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def Div(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def DivIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def Ne(cls, a: Term, b: Term) -> Term: ...
    @classmethod
    def NeIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @method(cost=100)
    @classmethod
    def Pow(cls, a: Term, b: Term) -> Term: ...
    @method(cost=100)
    @classmethod
    def PowIO(cls, io: Term, a: Term, b: Term) -> Term: ...
    @classmethod
    def AttrIO(cls, io: Term, obj: Term, attrname: StringLike) -> Term: ...
    @classmethod
    def LiteralI64(cls, val: i64Like) -> Term: ...
    @classmethod
    def LiteralF64(cls, val: f64Like) -> Term: ...
    @classmethod
    def LiteralBool(cls, val: Bool) -> Term: ...
    @classmethod
    def LiteralStr(cls, val: StringLike) -> Term: ...
    @classmethod
    def LiteralNone(cls) -> Term: ...
    @classmethod
    def IO(cls) -> Term: ...
    @classmethod
    def Undef(cls, name: StringLike) -> Term: ...
    @classmethod
    def Param(cls, idx: i64Like) -> Term: ...
    @classmethod
    def LoadGlobal(cls, io: Term, name: StringLike) -> Term: ...
    @classmethod
    def Call(cls, func: Term, io: Term, args: TermList) -> Term: ...
    @classmethod
    def RegionEnd(
        self,
        region: Region,
        ports: PortList,
    ) -> Term: ...

    def getPort(self, idx: i64Like) -> Term: ...


class TermList(Expr):
    def __init__(self, terms: Vec[Term]): ...

    def __getitem__(self, idx: i64) -> Term: ...


class PortList(Expr):
    def __init__(self, ports: Vec[Port]): ...

    def __getitem__(self, idx: i64) -> Port: ...


# class Debug(Expr):

#     @method(unextractable=True)
#     @classmethod
#     def ValueOf(cls, term: Term) -> Value: ...


def termlist(*args: Term) -> TermList:
    return TermList(Vec(*args))


def portlist(*args: Port) -> PortList:
    return PortList(Vec(*args))


@function
def PartialEval(env: Env, func: Term) -> Value: ...


@function
def GraphRoot(t: Term) -> Term: ...


@function
def IsConstantTrue(t: Term) -> Unit: ...


@function
def IsConstantFalse(t: Term) -> Unit: ...


# ------------------------------ RuleSets ------------------------------


@ruleset
def ruleset_portlist_basic(
    vecport: Vec[Port], idx: i64, a: Term, portname: String
):
    yield rewrite(
        # simplify PortList indexing
        PortList(vecport)[idx]
    ).to(
        vecport[idx],
        # given
        idx < vecport.length(),
    )
    yield rewrite(Port(portname, a).value, subsume=True).to(a)


@ruleset
def ruleset_termlist_basic(vecterm: Vec[Term], idx: i64):
    yield rewrite(
        # simplify TermList indexing
        TermList(vecterm)[idx]
    ).to(
        vecterm[idx],
        # given
        idx < vecterm.length(),
    )


@ruleset
def ruleset_apply_region(
    ports: PortList,
    idx: i64,
    region: Region,
    uid: String,
    inports: InPorts,
    operands: TermList,
):
    yield rewrite(
        # apply region is the same as the output ports of the region
        Term.Apply(Term.RegionEnd(region, ports), operands).getPort(idx)
    ).to(ports[idx].value)

    yield rule(
        # Unify apply-region inputs with apply-operand values.
        Term.Apply(Term.RegionEnd(region, ports), operands),
        region.get(idx),
    ).then(union(region.get(idx)).with_(operands[idx]))


ruleset_rvsdg_basic = (
    ruleset_portlist_basic | ruleset_termlist_basic | ruleset_apply_region
)


@ruleset
def ruleset_const_propagate(a: Term, ival: i64):
    yield rule(a == Term.LiteralI64(ival), ival != 0).then(IsConstantTrue(a))
    yield rule(a == Term.LiteralI64(ival), ival == 0).then(IsConstantFalse(a))


@ruleset
def ruleset_const_fold_if_else(a: Term, b: Term, c: Term, operands: TermList):
    yield rewrite(Term.IfElse(cond=a, then=b, orelse=c, operands=operands)).to(
        Term.Apply(b, operands),
        # given
        IsConstantTrue(a),
    )
    yield rewrite(Term.IfElse(cond=a, then=b, orelse=c, operands=operands)).to(
        Term.Apply(c, operands),
        # given
        IsConstantFalse(a),
    )


def make_rules(*, communtative=False):
    rules = (
        ruleset_rvsdg_basic
        | ruleset_const_propagate
        | ruleset_const_fold_if_else
    )
    # rules |= _PartialEval_rules
    # if communtative:
    #     rules |= _VBinOp_communtativity
    return rules
