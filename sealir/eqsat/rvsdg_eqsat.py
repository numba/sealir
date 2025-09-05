# mypy: disable-error-code="empty-body"

from __future__ import annotations

import sys

from egglog import (
    Bool,
    Expr,
    Map,
    Set,
    String,
    StringLike,
    Unit,
    Vec,
    delete,
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
    var,
)
from egglog.conversion import converter

MAXCOST = sys.maxsize


class DynInt(Expr):
    def __init__(self, num: i64Like): ...
    def get(self) -> i64: ...
    def __mul__(self, other: DynInt) -> DynInt: ...


converter(i64, DynInt, DynInt)


class InPorts(Expr):
    def __init__(self, names: Vec[String]): ...


def inports(*names: str):
    return InPorts(names=Vec[String](*names))


class Region(Expr):
    def __init__(self, uid: StringLike, inports: InPorts): ...

    def get(self, idx: i64Like) -> Term: ...

    def dyn_get(self, idx: DynInt) -> Term: ...


class Port(Expr):
    def __init__(self, name: StringLike, term: Term): ...

    @property
    def value(self) -> Term: ...

    @property
    def name(self) -> String: ...


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
    def Loop(cls, body: Term, operands: TermList) -> Term: ...

    @classmethod
    def RegionEnd(
        self,
        region: Region,
        ports: PortList,
    ) -> Term: ...

    @classmethod
    def IO(cls) -> Term: ...

    @classmethod
    def Undef(cls, name: StringLike) -> Term: ...

    @classmethod
    def Param(cls, idx: i64Like) -> Term: ...

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

    # TODO: add Loc
    # TODO: this node is merged with `value`. Extraction should look in the
    #       eclass for DbgValue and reconstruct them.
    @classmethod
    def DbgValue(cls, varname: StringLike, value: Term) -> Term: ...

    def getPort(self, idx: i64Like) -> Term: ...


class TermList(Expr):
    def __init__(self, terms: Vec[Term]): ...

    def __getitem__(self, idx: i64Like) -> Term: ...

    def dyn_index(self, target: Term) -> DynInt: ...


class TermDict(Expr):
    def __init__(self, term_map: Map[String, Term]): ...

    def lookup(self, key: StringLike) -> Unit:
        """Trigger rule to lookup a value in the dict.

        This is needed for .get() to match
        """
        ...

    def get(self, key: StringLike) -> Term: ...


@function(cost=MAXCOST)  # max cost to make it unextractable
def _dyn_index_partial(terms: Vec[Term], target: Term) -> DynInt: ...


class PortList(Expr):
    def __init__(self, ports: Vec[Port]): ...

    def __getitem__(self, idx: i64) -> Port: ...

    def getValue(self, idx: i64) -> Term: ...


# class Debug(Expr):

#     @method(unextractable=True)
#     @classmethod
#     def ValueOf(cls, term: Term) -> Value: ...


def termlist(*args: Term) -> TermList:
    return TermList(Vec(*args))


def termdict(**kwargs: Term) -> TermDict:
    mapping = Map[String, Term].empty()
    for k, v in kwargs.items():
        mapping = mapping.insert(k, v)
    return TermDict(mapping)


def portlist(*args: Port) -> PortList:
    return PortList(Vec(*args))


@function
def GraphRoot(t: Term) -> Term: ...


@function
def IsConstant(t: Term) -> Unit: ...


@function
def IsConstantTrue(t: Term) -> Unit: ...


@function
def IsConstantFalse(t: Term) -> Unit: ...


@function
def Select(cond: Term, then: Term, orelse: Term) -> Term: ...


# ------------------------------ RuleSets ------------------------------


def wildcard(ty):
    wildcard._count += 1
    return var(f"_{wildcard._count}", ty)


wildcard._count = 0


@ruleset
def ruleset_portlist_basic(
    vecport: Vec[Port],
    idx: i64,
    a: Term,
    portname: String,
    port: Port,
    portlist: PortList,
):
    yield rewrite(
        # simplify PortList indexing
        PortList(vecport)[idx]
    ).to(
        vecport[idx],
        # given
        idx < vecport.length(),
    )
    yield rule(port == Port(portname, a)).then(
        union(port.value).with_(a),
        set_(port.name).to(portname),
    )
    yield rewrite(portlist.getValue(idx)).to(portlist[idx].value)


@ruleset
def ruleset_portlist_ifelse(
    idx: i64,
    ifelse: Term,
    then_portlist: PortList,
    orelse_portlist: PortList,
):
    # IfElse
    yield rule(
        ifelse
        == Term.IfElse(
            cond=wildcard(Term),
            then=Term.RegionEnd(wildcard(Region), then_portlist),
            orelse=Term.RegionEnd(wildcard(Region), orelse_portlist),
            operands=wildcard(TermList),
        ),
        ifelse.getPort(idx),
    ).then(
        then_portlist.getValue(idx),
        orelse_portlist.getValue(idx),
    )


@ruleset
def ruleset_portlist_loop(
    loop: Term,
    body: Term,
    operands: Vec[Term],
    idx: i64,
    ports: PortList,
    region: Region,
):
    # Loop
    yield rule(
        loop == Term.Loop(body=body, operands=TermList(operands)),
        body == Term.RegionEnd(region=region, ports=ports),
        loop.getPort(idx),
    ).then(
        # Offset by one because the loop condition port (port 0) is stripped
        ports.getValue(idx + 1),
    )


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
def ruleset_termlist_dyn_index(vecterm: Vec[Term], target: Term):
    # Simplify TermList.dyn_index into DynInt(i64)
    yield rewrite(TermList(vecterm).dyn_index(target), subsume=True).to(
        _dyn_index_partial(vecterm, target),
        # given
        vecterm.contains(target),
    )
    last = vecterm.length() - 1
    yield rewrite(
        _dyn_index_partial(vecterm, target),
        subsume=True,
    ).to(
        # Recurse into vecterm.pop()
        _dyn_index_partial(vecterm.pop(), target),
        # given
        vecterm[last] != target,
    )
    yield rewrite(
        _dyn_index_partial(vecterm, target),
        subsume=True,
    ).to(
        # Found as the last element
        DynInt(last),
        # given
        vecterm[last] == target,
    )


@ruleset
def ruleset_simplify_dbgvalue(
    t: Term,
    varname: String,
):
    # DbgValue is merged with base value.
    yield rewrite(Term.DbgValue(varname, t)).to(t)


@ruleset
def ruleset_apply_region(
    ports: PortList,
    idx: i64,
    region: Region,
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


@ruleset
def ruleset_region_dyn_get(region: Region, idx: i64):
    yield rewrite(
        region.dyn_get(DynInt(idx)),
        subsume=True,
    ).to(region.get(idx))


@ruleset
def ruleset_region_propgate_output(
    body: Term,
    portlist: PortList,
    region: Region,
    idx: i64,
    vecports: Vec[Port],
    stop: i64,
):
    @function
    def _propagate_regionend(
        start: i64Like,
        stop: i64Like,
        portlist: PortList,
    ) -> Unit: ...

    yield rule(
        body == Term.RegionEnd(ports=portlist, region=region),
        portlist == PortList(vecports),
    ).then(_propagate_regionend(0, vecports.length(), portlist))
    yield rule(
        stmt := _propagate_regionend(idx, stop, portlist),
        idx < stop,
    ).then(
        # define the PortList.getValue
        portlist.getValue(idx),
        # subsume the temp statement
        delete(stmt),
    )

    yield rule(
        # Step to idx + 1
        _propagate_regionend(idx, stop, portlist),
        idx + 1 < stop,
    ).then(
        _propagate_regionend(idx + 1, stop, portlist),
    )


@function(unextractable=True)
def _define_region_outputs(body: Term, idx: i64, end: i64) -> Term: ...


@ruleset
def ruleset_func_outputs(
    func: Term,
    uid: String,
    fname: String,
    body: Term,
    region: Region,
    portlist: PortList,
    ports: Vec[Port],
    idx: i64,
    end: i64,
):
    yield rule(
        func == Term.Func(uid=uid, fname=fname, body=body),
        body == Term.RegionEnd(region, ports=portlist),
        portlist == PortList(ports),
    ).then(
        # Need this because limitation Vec.
        # Defer the expansion in rules below
        _define_region_outputs(body, i64(0), ports.length())
    )

    # Expand _define_region_outputs
    yield rule(
        x := _define_region_outputs(body, idx, end),
        idx < end,
    ).then(
        # March to next index
        _define_region_outputs(body, idx + 1, end),
        # Define
        union(x).with_(body.getPort(idx)),
        # Delete _define_region_outputs
        delete(x),
    )

    yield rule(
        # Delete if out of range
        x := _define_region_outputs(body, idx, idx)
    ).then(delete(x))


@ruleset
def ruleset_termdict(mapping: Map[String, Term], key: String):
    yield rule(
        TermDict(mapping).lookup(key),
        mapping.contains(key),
    ).then(set_(TermDict(mapping).get(key)).to(mapping[key]))


@ruleset
def ruleset_dynint(n: i64, m: i64):
    yield rule(DynInt(n)).then(set_(DynInt(n).get()).to(n))
    yield rewrite(DynInt(n) * DynInt(m)).to(DynInt(n * m))


ruleset_rvsdg_basic = (
    ruleset_simplify_dbgvalue
    | ruleset_portlist_basic
    | ruleset_portlist_ifelse
    | ruleset_portlist_loop
    | ruleset_termlist_basic
    | ruleset_apply_region
    | ruleset_termlist_dyn_index
    | ruleset_region_dyn_get
    | ruleset_region_propgate_output
    | ruleset_func_outputs
    | ruleset_termdict
    | ruleset_dynint
)


@ruleset
def ruleset_const_propagate(a: Term, ival: i64):
    # Constant boolean
    yield rule(a == Term.LiteralI64(ival), ival != i64(0)).then(
        IsConstantTrue(a)
    )
    yield rule(a == Term.LiteralI64(ival), ival == i64(0)).then(
        IsConstantFalse(a)
    )

    # Constant integers
    yield rule(a == Term.LiteralI64(ival)).then(IsConstant(a))
    yield rule(a == Term.LiteralI64(ival)).then(IsConstant(a))


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


@ruleset
def ruleset_ifelse_propagation(
    ifelse: Term,
    cond: Term,
    then: Term,
    orelse: Term,
    idx: i64,
    then_ports: PortList,
    else_ports: Vec[Port],
):
    yield rule(
        # Create Select statement from IfElse outputs that are constants on
        # both sides
        ifelse == Term.IfElse(cond, then, orelse, wildcard(TermList)),
        ifelse.getPort(idx),
        then == Term.RegionEnd(wildcard(Region), then_ports),
        orelse == Term.RegionEnd(wildcard(Region), PortList(else_ports)),
        IsConstant(then_ports.getValue(idx)),
        IsConstant(else_ports[idx].value),
    ).then(
        union(ifelse.getPort(idx)).with_(
            Select(cond, then_ports[idx].value, else_ports[idx].value)
        )
    )

    yield rewrite(
        # Simplify Select if both side are the same
        Select(cond, then, orelse)
    ).to(
        then,
        # given both side are the same
        then == orelse,
    )


def make_rules(*, communtative=False):
    rules = (
        ruleset_rvsdg_basic
        | ruleset_const_propagate
        | ruleset_const_fold_if_else
        | ruleset_ifelse_propagation
    )
    # rules |= _PartialEval_rules
    # if communtative:
    #     rules |= _VBinOp_communtativity
    return rules
