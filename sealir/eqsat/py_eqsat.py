# mypy: disable-error-code="empty-body"

from egglog import String, StringLike, function, rule, ruleset, union

from .rvsdg_eqsat import PortList, Region, Term, TermList


@function
def Py_NotIO(io: Term, term: Term) -> Term: ...


@function
def Py_Lt(a: Term, b: Term) -> Term: ...


@function
def Py_LtIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_Gt(a: Term, b: Term) -> Term: ...


@function
def Py_GtIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_Add(a: Term, b: Term) -> Term: ...


@function
def Py_AddIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_InplaceAdd(a: Term, b: Term) -> Term: ...


@function
def Py_InplaceAddIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_Mul(a: Term, b: Term) -> Term: ...


@function
def Py_MulIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_Div(a: Term, b: Term) -> Term: ...


@function
def Py_DivIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_Ne(a: Term, b: Term) -> Term: ...


@function
def Py_NeIO(io: Term, a: Term, b: Term) -> Term: ...


@function(cost=100)  # TODO: move cost to extraction
def Py_Pow(a: Term, b: Term) -> Term: ...


@function(cost=100)  # TODO: move cost to extraction
def Py_PowIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_AttrIO(io: Term, obj: Term, attrname: StringLike) -> Term: ...


@function
def Py_LoadGlobal(io: Term, name: StringLike) -> Term: ...


@function
def Py_Call(func: Term, io: Term, args: TermList) -> Term: ...


# -----------------------------------rulesets-----------------------------------


@ruleset
def ruleset_pyloop(
    loop: Term,
    body: Term,
    loopvar: String,
    operands: TermList,
    region: Region,
    ports: PortList,
):
    yield rule(
        loop == Term.Loop(body=body, loopvar=loopvar, operands=operands),
        body == Term.RegionEnd(ports=ports, region=region),
    ).then(union(loop).with_(Term.LiteralStr("matched")))


def make_rules():
    return ruleset_pyloop
