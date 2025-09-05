# mypy: disable-error-code="empty-body"

from egglog import (
    StringLike,
    Vec,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    union,
)

from sealir.eqsat.rvsdg_eqsat import (
    IsConstantFalse,
    IsConstantTrue,
    Port,
    PortList,
    Region,
    Select,
    Term,
    TermDict,
    TermList,
    wildcard,
)

from .rvsdg_eqsat import DynInt, PortList, Region, Term, TermList, wildcard


@function
def Py_NotIO(io: Term, term: Term) -> Term: ...


@function
def Py_NegIO(io: Term, term: Term) -> Term: ...


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
def Py_Add(a: Term, b: Term) -> Term: ...


@function
def Py_InplaceAdd(a: Term, b: Term) -> Term: ...


@function
def Py_InplaceAddIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_SubIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_Mul(a: Term, b: Term) -> Term: ...


@function
def Py_MulIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_Div(a: Term, b: Term) -> Term: ...


@function
def Py_DivIO(io: Term, a: Term, b: Term) -> Term: ...


@function
def Py_MatMult(a: Term, b: Term) -> Term: ...


@function
def Py_MatMultIO(io: Term, a: Term, b: Term) -> Term: ...


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
def Py_SubscriptIO(io: Term, obj: Term, index: Term) -> Term: ...


@function
def Py_SliceIO(io: Term, lower: Term, upper: Term, step: Term) -> Term: ...


@function
def Py_Tuple(elems: TermList) -> Term: ...


@function
def Py_LoadGlobal(io: Term, name: StringLike) -> Term: ...


@function
def Py_Call(func: Term, io: Term, args: TermList) -> Term: ...


@function
def Py_CallKwargs(
    func: Term, io: Term, args: TermList, kwargs: TermDict
) -> Term: ...


@function
def Py_ForLoop(
    iter_arg_idx: DynInt | i64Like,
    indvar_arg_idx: DynInt,
    iterlast_arg_idx: DynInt,
    body: Term,
    operands: TermList,
) -> Term: ...


# -----------------------------------rulesets-----------------------------------

_wc = wildcard


@function
def GetLoopCondition(loop: Term) -> Term: ...


@ruleset
def loop_rules(
    loop: Term,
    loopbody: Term,
    loopoperands: TermList,
    region: Region,
    outports: Vec[Port],
    loopctrl: Term,
    ctrl_0: Term,
    ctrl_1: Term,
    loopcheck: Term,
    next_call: Term,
    iterator: Term,
    sentinel: Term,
    next_call_args: Vec[Term],
    iterator_port_idx: i64,
    then: Term,
    bodyoperands: TermList,
):
    yield rule(
        loop == Term.Loop(body=loopbody, operands=loopoperands),
        loopbody == Term.RegionEnd(region, PortList(outports)),
    ).then(union(GetLoopCondition(loop)).with_(outports[0].value))

    yield rule(
        loop == Term.Loop(body=loopbody, operands=loopoperands),
        # The loop condition must be a select based on the control variable.
        loopbody == Term.RegionEnd(region, PortList(outports)),
        GetLoopCondition(loop) == Py_NotIO(_wc(Term), loopctrl).getPort(1),
        # The loop control is selecting two constants
        loopctrl == Select(loopcheck.getPort(1), ctrl_0, ctrl_1),
        loopctrl
        == Term.IfElse(_wc(Term), then, _wc(Term), bodyoperands).getPort(
            _wc(i64)
        ),
        IsConstantFalse(ctrl_0),
        IsConstantTrue(ctrl_1),
        # Check next(iterator, sentinel)
        sentinel == Term.LiteralStr("__scfg_sentinel__"),
        loopcheck == Py_NeIO(_wc(Term), next_call.getPort(1), sentinel),
        next_call
        == Py_Call(
            func=Py_LoadGlobal(_wc(Term), "next"),
            io=_wc(Term),
            args=TermList(next_call_args),
        ),
        iterator == next_call_args[0],
        sentinel == next_call_args[1],
        next_call_args.length() == i64(2),
        # Get the argument position of the iterator
        iterator == region.get(iterator_port_idx),
    ).then(
        union(loop).with_(
            Py_ForLoop(
                iter_arg_idx=iterator_port_idx,
                # TODO: resolve DynInt|dyn_index|dyn_get
                indvar_arg_idx=(
                    indvar_idx := bodyoperands.dyn_index(next_call.getPort(1))
                ),
                iterlast_arg_idx=bodyoperands.dyn_index(
                    region.dyn_get(indvar_idx)
                ),
                # The `then` region can be used directly because by RVSDG Loop
                # design the input/output ports are the same in the IfElse
                # region and Loop region. (Loop region just have an extra
                # loop-condition port prepended)
                body=then,
                operands=loopoperands,
            )
        )
    )


@ruleset
def ruleset_literal_i64_folding(io: Term, ival: i64):
    # Constant fold negation of integer literals
    yield rewrite(Py_NegIO(io, Term.LiteralI64(ival)).getPort(1)).to(
        Term.LiteralI64(0 - ival)
    )


def make_rules():
    return loop_rules | ruleset_literal_i64_folding
