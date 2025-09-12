"""
This file implements conversion logic from RVSDG S-expr into
Egraph
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass

from sealir import ase
from sealir.rvsdg import grammar as rg

from . import py_eqsat
from . import rvsdg_eqsat as eg

SExpr = ase.SExpr


@dataclass(frozen=True)
class RegionInfo:
    region: eg.Region

    def __getitem__(self, idx: int) -> eg.Term:
        return self.region.get(idx)


@dataclass(frozen=True)
class WrapIO:
    io: eg.Term
    val: eg.Term

    def __getitem__(self, idx: int) -> eg.Term:
        match idx:
            case 0:
                return self.io
            case 1:
                return self.val
        raise IndexError(idx)


@dataclass(frozen=True)
class WrapTerm:
    term: eg.Term

    def __getitem__(self, idx: int) -> eg.Term:
        return self.term.getPort(idx)


def egraph_conversion(root: SExpr):

    def node_uid(expr: SExpr) -> str:
        return str(expr._handle)

    @contextmanager
    def push(ri: RegionInfo):
        region_stacks.append(ri)
        try:
            yield
        finally:
            region_stacks.pop()

    # states

    region_stacks: list[RegionInfo] = []

    def coro(expr: SExpr, state: ase.TraverseState):
        match expr:
            case rg.Func(fname=str(fname), args=rg.Args(args), body=body):
                return eg.Term.Func(
                    uid=node_uid(expr),
                    fname=fname,
                    body=(yield body).term,
                )

            case rg.RegionBegin(inports=ins, attrs=attrs):
                rd = eg.Region(
                    node_uid(expr),
                    inports=eg.inports(*ins),
                )
                assert isinstance(attrs, rg.Attrs)
                return RegionInfo(rd)

            case rg.RegionEnd(begin=begin, ports=ports):
                ri: RegionInfo = (yield begin)
                with push(ri):
                    portnodes = []
                    for p in ports:
                        portnodes.append((yield p))
                return WrapTerm(
                    eg.Term.RegionEnd(ri.region, eg.portlist(*portnodes))
                )

            case rg.Port(name=str(name), value=value):
                return eg.Port(name, (yield value))

            case rg.Unpack(val=source, idx=int(idx)):
                outs = yield source
                return outs[idx]

            case rg.IfElse(
                cond=cond, body=body, orelse=orelse, operands=operands
            ):
                condval = yield cond
                outs_if = yield body
                outs_else = yield orelse
                out_operands = []
                for op in operands:
                    out_operands.append((yield op))
                bra = eg.Term.IfElse(
                    condval,
                    outs_if.term,
                    outs_else.term,
                    operands=eg.termlist(*out_operands),
                )
                return WrapTerm(bra)

            case rg.Loop(body=body, operands=operands):
                region = yield body
                out_operands = []
                for op in operands:
                    out_operands.append((yield op))
                loop = eg.Term.Loop(
                    region.term,
                    operands=eg.termlist(*out_operands),
                )
                return WrapTerm(loop)

            case rg.PyUnaryOp(op=str(op), io=io, operand=operand):
                ioterm = yield io
                operandterm = yield operand
                match op:
                    case "not":
                        res = py_eqsat.Py_NotIO(ioterm, operandterm)
                    case "-":
                        res = py_eqsat.Py_NegIO(ioterm, operandterm)
                    case _:
                        raise NotImplementedError(f"unsupported op: {op!r}")

                return WrapTerm(res)

            case rg.PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioterm = yield io
                lhsterm = yield lhs
                rhsterm = yield rhs
                match op:
                    case "<":
                        res = py_eqsat.Py_LtIO(ioterm, lhsterm, rhsterm)

                    case ">":
                        res = py_eqsat.Py_GtIO(ioterm, lhsterm, rhsterm)

                    case "+":
                        res = py_eqsat.Py_AddIO(ioterm, lhsterm, rhsterm)

                    case "-":
                        res = py_eqsat.Py_SubIO(ioterm, lhsterm, rhsterm)

                    case "*":
                        res = py_eqsat.Py_MulIO(ioterm, lhsterm, rhsterm)

                    case "/":
                        res = py_eqsat.Py_DivIO(ioterm, lhsterm, rhsterm)

                    case "@":
                        res = py_eqsat.Py_MatMultIO(ioterm, lhsterm, rhsterm)

                    case "**":
                        res = py_eqsat.Py_PowIO(ioterm, lhsterm, rhsterm)

                    case "!=":
                        res = py_eqsat.Py_NeIO(ioterm, lhsterm, rhsterm)

                    case _:
                        raise NotImplementedError(f"unsupported op: {op!r}")
                return WrapTerm(res)

            case rg.PyInplaceBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioterm = yield io
                lhsterm = yield lhs
                rhsterm = yield rhs
                match op:
                    case "+":
                        res = py_eqsat.Py_InplaceAddIO(
                            ioterm, lhsterm, rhsterm
                        )
                    case _:
                        raise NotImplementedError(f"unsupported op: {op!r}")
                return WrapTerm(res)

            case rg.PyCall(func=func, io=io, args=args):
                functerm = yield func
                ioterm = yield io
                argterms = []
                for arg in args:
                    argterms.append((yield arg))
                return WrapTerm(
                    py_eqsat.Py_Call(functerm, ioterm, eg.termlist(*argterms))
                )

            case rg.PyCallKwargs(
                func=func,
                io=io,
                args=rg.Posargs() as posargs,
                kwargs=rg.Kwargs() as kwargs,
            ):
                functerm = yield func
                ioterm = yield io
                posarg_terms = []
                for arg in posargs.args:
                    posarg_terms.append((yield arg))
                kwarg_dict = {}
                kw: rg.Keyword
                for kw in kwargs.kwargs:
                    kwarg_dict[kw.name] = yield kw.value
                return WrapTerm(
                    py_eqsat.Py_CallKwargs(
                        functerm,
                        ioterm,
                        eg.termlist(*posarg_terms),
                        eg.termdict(**kwarg_dict),
                    )
                )

            case rg.PyLoadGlobal(io=io, name=str(name)):
                ioterm = yield io
                return py_eqsat.Py_LoadGlobal(ioterm, name)

            case rg.PyAttr(io=io, value=value, attrname=str(attrname)):
                ioterm = yield io
                valterm = yield value
                return WrapTerm(py_eqsat.Py_AttrIO(ioterm, valterm, attrname))

            case rg.PySubscript(io=io, value=value, index=index):
                ioterm = yield io
                valterm = yield value
                idxterm = yield index
                return WrapTerm(
                    py_eqsat.Py_SubscriptIO(ioterm, valterm, idxterm)
                )

            case rg.PySetItem(io=io, obj=obj, value=value, index=index):
                ioterm = yield io
                objterm = yield obj
                valterm = yield value
                idxterm = yield index
                return WrapTerm(
                    py_eqsat.Py_SetitemIO(ioterm, objterm, idxterm, valterm)
                )

            case rg.PySlice(io=io, lower=lower, upper=upper, step=step):
                ioterm = yield io
                lowerterm = yield lower
                upperterm = yield upper
                stepterm = yield step
                return WrapTerm(
                    py_eqsat.Py_SliceIO(ioterm, lowerterm, upperterm, stepterm)
                )

            case rg.PyTuple(elems):
                elemvals = []
                for el in elems:
                    elemvals.append((yield el))
                return py_eqsat.Py_Tuple(eg.termlist(*elemvals))

            case rg.PyInt(int(intval)):
                assert intval.bit_length() < 64
                return eg.Term.LiteralI64(intval)

            case rg.PyFloat(float(fval)):
                return eg.Term.LiteralF64(fval)

            case rg.PyBool(bool(val)):
                return eg.Term.LiteralBool(val)

            case rg.PyStr(str(val)):
                return eg.Term.LiteralStr(val)

            case rg.PyNone():
                return eg.Term.LiteralNone()

            case rg.DbgValue(
                name=str(varname),
                value=value,
                srcloc=srcloc,  # TODO
                interloc=interloc,  # TODO
            ):
                return eg.Term.DbgValue(varname, (yield value))

            case rg.ArgRef(idx=int(i), name=str(name)):
                return eg.Term.Param(i)

            case rg.Undef(name=str(name)):
                return eg.Term.Undef(name=name)

            case rg.IO():
                return eg.Term.IO()

            case _:
                raise NotImplementedError(repr(expr))

    return ase.traverse(root, coro, state=ase.TraverseState())
