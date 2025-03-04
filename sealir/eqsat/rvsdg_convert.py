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
from sealir.rvsdg import internal_prefix

from . import rvsdg_eqsat as eg

SExpr = ase.SExpr


@dataclass(frozen=True)
class RegionInfo:
    region: eg.RegionBegin
    ins: eg.InputPorts

    def __getitem__(self, idx: int) -> eg.Term:
        return self.ins.get(idx)


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
                    body=(yield body),
                )

            case rg.RegionBegin(ins=ins, ports=ports):
                procports = []
                for p in ports:
                    procports.append((yield p))
                rd = eg.Region(
                    node_uid(expr), ins=ins, ports=eg.termlist(*procports)
                )
                return RegionInfo(rd, rd.begin())

            case rg.RegionEnd(begin=begin, outs=str(outnames), ports=ports):
                ri: RegionInfo = (yield begin)
                with push(ri):
                    outs = []
                    for p in ports:
                        outs.append((yield p))
                return WrapTerm(
                    eg.Term.RegionEnd(ri.region, outnames, eg.termlist(*outs))
                )

            case rg.Unpack(val=source, idx=int(idx)):
                outs = yield source
                return outs[idx]

            case rg.IfElse(cond=cond, body=body, orelse=orelse, outs=outs):
                condval = yield cond
                outs_if = yield body
                outs_else = yield orelse
                bra = eg.Term.Branch(condval, outs_if.term, outs_else.term)
                return WrapTerm(bra)

            case rg.Loop(body=body, outs=str(outs), loopvar=str(loopvar)):
                region = yield body
                loop = eg.Term.Loop(region.term)
                return WrapTerm(loop)

            case rg.PyUnaryOp(op=str(op), io=io, operand=operand):
                ioterm = yield io
                operandterm = yield operand
                match op:
                    case "not":
                        res = eg.Term.NotIO(ioterm, operandterm)
                    case _:
                        raise NotImplementedError(f"unsupported op: {op!r}")

                return WrapTerm(res)

            case rg.PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioterm = yield io
                lhsterm = yield lhs
                rhsterm = yield rhs
                match op:
                    case "<":
                        res = eg.Term.LtIO(ioterm, lhsterm, rhsterm)

                    case "+":
                        res = eg.Term.AddIO(ioterm, lhsterm, rhsterm)

                    case "*":
                        res = eg.Term.MulIO(ioterm, lhsterm, rhsterm)

                    case "/":
                        res = eg.Term.DivIO(ioterm, lhsterm, rhsterm)

                    case "**":
                        res = eg.Term.PowIO(ioterm, lhsterm, rhsterm)

                    case _:
                        raise NotImplementedError(f"unsupported op: {op!r}")
                return WrapTerm(res)

            case rg.PyInplaceBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioterm = yield io
                lhsterm = yield lhs
                rhsterm = yield rhs
                match op:
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
                    eg.Term.Call(functerm, ioterm, eg.termlist(*argterms))
                )

            case rg.PyLoadGlobal(io=io, name=str(name)):
                ioterm = yield io
                return eg.Term.LoadGlobal(ioterm, name)

            case rg.PyAttr(io=io, value=value, attrname=str(attrname)):
                ioterm = yield io
                valterm = yield value
                return WrapTerm(eg.Term.AttrIO(ioterm, valterm, attrname))

            case rg.PyInt(int(intval)):
                assert intval.bit_length() < 64
                return eg.Term.LiteralI64(intval)

            case rg.PyFloat(float(fval)):
                return eg.Term.LiteralF64(fval)

            case rg.PyBool(bool(val)):
                return eg.Term.LiteralBool(val)

            case rg.DbgValue(
                name=str(varname),
                value=value,
                srcloc=srcloc,
                interloc=interloc,
            ):
                return (yield value)

            case rg.ArgRef(idx=int(i), name=str(name)):
                return eg.Term.Param(i)

            case rg.Undef():
                return eg.Term.Undef()

            case rg.IO():
                return eg.Term.IO()

            case _:
                raise NotImplementedError(repr(expr))

    return ase.traverse(root, coro, state=ase.TraverseState())
