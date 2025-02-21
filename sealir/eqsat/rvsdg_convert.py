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

from . import rvsdg_eqsat as eg

SExpr = ase.SExpr


@dataclass(frozen=True)
class RegionInfo:
    region: eg.RegionDef
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

    def region_uid(expr: SExpr) -> str:
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
                return (yield body).getPort(1)

            case rg.RegionBegin(ins=ins, ports=ports):
                nin = ins.split()
                rd = eg.RegionDef(region_uid(expr), len(nin))
                return RegionInfo(rd, rd.begin())

            case rg.RegionEnd(begin=begin, outs=str(outnames), ports=ports):
                ri: RegionInfo = (yield begin)
                with push(ri):
                    outs = []
                    for p in ports:
                        outs.append((yield p))
                return ri.region.end(eg.termlist(*outs))

            case rg.Unpack(val=source, idx=int(idx)):
                outs = yield source
                return outs[idx]

            case rg.IfElse(cond=cond, body=body, orelse=orelse, outs=outs):
                condval = yield cond
                outs_if = yield body
                outs_else = yield orelse
                ins = []
                for p in body.begin.ports:
                    ins.append((yield p))
                bra = eg.Term.Branch(
                    condval, eg.termlist(*ins), outs_if, outs_else
                )
                return WrapTerm(bra)

            case rg.PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioterm = yield io
                lhsterm = yield lhs
                rhsterm = yield rhs
                match op:
                    case "<":
                        res = eg.Term.LtIO(ioterm, lhsterm, rhsterm)

                    case _:
                        raise NotImplementedError(f"unsupported op: {op!r}")
                return WrapIO(ioterm, res)

            case rg.DbgValue(
                name=str(varname),
                value=value,
                srcloc=srcloc,
                interloc=interloc,
            ):
                return (yield value)
            case _:
                raise NotImplementedError(repr(expr))

    return ase.traverse(root, coro, state=ase.TraverseState())
