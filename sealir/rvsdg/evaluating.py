from __future__ import annotations

import builtins
import inspect
import operator
from collections import ChainMap
from contextlib import contextmanager
from dataclasses import dataclass
from pprint import pprint
from typing import (
    Any,
    Mapping,
    MutableMapping,
    Sequence,
    Type,
    TypeAlias,
    cast,
)

from sealir import ase, rvsdg
from sealir.rvsdg import grammar as rg
from sealir.rvsdg import internal_prefix

SExpr: TypeAlias = ase.SExpr


@dataclass(frozen=True)
class EvalPorts:
    parent: SExpr
    values: tuple[Any, ...]

    def __getitem__(self, key: int) -> Any:
        return self.values[key]

    def get_by_name(self, k: str) -> Any:
        return self[self.get_port_names().index(k)]

    def get_port_names(self) -> Sequence[str]:
        match self.parent:
            case rg.RegionBegin(inports=ins):
                names = ins
            case rg.RegionEnd(ports=outs):
                names = tuple(k.name for k in outs)
            case _:
                raise ValueError(
                    f"get_port_names() not supported for parent={self.parent!r}"
                )
        return names

    def update_scope(self, scope: dict[str, Any]):
        scope.update(zip(self.get_port_names(), self.values, strict=True))

    def replace(self, ports: EvalPorts) -> EvalPorts:
        repl = []
        for k in self.get_port_names():
            repl.append(ports.get_by_name(k))
        return EvalPorts(self.parent, tuple(repl))

    def drop_first(self) -> EvalPorts:
        return EvalPorts(parent=self.parent, values=self.values[1:])


def evaluate(
    prgm: SExpr,
    callargs: tuple,
    callkwargs: dict,
    *,
    init_scope: Mapping | None = None,
    init_state: ase.TraverseState | None = None,
    init_memo: dict | None = None,
    global_ns: Mapping | None = None,
    dbginfo: rvsdg.SourceDebugInfo,
):
    stack: list[dict[str, Any]] = [{}]

    if global_ns is None:
        global_ns = {}

    glbs = ChainMap(global_ns, builtins.__dict__)  # type: ignore

    if init_scope is not None:
        stack[-1].update(init_scope)

    @contextmanager
    def push(callargs):
        stack.append(ChainMap({"__region_args__": callargs}, scope()))
        try:
            yield
        finally:
            stack.pop()

    def scope() -> MutableMapping[str, Any]:
        return stack[-1]

    def get_region_args() -> list[Any]:
        return scope()["__region_args__"]

    def ensure_io(expect_io: Any) -> Type[rg.IO]:
        assert expect_io is rg.IO, expect_io
        return expect_io

    def runner(expr: SExpr, state: ase.TraverseState):

        match expr:
            case rg.Func(fname=str(fname), args=funcargs, body=body):
                # assume we always evaluate a function
                assert isinstance(funcargs, rg.Args)
                params = []
                for arg in funcargs.arguments:
                    assert isinstance(arg, rg.ArgSpec)
                    params.append(
                        inspect.Parameter(
                            arg.name,
                            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    )
                sig = inspect.Signature(params)
                ba = sig.bind(*callargs, **callkwargs)

                # prepare region arguments
                region_args = []
                for k in body.begin.inports:
                    if k == internal_prefix("io"):
                        v = rg.IO
                    else:
                        v = ba.arguments[k]
                    region_args.append(v)

                with push(region_args):
                    out = yield body
                return out.get_by_name(internal_prefix("ret"))

            case rg.RegionBegin(inports=ins):
                region_args = get_region_args()
                ports = []
                for k, v in zip(ins, region_args, strict=True):
                    ports.append(v)
                return EvalPorts(parent=expr, values=tuple(ports))

            case rg.RegionEnd(begin=begin, ports=ports):
                inports = yield begin
                inports.update_scope(scope())
                portvals = []
                ports_list: tuple[Any, ...] = ports
                for port in ports_list:
                    portvals.append((yield port.value))
                dbginfo.print("In region", dict(scope()))
                return EvalPorts(expr, tuple(portvals))

            case rg.IfElse(
                cond=cond, body=body, orelse=orelse, operands=operands
            ):
                # handle condition
                condval = yield cond
                # handle region_args
                ops = []
                for op in operands:
                    ops.append((yield op))
                with push(ops):
                    if condval:
                        if_ports: EvalPorts = yield body
                    else:
                        if_ports = yield orelse
                if_ports.update_scope(cast(dict[str, Any], scope()))
                dbginfo.print("end if", dict(scope()))
                return EvalPorts(expr, if_ports.values)

            case rg.Loop(body=body, operands=operands):
                # handle region args
                ops = []
                for op in operands:
                    ops.append((yield op))
                # loop logic
                loop_cond: Any = True
                assert isinstance(body, rg.RegionEnd)
                begin = body.begin
                with push(ops):
                    memo = {}
                    memo[begin] = yield begin

                    while loop_cond:
                        loop_ports: EvalPorts = evaluate(
                            cast(SExpr, body),
                            (),
                            {},
                            init_scope=scope(),
                            init_memo=memo,
                            global_ns=global_ns,
                            dbginfo=dbginfo,
                        )
                        loop_ports.update_scope(cast(dict[str, Any], scope()))
                        # First port is expected to be the loop condition
                        loop_cond = loop_ports[0]
                        dbginfo.print("after loop iterator", dict(scope()))
                        memo[begin] = memo[begin].replace(loop_ports)

                # The loop output will drop the internal loop condition
                return loop_ports.drop_first()

            case rg.Unpack(val=source, idx=int(idx)):
                ports = yield source
                return ports[idx]

            case rg.ArgRef(idx=int(idx), name=str(name)):
                return scope()[name]

            case rg.IO():
                return rg.IO

            case rg.Undef(str(name)):
                return rg.Undef(name)

            case rg.DbgValue(
                name=str(varname),
                value=value,
                srcloc=srcloc,
                interloc=interloc,
            ):
                val = yield value
                with dbginfo.setup(srcloc=srcloc, interloc=interloc):
                    dbginfo.print("assign", varname, "=", val)
                scope()[varname] = val
                return val

            case rg.PyNone():
                return None

            case rg.PyBool(bool(v)):
                return v

            case rg.PyInt(int(v)):
                return v

            case rg.PyComplex(float(real), float(imag)):
                return complex(real, imag)

            case rg.PyStr(str(v)):
                return v

            case rg.PyUnaryOp(op=op, io=io, operand=operand):
                ioval = ensure_io((yield io))
                operandval = yield operand
                match op:
                    case "not":
                        res = not operandval
                    case "-":
                        res = -operandval
                    case _:
                        raise NotImplementedError(op)
                return EvalPorts(expr, (ioval, res))

            case rg.PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioval = ensure_io((yield io))
                lhsval = yield lhs
                rhsval = yield rhs
                match op:
                    case "+":
                        res = lhsval + rhsval
                    case "-":
                        res = lhsval - rhsval
                    case "*":
                        res = lhsval * rhsval
                    case "/":
                        res = lhsval / rhsval
                    case "//":
                        res = lhsval // rhsval
                    case "<":
                        res = lhsval < rhsval
                    case ">":
                        res = lhsval > rhsval
                    case "!=":
                        res = lhsval != rhsval
                    case "in":
                        res = lhsval in rhsval
                    case _:
                        raise NotImplementedError(op)
                return EvalPorts(expr, (ioval, res))

            case rg.PyInplaceBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioval = ensure_io((yield io))
                lhsval = yield lhs
                rhsval = yield rhs
                match op:
                    case "+":
                        res = operator.iadd(lhsval, rhsval)
                    case "-":
                        res = operator.isub(lhsval, rhsval)
                    case "*":
                        res = operator.imul(lhsval, rhsval)
                    case "/":
                        res = operator.itruediv(lhsval, rhsval)
                    case "//":
                        res = operator.ifloordiv(lhsval, rhsval)
                    case _:
                        raise NotImplementedError(op)
                return EvalPorts(expr, (ioval, res))

            case rg.PyCall(func=func, io=io, args=args):
                funcval = yield func
                ioval = ensure_io((yield io))
                argvals = []
                for arg in args:
                    argvals.append((yield arg))
                out = funcval(*argvals)
                return EvalPorts(expr, values=tuple([ioval, out]))

            case rg.PyLoadGlobal(io=io, name=str(varname)):
                ioval = ensure_io((yield io))
                return glbs[varname]

            case rg.PyTuple(elems):
                elemvals = []
                for el in elems:
                    elemvals.append((yield el))
                out = tuple(elemvals)
                return out

            case rg.PyList(elems):
                elemvals = []
                for el in elems:
                    elemvals.append((yield el))
                return elemvals

            case rg.PyAttr(io=io, value=value, attrname=str(attr)):
                ioval = ensure_io((yield io))
                out = getattr((yield value), attr)
                return EvalPorts(expr, values=tuple([ioval, out]))

            case rg.PySubscript(io=io, value=value, index=index):
                ioval = ensure_io((yield io))
                out = (yield value)[(yield index)]
                return EvalPorts(expr, values=tuple([ioval, out]))

            case _:
                raise NotImplementedError(expr)

    try:
        memo = ase.traverse(
            prgm, runner, state=init_state, init_memo=init_memo
        )
    except:
        print("-----debug-----")
        pprint(dict(scope()))
        raise
    return memo[prgm]
