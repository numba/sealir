from __future__ import annotations

import builtins
import inspect
import operator
from collections import ChainMap
from contextlib import contextmanager
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Sequence, Type, TypeAlias

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
            case rg.RegionBegin(ins=str(ins)):
                names = ins.split()
            case rg.RegionEnd(outs=str(outs)):
                names = outs.split()
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


def evaluate(
    prgm: SExpr,
    callargs: tuple,
    callkwargs: dict,
    *,
    init_scope: dict | None = None,
    init_state: ase.TraverseState | None = None,
    init_memo: dict | None = None,
    debugger: rvsdg.SourceInfoDebugger,
):
    stack: list[dict[str, Any]] = [{}]
    glbs = builtins.__dict__

    if init_scope is not None:
        stack[-1].update(init_scope)

    @contextmanager
    def push():
        stack.append(ChainMap({}, scope()))
        try:
            yield
        finally:
            stack.pop()

    def scope() -> dict[str, Any]:
        return stack[-1]

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
                scope().update(ba.arguments)
                with push():
                    ports = yield body
                return ports.get_by_name(internal_prefix("ret"))
            case rg.RegionBegin(ins=ins, ports=ports):
                paired = zip(ins.split(), ports, strict=True)
                ports = []
                for k, v in paired:
                    val = yield v
                    ports.append(val)
                    scope()[k] = val
                return EvalPorts(parent=expr, values=tuple(ports))

            case rg.RegionEnd(begin=begin, outs=str(outs), ports=ports):
                inports = yield begin
                with push():
                    inports.update_scope(scope())
                    debugger.print("In region", dict(scope()))
                    outvals = []
                    for port in ports:
                        outvals.append((yield port))
                    return EvalPorts(expr, tuple(outvals))

            case rg.IfElse(cond=cond, body=body, orelse=orelse, outs=outs):
                condval = yield cond
                if condval:
                    ports = yield body
                else:
                    ports = yield orelse
                ports.update_scope(scope())
                debugger.print("end if", dict(scope()))
                return EvalPorts(expr, ports.values)

            case rg.Loop(body=body, outs=outs, loopvar=loopvar):
                cond = True
                assert isinstance(body, rg.RegionEnd)
                begin = body.begin
                memo = {}
                memo[begin] = yield begin

                while cond:
                    ports = evaluate(
                        body,
                        (),
                        {},
                        init_scope=scope(),
                        init_memo=memo,
                        debugger=debugger,
                    )
                    ports.update_scope(scope())
                    cond = ports.get_by_name(loopvar)
                    debugger.print("after loop iterator", dict(scope()))
                    memo[begin] = memo[begin].replace(ports)
                return ports

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
                with debugger.setup(srcloc=srcloc, interloc=interloc):
                    debugger.print("assign", varname, "=", val)
                scope()[varname] = val
                return val

            case rg.PyNone():
                return None

            case rg.PyBool(bool(v)):
                return v

            case rg.PyInt(int(v)):
                return v

            case rg.PyStr(str(v)):
                return v

            case rg.PyUnaryOp(op=op, io=io, operand=operand):
                ioval = ensure_io((yield io))
                operandval = yield operand
                match op:
                    case "not":
                        res = not operandval
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
                    case "<":
                        res = lhsval < rhsval
                    case "!=":
                        res = lhsval != rhsval
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
