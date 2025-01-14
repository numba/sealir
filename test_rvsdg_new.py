
from __future__ import annotations

import ast
import inspect
from string import Formatter
import logging
import operator
import time
import textwrap
from collections import ChainMap
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import reduce
from pprint import pformat, pprint
from textwrap import dedent
from typing import Any, Iterable, Iterator, NamedTuple, Sequence, TypeAlias, cast

from sealir import ase, grammar, lam
from sealir.rewriter import TreeRewriter

from sealir import rvsdg

_DEBUG=True

SExpr: TypeAlias = ase.SExpr

from numba_rvsdg.core.datastructures.ast_transforms import (
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
    unparse_code,
)

def pp(expr: SExpr):
    if _DEBUG:
        print(
            pformat(ase.as_tuple(expr, -1, dedup=True))
            .replace(",", "")
            .replace("'", "")
        )
        # print(pformat(expr.as_dict()))


def _internal_prefix(name: str) -> str:
    # "!" will always sort to the front of all visible characters.
    return "!" + name


def restructure_source(function):
    ast2scfg_transformer = AST2SCFGTransformer(function)
    astcfg = ast2scfg_transformer.transform_to_ASTCFG()
    scfg = astcfg.to_SCFG()
    scfg.restructure()
    scfg2ast = SCFG2ASTTransformer()
    original_ast = unparse_code(function)[0]
    transformed_ast = scfg2ast.transform(original=original_ast, scfg=scfg)

    transformed_ast = ast.fix_missing_locations(transformed_ast)

    srclines, firstline = inspect.getsourcelines(function)
    firstline = 0

    source_text = dedent("".join(srclines))

    t_start = time.time()

    prgm = rvsdg.convert_to_sexpr(transformed_ast, firstline)

    # _logger.debug("convert_to_sexpr", time.time() - t_start)

    # varinfo = rvsdg.find_variable_info(prgm)
    # _logger.debug(varinfo)

    # _logger.debug("find_variable_info", time.time() - t_start)

    grm = Grammar(prgm._tape)

    rvsdg_out = convert_to_rvsdg(grm, prgm)

    # _logger.debug("convert_to_rvsdg", time.time() - t_start)

    # from sealir.prettyformat import html_format

    # pp(rvsdg)

    # lam_node = convert_to_lambda(rvsdg, varinfo)
    # _logger.debug("convert_to_lambda", time.time() - t_start)
    # if _DEBUG:
    #     pp(lam_node)
    #     print(ase.pretty_str(lam_node))

    # if _DEBUG_HTML:
    #     # FIXME: This is currently slow due to inefficient metadata lookup.
    #     print("writing html...")
    #     ts = time.time()
    #     with open("debug.html", "w") as fout:
    #         html_format.write_html(
    #             fout,
    #             html_format.prepare_source(source_text),
    #             html_format.to_html(rvsdg),
    #             html_format.to_html(lam_node),
    #         )
    #     print("   took", time.time() - ts)
    # return lam_node



class _Root(grammar.Rule):
    pass



class Args(_Root):
    arguments: tuple[SExpr, ...]


class RegionBegin(_Root):
    ins: str
    ports: tuple[SExpr, ...]

class RegionEnd(_Root):
    begin: RegionBegin
    outs: str
    ports: tuple[SExpr, ...]

class Func(_Root):
    fname: str
    args: Args
    body: RegionEnd


class IfElse(_Root):
    cond: SExpr
    body: SExpr
    orelse: SExpr
    outs: str

class DoWhile(_Root):
    body: SExpr
    outs: str

class IO(_Root):
    pass

class ArgSpec(_Root):
    name: str
    annotation: SExpr

class Return(_Root):
    io: SExpr
    val: SExpr

class PyNone(_Root):
    ...

class PyInt(_Root):
    value: int

class PyStr(_Root):
    value: str

class PyCall(_Root):
    func: SExpr
    io: SExpr
    args: tuple[SExpr]

class PyUnaryOp(_Root):
    op: str
    io: SExpr
    operand: SExpr

class PyBinOp(_Root):
    op: str
    io: SExpr
    lhs: SExpr
    rhs: SExpr


class PyInplaceBinOp(_Root):
    op: str
    io: SExpr
    lhs: SExpr
    rhs: SExpr

class PyLoadGlobal(_Root):
    io: SExpr
    name: str

class Var(_Root):
    name: str

class Undef(_Root):
    name: str

class Unpack(_Root):
    val: SExpr
    idx: int

class ArgRef(_Root):
    idx: int
    name: str

class Grammar(grammar.Grammar):
    start = _Root


@dataclass(frozen=True)
class RvsdgizeState(ase.TraverseState):
    context: RvsdgizeCtx


@dataclass(frozen=True)
class Scope:
    kind: str
    varmap: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RvsdgizeCtx:
    grm: Grammar
    scope_stack: list[Scope] = field(default_factory=list)
    scope_map: dict[SExpr, Scope] = field(default_factory=dict)

    @property
    def scope(self) -> Scope:
        return self.scope_stack[-1]

    def initialize_scope(self, rb: SExpr) -> None:
        grm = self.grm
        varmap = self.scope.varmap
        for i, k in enumerate(rb.ins.split()):
            varmap[k] = grm.write(Unpack(val=rb, idx=i))

    @contextmanager
    def new_function(self, node: SExpr):
        scope = Scope(kind="function")

        scope.varmap[_internal_prefix("io")] = self.grm.write(IO())
        self.scope_map[node] = scope
        self.scope_stack.append(scope)
        try:
            yield scope
        finally:
            self.scope_stack.pop()

    @contextmanager
    def new_block(self, node: SExpr):
        scope = Scope(kind="block")
        self.scope_map[node] = scope
        self.scope_stack.append(scope)
        try:
            yield scope
        finally:
            self.scope_stack.pop()

    def add_argument(self, i: int, name: str):
        scope = self.scope
        assert scope.kind == "function"
        assert name not in scope.varmap
        scope.varmap[name] = self.grm.write(ArgRef(idx=i, name=name))

    def load_var(self, name: str) -> SExpr:
        scope = self.scope
        if v:=scope.varmap.get(name):
            return v
        else:
            # Treat as global
            return self.grm.write(PyLoadGlobal(io=self.load_io(), name=name))

    def store_var(self, name: str, value: SExpr) -> None:
        scope = self.scope
        scope.varmap[name] = value

    def store_io(self, value: SExpr) -> None:
        self.store_var(_internal_prefix("io"), value)

    def load_io(self) -> SExpr:
        return self.load_var(_internal_prefix("io"))

    def updated_vars(self, scopelist) -> list[str]:
        updated = set()
        for scope in scopelist:
            updated.update(scope.varmap)

        return sorted(updated)

    def load_vars(self, names) -> tuple[SExpr, ...]:
        return tuple(self.load_var(k) for k in names)

    def insert_io_node(self, node: grammar.Rule):
        grm = self.grm
        written = grm.write(node)
        io, res = (grm.write(Unpack(val=written, idx=i)) for i in range(2))
        self.store_io(io)
        return res


def rvsdgization(expr: ase.BasicSExpr, state: RvsdgizeState):
    ctx = state.context
    grm = ctx.grm

    def prep_varnames(d) -> str:
        return ' '.join(d)

    def fixup(regend, updated_vars):
        updated = prep_varnames(updated_vars)

        if updated == regend.outs:
            return regend
        else:
            oldports = dict(zip(regend.outs.split(), regend.ports))
            ports = tuple(oldports[k] if k in oldports else grm.write(Undef(k))
                            for k in updated_vars)
            return grm.write(RegionEnd(begin=regend.begin, outs=updated, ports=ports))

    match (expr._head, expr._args):
        case ("PyAst_FunctionDef", (str(fname), args, body, loc)):
            with ctx.new_function(expr):
                return grm.write(Func(fname=fname, args=(yield args), body=(yield body)))
        case ("PyAst_arg", (str(name), annotation, loc)):
            return grm.write(ArgSpec(name=name, annotation=(yield annotation)))
        case ("PyAst_arguments", args):
            arg_done = []
            for i, arg in enumerate(args):
                x = (yield arg)
                arg_done.append(x)
                ctx.add_argument(i, x.name)
            return grm.write(Args(tuple(arg_done)))
        case ("PyAst_block", body):

            begin = grm.write(RegionBegin(ins=prep_varnames(ctx.scope.varmap),
                                          ports=ctx.load_vars(ctx.scope.varmap)))
            with ctx.new_block(expr) as scope:
                ctx.initialize_scope(begin)
                for expr in body:
                    (yield expr)
                vars = sorted(scope.varmap)
                ports = ctx.load_vars(vars)
            return grm.write(RegionEnd(begin=begin,
                                       outs=prep_varnames(vars),
                                       ports=ports))

        case ("PyAst_If", (test, body, orelse, loc)):
            cond = (yield test)
            br_true = (yield body)
            br_false = (yield orelse)
            updated_vars = ctx.updated_vars([ctx.scope_map[body], ctx.scope_map[orelse]])
            # fixup mismatching updated vars
            br_true = fixup(br_true, updated_vars)
            br_false = fixup(br_false, updated_vars)

            swt = grm.write(IfElse(cond=cond, body=br_true, orelse=br_false, outs=' '.join(updated_vars)))
            # update scope
            for i, k in enumerate(updated_vars):
                ctx.store_var(k, grm.write(Unpack(val=swt, idx=i)))

        case ("PyAst_While", (_alwaystrue, body, loc)):
            loopbody = (yield body)
            updated_vars = ctx.updated_vars([ctx.scope_map[body]])
            dow = grm.write(DoWhile(body=loopbody, outs=' '.join(updated_vars)))
            # update scope
            for i, k in enumerate(updated_vars):
                ctx.store_var(k, grm.write(Unpack(val=dow, idx=i)))

        case ("PyAst_Return", (value, loc)):
            v = (yield value)
            ctx.store_var(_internal_prefix("ret"), v)

        case ("PyAst_Assign", (rval, *targets, loc)):
            res = (yield rval)
            tar: SExpr
            for tar in targets:
                assert tar._head == "PyAst_Name", tar
                name = tar._args[0]
                ctx.store_var(name, res)
            return

        case ("PyAst_AugAssign", (str(op), target, rhs, loc)):
            match target._head, target._args:
                case ("PyAst_Name", (str(varname), "store", _)):
                    pass
                case _:
                    raise AssertionError(target)
            lhs = ctx.load_var(varname)
            rhs = (yield rhs)
            res = PyInplaceBinOp(op=op, io=ctx.load_io(), lhs=lhs, rhs=rhs)
            return ctx.insert_io_node(res)

        case ("PyAst_UnaryOp", (str(op), operand, loc)):
            res = PyUnaryOp(op=op, io=ctx.load_io(),
                            operand=(yield operand))
            return ctx.insert_io_node(res)

        case ("PyAst_BinOp", (str(op), lhs, rhs, loc)):
            res = PyBinOp(op=op, io=ctx.load_io(),
                          lhs=(yield lhs), rhs=(yield rhs))
            return ctx.insert_io_node(res)

        case ("PyAst_Compare", (str(op), lhs, rhs, loc)):
            res =  PyBinOp(op=op, io=ctx.load_io(),
                           lhs=(yield lhs), rhs=(yield rhs))
            return ctx.insert_io_node(res)

        case ("PyAst_Call", (SExpr() as func , SExpr() as posargs, loc)):
            proc_args = []
            for arg in posargs._args:
                proc_args.append((yield arg))

            call = PyCall(func=(yield func), io=ctx.load_io(), args=tuple(proc_args))
            return  ctx.insert_io_node(call)

        case ("PyAst_Name", (str(name), "load", loc)):
            return ctx.load_var(name)

        case ("PyAst_Constant_int", (int(value), loc)):
            return grm.write(PyInt(value))

        case ("PyAst_Constant_str", (str(value), loc)):
            return grm.write(PyStr(value))

        case ("PyAst_None", (loc,)):
            return grm.write(PyNone())
        case ("PyAst_Pass", (loc,)):
            return
        case _:
            raise NotImplementedError(expr)



def format_rvsdg(grm: Grammar, prgm: SExpr) -> str:

    def _inf_counter():
        c = 0
        while True:
            yield c
            c += 1

    counter = _inf_counter()

    def fresh_name() -> str:
        return f"${next(counter)}"

    def indent(text: str) -> str:
        return textwrap.indent(text, ' ' * 4)

    buffer = []

    def formatter(expr: SExpr, state: ase.TraverseState):
        match expr:
            case Func(fname=str(fname), args=args, body=body):
                buffer.append(f"{fname} = Func {ase.pretty_str(args)}")
                (yield body)
            case RegionBegin(ins, ports):
                inports = []
                for port in ports:
                    inports.append((yield port))
                name = fresh_name()
                buffer.append(f"{name} = Region {ins} <- {' '.join(inports)}")
                return name
            case RegionEnd(begin=begin, outs=str(outs), ports=ports):
                (yield begin)
                buffer.append("{")
                for port in ports:
                    (yield port)
                buffer.append(f"}} -> {outs}")
                # name = fresh_name()
                # buffer.append(f"=> {name}")
                # return name
            case IfElse(cond, body, orelse, outs):
                condref = (yield cond)
                name = fresh_name()
                buffer.append(f"{name} = If {condref} ")
                (yield body)
                buffer.append("Else")
                (yield orelse)
                buffer.append(f"Endif -> {outs}")
                return name

            case Unpack(val=source, idx=int(idx)):
                ref = (yield source)
                return f"({ref}:{idx})"
            case ArgRef(idx=int(idx), name=str(name)):
                return f"(ArgRef {idx} {name})"

            case IO():
                return "IO"

            case Undef(str(k)):
                name = fresh_name()
                buffer.append(f"{name} = Undef {k}")
                return name

            case PyInt(int(v)):
                name = fresh_name()
                buffer.append(f"{name} = PyInt {v}")
                return name

            case PyBinOp(op, io, lhs, rhs):
                ioref = (yield io)
                lhsref = (yield lhs)
                rhsref = (yield rhs)
                name = fresh_name()
                buffer.append(f"{name} = PyBinOp {op} {ioref}, {lhsref}, {rhsref}")
                return name
            case _:
                print("----debug")
                print('\n'.join(buffer))
                raise NotImplementedError(expr)


    ase.traverse(prgm, formatter)

    return'\n'.join(buffer)


def convert_to_rvsdg(grm: Grammar, prgm: SExpr):
    pp(prgm)

    state = RvsdgizeState(RvsdgizeCtx(grm=grm))
    out = ase.traverse(prgm, rvsdgization, state)[prgm]

    # out._tape.render_dot(only_reachable=True).view()

    print(out._tape.dump())
    pp(out)

    print(format_rvsdg(grm, out))



def test_if_else():

    def udt(c):
        a = c + 1
        if a < c:
            b = a + 1
        else:
            pass
        return b + 1
    restructure_source(udt)


def main():

    def udt(n):
        c = 0
        for i in range(n):
            c += 1
        return c

    restructure_source(udt)


if __name__ == "__main__":
    # main()
    test_if_else()