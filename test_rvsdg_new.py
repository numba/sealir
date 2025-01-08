from __future__ import annotations

import ast
import inspect
import logging
import operator
import time
from collections import ChainMap
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import reduce
from pprint import pformat, pprint
from textwrap import dedent
from typing import Any, Iterable, Iterator, NamedTuple, Sequence, TypeAlias

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

    varinfo = rvsdg.find_variable_info(prgm)
    # _logger.debug(varinfo)

    # _logger.debug("find_variable_info", time.time() - t_start)

    grm = Grammar(prgm._tape)

    rvsdg_out = convert_to_rvsdg(grm, prgm, varinfo)

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

class PyBinOp(_Root):
    op: str
    io: SExpr
    lhs: SExpr
    rhs: SExpr

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
        scope.varmap[".io"] = self.grm.write(IO())
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
            raise NameError(name)

    def store_var(self, name: str, value: SExpr) -> None:
        scope = self.scope
        scope.varmap[name] = value

    def store_io(self, value: SExpr) -> None:
        self.store_var(".io", value)

    def load_io(self) -> SExpr:
        return self.load_var(".io")

    def updated_vars(self, scopelist) -> list[str]:
        updated = set()
        for scope in scopelist:
            updated.update(scope.varmap)

        return sorted(updated)

    def load_vars(self, names) -> tuple[SExpr, ...]:
        return tuple(self.load_var(k) for k in names)



def rvsdgization(expr: ase.BasicSExpr, state: RvsdgizeState):
    ctx = state.context
    grm = ctx.grm
    def prep_varnames(d) -> str:
        return ' '.join(d)
    def unpack(val):
        io = grm.write(Unpack(val=val, idx=0))
        res = grm.write(Unpack(val=val, idx=1))
        return io, res

    match (expr._head, expr._args):
        case ("PyAst_FunctionDef", (str(fname), args, body, loc)):
            with ctx.new_function(expr):
                return grm.write(Func(fname=fname, args=(yield args), body=(yield body)))
        case ("PyAst_arg", (str(name), annotation, loc)):
            arg = grm.write(ArgSpec(name=name, annotation=(yield annotation)))
            return arg
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
            io, cond = unpack((yield test))
            ctx.store_io(io)
            br_true = (yield body)
            br_false = (yield orelse)
            updated_vars = ctx.updated_vars([ctx.scope_map[body], ctx.scope_map[orelse]])
            # fixup mismatching updated vars
            def fixup(regend, updated_vars):
                updated = prep_varnames(updated_vars)

                if updated == regend.outs:
                    return regend
                else:
                    oldports = dict(zip(regend.outs.split(), regend.ports))
                    ports = tuple(oldports[k] if k in oldports else grm.write(Undef(k))
                                  for k in updated_vars)
                    return grm.write(RegionEnd(begin=regend.begin, outs=updated, ports=ports))
            br_true = fixup(br_true, updated_vars)
            br_false = fixup(br_false, updated_vars)

            swt = grm.write(IfElse(cond=cond, body=br_true, orelse=br_false, outs=' '.join(updated_vars)))
            # update scope
            for i, k in enumerate(updated_vars):
                ctx.store_var(k, grm.write(Unpack(val=swt, idx=i)))


        case ("PyAst_Return", (value, loc)):
            v = (yield value)
            ctx.store_var(".ret", v)

        case ("PyAst_Assign", (rval, *targets, loc)):
            res = (yield rval)
            tar: SExpr
            for tar in targets:
                assert tar._head == "PyAst_Name", tar
                name = tar._args[0]
                ctx.store_var(name, res)
            return

        case ("PyAst_BinOp", (str(op), lhs, rhs, loc)):
            res = grm.write(PyBinOp(op=op, io=ctx.load_io(),
                                    lhs=(yield lhs), rhs=(yield rhs)))
            io, res = unpack(res)
            ctx.store_io(io)
            return res

        case ("PyAst_Compare", (str(op), lhs, rhs, loc)):
            res =  grm.write(PyBinOp(op=op, io=ctx.load_io(),
                                    lhs=(yield lhs), rhs=(yield rhs)))
            io, res = unpack(res)
            ctx.store_io(io)
            return res

        case ("PyAst_Name", (str(name), "load", loc)):
            return ctx.load_var(name)

        case ("PyAst_Constant_int", (int(value), loc)):
            return grm.write(PyInt(value))

        case ("PyAst_None", (loc,)):
            return grm.write(PyNone())
        case ("PyAst_Pass", (loc,)):
            return
        case _:
            raise NotImplementedError(expr)



def convert_to_rvsdg(grm: Grammar, prgm: SExpr, varinfo: rvsdg.VariableInfo):
    pp(prgm)

    state = RvsdgizeState(RvsdgizeCtx(grm=grm))
    out = ase.traverse(prgm, rvsdgization, state)[prgm]

    print(out._tape.dump())
    pp(out)

    out._tape.render_dot(only_reachable=True).view()



def udt(c):
    a = c + 1
    if a < c:
        b = a + 1
    else:
        pass
    return b

def main():
    restructure_source(udt)


if __name__ == "__main__":
    main()