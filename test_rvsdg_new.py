from __future__ import annotations

import sys
import os
import re
import ast
import inspect
import pathlib
import builtins
from string import Formatter
import logging
import operator
import time
from itertools import starmap
from collections import ChainMap
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
from functools import reduce
from pprint import pformat, pprint
from textwrap import dedent
from typing import (
    Any,
    Iterable,
    Iterator,
    NamedTuple,
    Sequence,
    TypeAlias,
    cast,
    Type,
    TypedDict,
)
import tempfile

from sealir import ase, grammar, lam
from sealir.rewriter import insert_metadata_map

from sealir import rvsdg
from sealir.rvsdg import _internal_prefix
from sealir.prettyformat.html_format import find_source_md

_DEBUG = True

SExpr: TypeAlias = ase.SExpr

from numba_rvsdg.core.datastructures.ast_transforms import (
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
    unparse_code,
)


re_loc_directive = re.compile(
    r"(?P<line>\d+):(?P<col_offset>\d+)-(?P<end_line>\d+):(?P<end_col_offset>\d+)"
)


def pp(expr: SExpr):
    if _DEBUG:
        print(
            pformat(ase.as_tuple(expr, -1, dedup=True))
            .replace(",", "")
            .replace("'", "")
        )
        # print(pformat(expr.as_dict()))


class BakeInLocAsStr(ast.NodeTransformer):
    """
    A class that bake-in source location into
      the AST as a string preceding
    each statement.
    """

    srcfile: str
    srclineoffset: int
    coloffset: int

    def __init__(self, srcfile: str, srclineoffset: int, coloffset: int):
        self.srcfile = srcfile
        self.srclineoffset = srclineoffset
        self.coloffset = coloffset

    def generic_visit(self, node):
        if hasattr(node, "body"):
            stmt: ast.stmt
            offset = self.srclineoffset
            coloffset = self.coloffset
            newbody = []
            for i, stmt in enumerate(node.body):
                if i == 0 and isinstance(node, ast.FunctionDef):
                    newbody.append(
                        ast.Expr(ast.Constant(f"#file: {self.srcfile}"))
                    )

                begin = f"{stmt.lineno + offset}:{coloffset + stmt.col_offset}"
                end = f"{stmt.end_lineno + offset}:{coloffset + stmt.end_col_offset}"
                newbody.append(ast.Expr(ast.Constant(f"#loc: {begin}-{end}")))
                newbody.append(self.visit(stmt))
            node.body = newbody
            return node
        else:
            return super().generic_visit(node)


def restructure_source(function):
    # TODO: a lot of duplication here
    # Get source info
    srcfile = pathlib.Path(inspect.getsourcefile(function)).relative_to(
        os.getcwd()
    )
    srclineoffset = min(ln for _, _, ln in function.__code__.co_lines())

    #   Get column offset
    lines, line_offset = inspect.getsourcelines(function)
    re_space = re.compile(r"\s*")

    def count_space(sub):
        if sub.strip() == "":
            return 0
        m = re_space.match(sub)
        return len(m.group())

    col_offset = min(map(count_space, lines))
    print("col_offset", col_offset)

    # Bake in the source location into the AST as dangling strings
    # (like docstrings)
    [raw_tree] = unparse_code(function)

    raw_tree = BakeInLocAsStr(srcfile, srclineoffset - 1, col_offset).visit(
        raw_tree
    )
    # SCF
    ast2scfg_transformer = AST2SCFGTransformer([raw_tree])
    astcfg = ast2scfg_transformer.transform_to_ASTCFG()
    scfg = astcfg.to_SCFG()
    scfg.restructure()
    scfg2ast = SCFG2ASTTransformer()
    original_ast = unparse_code(function)[0]
    transformed_ast = scfg2ast.transform(original=original_ast, scfg=scfg)

    transformed_ast = ast.fix_missing_locations(transformed_ast)
    # Roundtrip it to fix source location
    inter_source = ast.unparse(transformed_ast)
    [transformed_ast] = ast.parse(inter_source).body
    debugger = SourceInfoDebugger(
        line_offset, lines, inter_source.splitlines()
    )

    prgm = rvsdg.convert_to_sexpr(transformed_ast, line_offset)

    grm = Grammar(prgm._tape)

    rvsdg_out = convert_to_rvsdg(grm, prgm)
    return rvsdg_out, debugger


class _Root(grammar.Rule):
    pass


class Loc(_Root):
    filename: str
    line_first: int
    line_last: int
    col_first: int
    col_last: int


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


class Loop(_Root):
    body: SExpr
    outs: str
    loopvar: str


class IO(_Root):
    pass


class ArgSpec(_Root):
    name: str
    annotation: SExpr


class Return(_Root):
    io: SExpr
    val: SExpr


class PyNone(_Root): ...


class PyInt(_Root):
    value: int


class PyBool(_Root):
    value: bool


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


class DbgValue(_Root):
    name: str
    value: SExpr
    srcloc: Loc
    "Loc for original source"
    interloc: Loc
    "Loc for intermediate form (SCFG)"


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


class LocDirective(TypedDict):
    line: str
    end_line: str
    col_offset: str
    end_col_offset: str


class LocTracker:
    """Track source location in RvsdgizeCtx

    Tracks inline string that encode `#file` and `#loc` directives
    """

    file: str
    lineinfos: LocDirective

    def __init__(self):
        self.file = "unknown"
        self.lineinfos = {}

    def get_loc(self) -> Loc:
        li = self.lineinfos
        return Loc(
            filename=self.file,
            line_first=int(li["line"]),
            line_last=int(li["end_line"]),
            col_first=int(li["col_offset"]),
            col_last=int(li["end_col_offset"]),
        )


@dataclass(frozen=True)
class RvsdgizeCtx:
    grm: Grammar
    scope_stack: list[Scope] = field(default_factory=list)
    scope_map: dict[SExpr, Scope] = field(default_factory=dict)
    loc_tracker: LocTracker = field(default_factory=LocTracker)

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
        assert name != _internal_prefix("_")
        scope = self.scope
        if v := scope.varmap.get(name):
            return v
        else:
            # Treat as global
            return self.grm.write(PyLoadGlobal(io=self.load_io(), name=name))

    def store_var(self, name: str, value: SExpr) -> None:
        if name == _internal_prefix("_"):
            # A store to unused name
            pass
        else:
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

    def read_directive(self, directive: Directive) -> None:
        match directive.kind:
            case "#file":
                self.loc_tracker.file = directive.content
            case "#loc":
                m = re_loc_directive.match(directive.content)
                assert m, "invalid #loc directive"
                self.loc_tracker.lineinfos.update(m.groupdict())
            case _:
                raise ValueError(f"unknown directive: {directive}")

    def write_src_loc(self):
        return self.grm.write(self.loc_tracker.get_loc())


def unpack_pystr(sexpr: SExpr) -> str:
    match sexpr:
        case PyStr(text):
            return text
        case _:
            raise ValueError(f"expecting PyStr but got {sexpr._head}")


def unpack_pyast_name(sexpr: SExpr) -> str:
    assert sexpr._head == "PyAst_Name"
    return cast(str, sexpr._args[0])


def is_directive(text: str) -> str:
    return text.startswith("#file:") or text.startswith("#loc:")


def parse_directive(text: str) -> Directive | None:
    if not is_directive(text):
        return

    def cleanup(s: str) -> str:
        return s.strip()

    kind, content = map(cleanup, text.split(":", 1))
    return Directive(kind=kind, content=content)


@dataclass(frozen=True)
class Directive:
    kind: str
    content: str


def rvsdgization(expr: ase.BasicSExpr, state: RvsdgizeState):
    ctx = state.context
    grm = ctx.grm

    def prep_varnames(d) -> str:
        return " ".join(d)

    def fixup(regend, updated_vars):
        updated = prep_varnames(updated_vars)

        if updated == regend.outs:
            return regend
        else:
            oldports = dict(
                zip(regend.outs.split(), regend.ports, strict=True)
            )
            ports = tuple(
                oldports[k] if k in oldports else grm.write(Undef(k))
                for k in updated_vars
            )
            return grm.write(
                RegionEnd(begin=regend.begin, outs=updated, ports=ports)
            )

    def extract_name_load(var_load: SExpr) -> str:
        match var_load._head, var_load._args:
            case "PyAst_Name", (str(name), "load", _):
                return name
        raise NotImplementedError(var_load)

    def write_loc(pyloc: SExpr) -> SExpr:
        line_first, col_first, line_last, col_last = pyloc._args
        print("????", pyloc._args)
        return grm.write(
            Loc(
                filename="",
                line_first=line_first,
                line_last=line_last,
                col_first=col_first,
                col_last=col_last,
            )
        )

    match (expr._head, expr._args):
        case ("PyAst_FunctionDef", (str(fname), args, body, interloc)):
            with ctx.new_function(expr):
                return grm.write(
                    Func(fname=fname, args=(yield args), body=(yield body))
                )
        case ("PyAst_arg", (str(name), annotation, interloc)):
            return grm.write(ArgSpec(name=name, annotation=(yield annotation)))
        case ("PyAst_arguments", args):
            arg_done = []
            for i, arg in enumerate(args):
                x = yield arg
                arg_done.append(x)
                ctx.add_argument(i, x.name)
            return grm.write(Args(tuple(arg_done)))
        case ("PyAst_block", body):

            vars = sorted(ctx.scope.varmap)
            begin = grm.write(
                RegionBegin(
                    ins=prep_varnames(vars),
                    ports=ctx.load_vars(vars),
                )
            )
            with ctx.new_block(expr) as scope:
                ctx.initialize_scope(begin)
                for expr in body:
                    (yield expr)
                vars = sorted(scope.varmap)
                ports = ctx.load_vars(vars)
            return grm.write(
                RegionEnd(begin=begin, outs=prep_varnames(vars), ports=ports)
            )

        case ("PyAst_If", (test, body, orelse, interloc)):
            cond = yield test
            br_true = yield body
            br_false = yield orelse
            updated_vars = ctx.updated_vars(
                [ctx.scope_map[body], ctx.scope_map[orelse]]
            )
            # fixup mismatching updated vars
            br_true = fixup(br_true, updated_vars)
            br_false = fixup(br_false, updated_vars)

            swt = grm.write(
                IfElse(
                    cond=cond,
                    body=br_true,
                    orelse=br_false,
                    outs=" ".join(updated_vars),
                )
            )
            # update scope
            for i, k in enumerate(updated_vars):
                ctx.store_var(k, grm.write(Unpack(val=swt, idx=i)))

        case ("PyAst_While", (loopcondvar, body, interloc)):
            loopbody = yield body
            loopvar = extract_name_load(loopcondvar)
            updated_vars = ctx.updated_vars([ctx.scope_map[body]])
            dow = grm.write(
                Loop(
                    body=loopbody, outs=" ".join(updated_vars), loopvar=loopvar
                )
            )
            # update scope
            for i, k in enumerate(updated_vars):
                ctx.store_var(k, grm.write(Unpack(val=dow, idx=i)))

        case ("PyAst_Return", (value, interloc)):
            v = yield value
            ctx.store_var(_internal_prefix("ret"), v)

        case ("PyAst_Assign", (rval, *targets, interloc)):
            res = yield rval
            tar: SExpr

            if (
                len(targets) == 1
                and unpack_pyast_name(targets[0]) == _internal_prefix("_")
                and (directive := parse_directive(unpack_pystr(res)))
            ):
                ctx.read_directive(directive)
                return  # end early

            for tar in targets:
                name = unpack_pyast_name(tar)
                ctx.store_var(
                    name,
                    grm.write(
                        DbgValue(
                            name=name,
                            value=res,
                            srcloc=ctx.write_src_loc(),
                            interloc=write_loc(interloc),
                        )
                    ),
                )
            return

        case ("PyAst_AugAssign", (str(op), target, rhs, interloc)):
            match target._head, target._args:
                case ("PyAst_Name", (str(varname), "store", _)):
                    pass
                case _:
                    raise AssertionError(target)
            lhs = ctx.load_var(varname)
            rhs = yield rhs
            res = PyInplaceBinOp(op=op, io=ctx.load_io(), lhs=lhs, rhs=rhs)
            ctx.store_var(varname, ctx.insert_io_node(res))
            return
        case ("PyAst_UnaryOp", (str(op), operand, interloc)):
            res = PyUnaryOp(op=op, io=ctx.load_io(), operand=(yield operand))
            return ctx.insert_io_node(res)

        case ("PyAst_BinOp", (str(op), lhs, rhs, interloc)):
            res = PyBinOp(
                op=op, io=ctx.load_io(), lhs=(yield lhs), rhs=(yield rhs)
            )
            return ctx.insert_io_node(res)

        case ("PyAst_Compare", (str(op), lhs, rhs, interloc)):
            res = PyBinOp(
                op=op, io=ctx.load_io(), lhs=(yield lhs), rhs=(yield rhs)
            )
            return ctx.insert_io_node(res)

        case ("PyAst_Call", (SExpr() as func, SExpr() as posargs, interloc)):
            proc_args = []
            for arg in posargs._args:
                proc_args.append((yield arg))

            call = PyCall(
                func=(yield func), io=ctx.load_io(), args=tuple(proc_args)
            )
            return ctx.insert_io_node(call)

        case ("PyAst_Name", (str(name), "load", interloc)):
            return ctx.load_var(name)

        case ("PyAst_Constant_int", (int(value), interloc)):
            return grm.write(PyInt(value))

        case ("PyAst_Constant_bool", (bool(value), interloc)):
            return grm.write(PyBool(value))

        case ("PyAst_Constant_str", (str(value), interloc)):
            return grm.write(PyStr(value))

        case ("PyAst_None", (interloc,)):
            return grm.write(PyNone())
        case ("PyAst_Pass", (interloc,)):
            return
        case _:
            raise NotImplementedError(expr)


def format_rvsdg(grm: Grammar, prgm: SExpr) -> str:

    def _inf_counter():
        c = 0
        while True:
            yield c
            c += 1

    buffer = []
    indentlevel = 0
    counter = _inf_counter()

    def fresh_name() -> str:
        return f"${next(counter)}"

    @contextmanager
    def indent():
        nonlocal indentlevel
        indentlevel += 1
        try:
            yield
        finally:
            indentlevel -= 1

    def put(text: str):
        prefix = " " * indentlevel * 2
        buffer.append(prefix + text)

    def formatter(expr: SExpr, state: ase.TraverseState):
        match expr:
            case Func(fname=str(fname), args=args, body=body):
                put(f"{fname} = Func {ase.pretty_str(args)}")
                (yield body)
            case RegionBegin(ins=ins, ports=ports):
                inports = []
                for port in ports:
                    inports.append((yield port))
                name = fresh_name()
                fmtins = starmap(
                    lambda x, y: f"{x}={y}",
                    zip(ins.split(), inports, strict=True),
                )
                put(f"{name} = Region <- {' '.join(fmtins)}")
                return name
            case RegionEnd(begin=begin, outs=str(outs), ports=ports):
                (yield begin)
                put("{")
                outrefs = []
                with indent():
                    for port in ports:
                        outrefs.append((yield port))
                fmtoutports = starmap(
                    lambda x, y: f"{y}={x}",
                    zip(outrefs, outs.split(), strict=True),
                )
                put(f"}} -> {' '.join(fmtoutports)}")
            case IfElse(cond=cond, body=body, orelse=orelse, outs=outs):
                condref = yield cond
                name = fresh_name()
                put(f"{name} = If {condref} ")
                with indent():
                    (yield body)
                    put("Else")
                    (yield orelse)
                put(f"Endif -> {outs}")
                return name

            case Loop(body=body, outs=outs, loopvar=loopvar):
                name = fresh_name()
                put(f"{name} = Loop #{loopvar}")
                with indent():
                    (yield body)
                put(f"EndLoop -> {outs}")
                return name

            case Unpack(val=source, idx=int(idx)):
                ref = yield source
                return f"{ref}[{idx}]"
            case ArgRef(idx=int(idx), name=str(name)):
                return f"(ArgRef {idx} {name})"

            case IO():
                return "IO"

            case Undef(str(k)):
                name = fresh_name()
                put(f"{name} = Undef {k}")
                return name

            case DbgValue(name=str(varname), value=value):
                valref = yield value
                name = fresh_name()
                put(f"{name} = DbgValue {varname!r} {valref}")
                return name

            case PyNone():
                name = fresh_name()
                put(f"{name} = PyNone")
                return name

            case PyBool(bool(v)):
                name = fresh_name()
                put(f"{name} = PyBool {v}")
                return name
            case PyInt(int(v)):
                name = fresh_name()
                put(f"{name} = PyInt {v}")
                return name
            case PyStr(str(v)):
                name = fresh_name()
                put(f"{name} = PyStr {v!r}")
                return name

            case PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioref = yield io
                lhsref = yield lhs
                rhsref = yield rhs
                name = fresh_name()
                put(f"{name} = PyBinOp {op} {ioref} {lhsref}, {rhsref}")
                return name

            case PyInplaceBinOp(op, io, lhs, rhs):
                ioref = yield io
                lhsref = yield lhs
                rhsref = yield rhs
                name = fresh_name()
                put(
                    f"{name} = PyInplaceBinOp {op} {ioref}, {lhsref}, {rhsref}"
                )
                return name

            case PyUnaryOp(op=op, io=io, operand=operand):
                ioref = yield io
                operandref = yield operand
                name = fresh_name()
                put(f"{name} = PyUnaryOp {op} {ioref} {operandref}")
                return name

            case PyCall(func=func, io=io, args=args):
                funcref = yield func
                ioref = yield io
                argrefs = []
                for arg in args:
                    argrefs.append((yield arg))
                fmtargs = ", ".join(argrefs)
                name = fresh_name()
                put(f"{name} = PyCall {funcref} {ioref} {fmtargs}")
                return name

            case PyLoadGlobal(io=io, name=str(varname)):
                ioref = yield io
                name = fresh_name()
                put(f"{name} = PyLoadGlobal {ioref} {varname!r}")
                return name

            case _:
                print("----debug")
                print("\n".join(buffer))
                raise NotImplementedError(expr)

    ase.traverse(prgm, formatter)

    return "\n".join(buffer)


def convert_to_rvsdg(grm: Grammar, prgm: SExpr):
    pp(prgm)

    state = RvsdgizeState(RvsdgizeCtx(grm=grm))
    memo = ase.traverse(prgm, rvsdgization, state)
    insert_metadata_map(memo, "rvsdgization")
    out = memo[prgm]

    # out._tape.render_dot(only_reachable=True).view()

    print(out._tape.dump())
    pp(out)

    print(format_rvsdg(grm, out))
    return out


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
            case RegionBegin(ins=str(ins)):
                names = ins.split()
            case RegionEnd(outs=str(outs)):
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


class SourceInfoDebugger:

    def __init__(
        self,
        source_offset: int,
        src_lines: Sequence[str],
        inter_lines: Sequence[str],
        stream=None,
    ):
        self._source_info = {
            i: ln.rstrip()
            for i, ln in enumerate(src_lines, start=source_offset)
        }
        self._inter_source_info = dict(enumerate(inter_lines, start=1))
        if stream is None:
            stream = sys.stderr
        self.stream = stream

    def show_sources(self) -> str:
        buf = []
        buf.append("original source".center(80, "-"))
        for lno, text in self._source_info.items():
            buf.append(f"{lno:4}|{text}")
        buf.append("inter source".center(80, "-"))
        for lno, text in self._inter_source_info.items():
            buf.append(f"{lno:4}|{text}")
        return "\n".join(buf)

    def set_src_loc(self, srcloc):
        self._srcloc = srcloc

    def set_inter_loc(self, interloc):
        self._interloc = interloc

    def show_source_lines(self):
        loc = self._srcloc
        first = loc.line_first
        last = loc.line_last
        self.print(
            f"At source {loc.filename!r} {first}:{last}".center(80, "-")
        )
        # self.print(ase.as_tuple(loc))
        for lineno in range(first, last + 1):
            linetext = self._source_info[lineno].rstrip()
            self.print(linetext)
            marker = " " * loc.col_first + "^" * (loc.col_last - loc.col_first)
            self.print(marker)

    def show_inter_source_lines(self):
        loc = self._interloc
        first = loc.line_first
        last = loc.line_last
        self.print(
            f"At SCFG source {loc.filename!r} {first}:{last}".center(80, "-")
        )
        # self.print(ase.as_tuple(loc))
        for lineno in range(first, last + 1):
            linetext = self._inter_source_info[lineno].rstrip()
            self.print(linetext)
            marker = " " * loc.col_first + "^" * (loc.col_last - loc.col_first)
            self.print(marker)

    @contextmanager
    def setup(self, srcloc=None, interloc=None):
        if srcloc:
            self.set_src_loc(srcloc)
        if interloc:
            self.set_inter_loc(interloc)
        try:
            yield
        finally:
            self.show_source_lines()
            self.show_inter_source_lines()
            self.print("=" * 80)

    def print(self, *args, **kwargs):
        print(">", *args, **kwargs, file=self.stream)


def execute(
    prgm: SExpr,
    callargs: tuple,
    callkwargs: dict,
    *,
    init_scope: dict | None = None,
    init_state: ase.TraverseState | None = None,
    init_memo: dict | None = None,
    debugger: SourceInfoDebugger,
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

    def ensure_io(expect_io: Any) -> Type[IO]:
        assert expect_io is IO, expect_io
        return expect_io

    def runner(expr: SExpr, state: ase.TraverseState):

        match expr:
            case Func(fname=str(fname), args=funcargs, body=body):
                # assume we always evaluate a function
                assert isinstance(funcargs, Args)
                params = []
                for arg in funcargs.arguments:
                    assert isinstance(arg, ArgSpec)
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
                return ports.get_by_name("!ret")
            case RegionBegin(ins=ins, ports=ports):
                paired = zip(ins.split(), ports, strict=True)
                ports = []
                for k, v in paired:
                    val = yield v
                    ports.append(val)
                    scope()[k] = val
                return EvalPorts(parent=expr, values=tuple(ports))

            case RegionEnd(begin=begin, outs=str(outs), ports=ports):
                inports = yield begin
                with push():
                    inports.update_scope(scope())
                    debugger.print("In region", dict(scope()))
                    outvals = []
                    for port in ports:
                        outvals.append((yield port))
                    return EvalPorts(expr, tuple(outvals))

            case IfElse(cond=cond, body=body, orelse=orelse, outs=outs):
                condval = yield cond
                if condval:
                    ports = yield body
                else:
                    ports = yield orelse
                ports.update_scope(scope())
                debugger.print("end if", dict(scope()))
                return EvalPorts(expr, ports.values)

            case Loop(body=body, outs=outs, loopvar=loopvar):
                cond = True
                assert isinstance(body, RegionEnd)
                begin = body.begin
                memo = {}
                memo[begin] = yield begin

                while cond:
                    ports = execute(
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

            case Unpack(val=source, idx=int(idx)):
                ports = yield source
                return ports[idx]

            case ArgRef(idx=int(idx), name=str(name)):
                return scope()[name]

            case IO():
                return IO

            case Undef(str(name)):
                return Undef(name)

            case DbgValue(
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

            case PyNone():
                return None

            case PyBool(bool(v)):
                return v

            case PyInt(int(v)):
                return v

            case PyStr(str(v)):
                return v

            case PyUnaryOp(op=op, io=io, operand=operand):
                ioval = ensure_io((yield io))
                operandval = yield operand
                match op:
                    case "not":
                        res = not operandval
                    case _:
                        raise NotImplementedError(op)
                return EvalPorts(expr, (ioval, res))

            case PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
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

            case PyInplaceBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioval = ensure_io((yield io))
                lhsval = yield lhs
                rhsval = yield rhs
                match op:
                    case "+":
                        res = operator.iadd(lhsval, rhsval)
                    case _:
                        raise NotImplementedError(op)
                return EvalPorts(expr, (ioval, res))

            case PyCall(func=func, io=io, args=args):
                funcval = yield func
                ioval = ensure_io((yield io))
                argvals = []
                for arg in args:
                    argvals.append((yield arg))
                out = funcval(*argvals)
                return EvalPorts(expr, values=tuple([ioval, out]))

            case PyLoadGlobal(io=io, name=str(varname)):
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


def test_if_else():

    def udt(c):
        a = c - 1
        if a < c:
            b = a + 2
        else:
            pass
        return b + 3

    rvsdg_ir, debugger = restructure_source(udt)
    print(debugger.show_sources())
    args = (10,)
    kwargs = {}
    res = execute(rvsdg_ir, args, kwargs, debugger=debugger)
    print("res =", res)
    assert res == udt(*args, **kwargs)


def test_for_loop():

    def udt(n):
        c = 0
        for i in range(n):
            c += i
        return c

    rvsdg_ir, debugger = restructure_source(udt)
    print(debugger.show_sources())
    args = (10,)
    kwargs = {}
    res = execute(rvsdg_ir, args, kwargs, debugger=debugger)
    print("res =", res)
    assert res == udt(*args, **kwargs)


if __name__ == "__main__":
    # test_for_loop()
    test_if_else()
