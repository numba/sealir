from __future__ import annotations

import ast
import inspect
import os
import pathlib
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import starmap
from types import FunctionType, coroutine
from typing import Any, NamedTuple, Sequence, TypeAlias, TypedDict, cast, Coroutine, Generator

from numba_rvsdg.core.datastructures.ast_transforms import (  # type: ignore[import-untyped]
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
    unparse_code,
)

from sealir import ase, grammar, rvsdg
from sealir.rewriter import insert_metadata_map
from sealir.rvsdg import _DEBUG, internal_prefix, pp

from . import grammar as rg

SExpr: TypeAlias = ase.SExpr


re_loc_directive = re.compile(
    r"(?P<line>\d+):(?P<col_offset>\d+)-(?P<end_line>\d+):(?P<end_col_offset>\d+)"
)


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

    def generic_visit(self, node: ast.AST) -> ast.AST:
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

                begin = f"{(stmt.lineno or 0) + offset}:{coloffset + (stmt.col_offset or 0)}"
                end = f"{(stmt.end_lineno or 0) + offset}:{coloffset + (stmt.end_col_offset or 0)}"
                newbody.append(ast.Expr(ast.Constant(f"#loc: {begin}-{end}")))
                newbody.append(self.visit(stmt))
            node.body = newbody
            return node
        else:
            return super().generic_visit(node)


class SourceInfo(NamedTuple):
    """Information about the source code of a function."""

    srcfile: str
    srclineoffset: int
    lines: list[str]
    line_offset: int
    col_offset: int


def get_source_info(function: FunctionType):
    """Extract source file information and offsets for a function."""
    srcfile = inspect.getsourcefile(function)
    assert srcfile is not None
    try:
        srcfile = str(pathlib.Path(srcfile).relative_to(os.getcwd()))
    except ValueError:
        pass  # ignore fail to get relative path
    srclineoffset = min(
        (0xFFFFFFFF if ln is None else ln)
        for _, _, ln in function.__code__.co_lines()
    )
    #   Get column offset
    lines, line_offset = inspect.getsourcelines(function)
    re_space = re.compile(r"\s*")

    def count_space(sub):
        if sub.strip() == "":
            return  # to be filtered out
        m = re_space.match(sub)
        return len(m.group())

    col_offset = min(filter(bool, map(count_space, lines)))

    return SourceInfo(srcfile, srclineoffset, lines, line_offset, col_offset)


def get_source_info_from_ast(ast_node: ast.AST):
    """Extract source file information from an AST node."""
    # Try to get filename from the AST node
    srcfile = getattr(ast_node, "filename", "<unknown>")

    if srcfile == "<unknown>":
        # Fallback to trying to get it from nested nodes
        for node in ast.walk(ast_node):
            if hasattr(node, "filename") and node.filename:
                srcfile = node.filename
                break

    # Convert to relative path if possible
    if srcfile != "<unknown>":
        try:
            srcfile = str(pathlib.Path(srcfile).relative_to(os.getcwd()))
        except (ValueError, OSError):
            pass  # ignore fail to get relative path

    # Get line offset from AST node
    srclineoffset = getattr(ast_node, "lineno", 1)

    # For AST input, we don't have the original source lines
    # So we generate placeholder lines or use ast.unparse
    try:
        lines = ast.unparse(ast_node).splitlines()
        lines = [line + "\n" for line in lines]  # Add newlines back
    except AttributeError:
        # Fallback for older Python versions without ast.unparse
        lines = ["<AST source unavailable>\n"]

    line_offset = srclineoffset

    # Calculate column offset from AST
    col_offset = getattr(ast_node, "col_offset", 0)

    return SourceInfo(srcfile, srclineoffset, lines, line_offset, col_offset)


def restructure_source(function: FunctionType | ast.AST):
    # Get source info based on input type
    if isinstance(function, ast.AST):
        source_info = get_source_info_from_ast(function)
        raw_tree = function
    else:
        source_info = get_source_info(function)
        [raw_tree] = unparse_code(function)

    # Bake in the source location into the AST as dangling strings
    # (like docstrings)
    raw_tree = BakeInLocAsStr(
        source_info.srcfile,
        source_info.srclineoffset - 1,
        source_info.col_offset,
    ).visit(raw_tree)

    # SCF
    ast2scfg_transformer = AST2SCFGTransformer([raw_tree])
    astcfg = ast2scfg_transformer.transform_to_ASTCFG()
    scfg = astcfg.to_SCFG()
    scfg.restructure()
    scfg2ast = SCFG2ASTTransformer()

    # Handle original AST differently based on input type
    if isinstance(function, ast.AST):
        original_ast = function
    else:
        original_ast = unparse_code(function)[0]

    transformed_ast = scfg2ast.transform(original=original_ast, scfg=scfg)

    transformed_ast = ast.fix_missing_locations(transformed_ast)
    # Roundtrip it to fix source location
    inter_source = ast.unparse(transformed_ast)
    [transformed_ast] = ast.parse(inter_source).body
    debugger = SourceDebugInfo(
        source_info.line_offset,
        source_info.lines,
        inter_source.splitlines(),
        suppress=not _DEBUG,
    )

    prgm = rvsdg.convert_to_sexpr(transformed_ast, source_info.line_offset)

    grm = rg.Grammar(prgm._tape)

    rvsdg_out = convert_to_rvsdg(grm, prgm)
    return rvsdg_out, debugger


class SourceDebugInfo:

    def __init__(
        self,
        source_offset: int,
        src_lines: Sequence[str],
        inter_lines: Sequence[str],
        *,
        stream=None,
        suppress: bool = False,
    ):
        self._source_info = {
            i: ln.rstrip()
            for i, ln in enumerate(src_lines, start=source_offset)
        }
        self._inter_source_info = dict(enumerate(inter_lines, start=1))
        if stream is None:
            stream = sys.stderr
        self.stream = stream
        self.suppress = suppress

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

    def print(self, *args, **kwargs) -> None:
        if self.suppress:
            return
        print(">", *args, **kwargs, file=self.stream)


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

    def get_loc(self) -> rg.Loc:
        li = self.lineinfos
        return rg.Loc(
            filename=self.file,
            line_first=int(li["line"]),
            line_last=int(li["end_line"]),
            col_first=int(li["col_offset"]),
            col_last=int(li["end_col_offset"]),
        )


@dataclass(frozen=True)
class RvsdgizeCtx:
    grm: rg.Grammar
    scope_stack: list[Scope] = field(default_factory=list)
    scope_map: dict[SExpr, Scope] = field(default_factory=dict)
    loc_tracker: LocTracker = field(default_factory=LocTracker)

    @property
    def scope(self) -> Scope:
        return self.scope_stack[-1]

    def initialize_scope(self, rb: SExpr) -> None:
        grm = self.grm
        varmap = self.scope.varmap
        for i, k in enumerate(rb.inports):
            varmap[k] = grm.write(rg.Unpack(val=rb, idx=i))

    @contextmanager
    def new_function(self, node: SExpr):
        scope = Scope(kind="function")

        scope.varmap[internal_prefix("io")] = self.grm.write(rg.IO())
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
        scope.varmap[name] = self.grm.write(rg.ArgRef(idx=i, name=name))

    def load_var(self, name: str) -> SExpr:
        assert name != internal_prefix("_")
        scope = self.scope
        if v := scope.varmap.get(name):
            return v
        else:
            # Treat as global
            return self.grm.write(
                rg.PyLoadGlobal(io=self.load_io(), name=name)
            )

    def store_var(self, name: str, value: SExpr) -> None:
        if name == internal_prefix("_"):
            # A store to unused name
            pass
        else:
            scope = self.scope
            scope.varmap[name] = value

    def store_io(self, value: SExpr) -> None:
        self.store_var(internal_prefix("io"), value)

    def load_io(self) -> SExpr:
        return self.load_var(internal_prefix("io"))

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
        io, res = (grm.write(rg.Unpack(val=written, idx=i)) for i in range(2))
        self.store_io(io)
        return res

    def read_directive(self, directive: Directive) -> None:
        match directive.kind:
            case "#file":
                self.loc_tracker.file = directive.content
            case "#loc":
                m = re_loc_directive.match(directive.content)
                assert m, "invalid #loc directive"
                self.loc_tracker.lineinfos.update(m.groupdict())  # type: ignore[typeddict-item]
            case _:
                raise ValueError(f"unknown directive: {directive}")

    def write_src_loc(self):
        return self.grm.write(self.loc_tracker.get_loc())


def unpack_pystr(sexpr: SExpr) -> str | None:
    match sexpr:
        case rg.PyStr(text):
            return text
        case _:
            return None


def unpack_pyast_name(sexpr: SExpr) -> str:
    assert sexpr._head == "PyAst_Name", sexpr
    return cast(str, sexpr._args[0])


def is_directive(text: str) -> bool:
    return text.startswith("#file:") or text.startswith("#loc:")


def parse_directive(text: str) -> Directive | None:
    if not is_directive(text):
        return None

    def cleanup(s: str) -> str:
        return s.strip()

    kind, content = map(cleanup, text.split(":", 1))
    return Directive(kind=kind, content=content)


@dataclass(frozen=True)
class Directive:
    kind: str
    content: str


def get_vars_defined(body: ase.SExpr) -> set[str]:
    names = set()

    class VarDefined(ase.TreeVisitor):
        def visit(self, expr: SExpr):
            match (expr._head, expr._args):
                case ("PyAst_Name", (str(name), "store", loc)):
                    if name != internal_prefix("_"):  # ignore unamed assign
                        names.add(name)

    ase.apply_bottomup(body, VarDefined(), reachable="compute")
    return names


def rvsdgization(expr: SExpr, state: RvsdgizeState) -> Generator[Any, Any, Any]:
    ctx = state.context
    grm = ctx.grm

    def fixup(regend, updated_vars: Sequence[str]):
        if updated_vars == tuple(p.name for p in regend.ports):
            return regend
        else:
            # Insert Undef for the missing varnames
            oldports = dict((p.name, p.value) for p in regend.ports)
            portvals = tuple(
                oldports[k] if k in oldports else grm.write(rg.Undef(k))
                for k in updated_vars
            )
            return grm.write(
                rg.RegionEnd(
                    begin=regend.begin,
                    ports=tuple(
                        grm.write(rg.Port(name=k, value=v))
                        for k, v in zip(updated_vars, portvals, strict=True)
                    ),
                )
            )

    def extract_name_load(var_load: SExpr) -> str:
        match var_load._head, var_load._args:
            case "PyAst_Name", (str(name), "load", _):
                return name
        raise NotImplementedError(var_load)

    def write_loc(pyloc: SExpr) -> SExpr:
        line_first, col_first, line_last, col_last = pyloc._args
        return grm.write(
            rg.Loc(
                filename="",
                line_first=cast(int, line_first),
                line_last=cast(int, line_last),
                col_first=cast(int, col_first),
                col_last=cast(int, col_last),
            )
        )

    match (expr._head, expr._args):
        case ("PyAst_FunctionDef", (str(fname), args, body, interloc)):
            with ctx.new_function(expr):
                args = yield args
                body = yield body

                def strip_unused(term: rg.RegionEnd):
                    ports = [
                        p
                        for p in term.ports
                        if p.name.startswith(internal_prefix(""))
                    ]
                    return grm.write(
                        rg.RegionEnd(begin=term.begin, ports=tuple(ports))
                    )

                return grm.write(
                    rg.Func(
                        fname=fname,
                        args=cast(rg.Args, args),
                        body=strip_unused(cast(rg.RegionEnd, body)),
                    )
                )
        case ("PyAst_arg", (str(name), annotation, interloc)):
            return grm.write(
                rg.ArgSpec(name=name, annotation=(yield annotation))
            )
        case ("PyAst_arguments", args):
            arg_done = []
            for i, arg in enumerate(args):
                x = yield arg
                arg_done.append(x)
                ctx.add_argument(i, x.name)
            return grm.write(rg.Args(tuple(arg_done)))
        case ("PyAst_keyword", (str(name), value, interloc)):
            return grm.write(rg.Keyword(name=name, value=(yield value)))
        case ("PyAst_block", body):
            vars = sorted(ctx.scope.varmap)
            begin = grm.write(
                rg.RegionBegin(
                    inports=tuple(vars), attrs=grm.write(rg.Attrs(()))
                )
            )
            with ctx.new_block(expr) as scope:
                ctx.initialize_scope(begin)
                for stmt in body:
                    (yield stmt)
                vars = sorted(scope.varmap)
                port_values = ctx.load_vars(vars)

            ports = [
                grm.write(rg.Port(name=k, value=v))
                for k, v in zip(vars, port_values, strict=True)
            ]
            return grm.write(rg.RegionEnd(begin=cast(rg.RegionBegin, begin), ports=tuple(ports)))

        case ("PyAst_If", (test, SExpr() as body, SExpr() as orelse, interloc)):
            vars = sorted(ctx.scope.varmap)
            operands = ctx.load_vars(vars)
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
                rg.IfElse(
                    cond=cond,
                    body=br_true,
                    orelse=br_false,
                    operands=operands,
                )
            )
            # update scope
            for i, k in enumerate(updated_vars):
                ctx.store_var(k, grm.write(rg.Unpack(val=swt, idx=i)))

        case ("PyAst_While", (SExpr() as loopcondvar, SExpr() as body, interloc)):
            # Populate variables that are not yet defined but will be defined in
            # the loop.
            names = get_vars_defined(body)
            loopvar = extract_name_load(loopcondvar)
            for k in names - set(ctx.scope.varmap):
                ctx.store_var(k, grm.write(rg.Undef(k)))

            loopbody = yield body

            # rewrite the loopbody to inject loop-condition as the first port
            def get_loopvar(ports):
                for p in ports:
                    if p.name == loopvar:
                        return p.value
                raise AssertionError("malformed")

            output_loopvar = get_loopvar(loopbody.ports)

            injected = grm.write(
                rg.Port(
                    # Note: All port names must be in sorted order.
                    #       The _loopcond_ prefix is selected to ensure that.
                    name=internal_prefix(
                        f"_loopcond_{len(ctx.scope_stack):04}"
                    ),
                    value=output_loopvar,
                )
            )
            loopbody = grm.write(
                rg.RegionEnd(
                    begin=loopbody.begin, ports=(injected,) + loopbody.ports
                )
            )

            # wrap up the Loop
            updated_vars = ctx.updated_vars([ctx.scope_map[body]])
            operands = ctx.load_vars(cast(rg.RegionBegin, loopbody.begin).inports)
            dow = grm.write(rg.Loop(body=loopbody, operands=operands))
            # update scope
            for i, k in enumerate(updated_vars):
                ctx.store_var(k, grm.write(rg.Unpack(val=dow, idx=i)))

        case ("PyAst_Return", (value, interloc)):
            v = yield value
            ctx.store_var(internal_prefix("ret"), v)

        case ("PyAst_Assign", (rval, *targets, interloc)):
            res = yield rval

            if len(targets) == 1 and isinstance(targets[0], SExpr) and targets[0]._head == "PyAst_Subscript":
                [lhs, indices, loc] = targets[0]._args
                lhs = yield lhs
                indices = yield indices
                rval = yield rval
                setitem = rg.PySetItem(
                    io=ctx.load_io(), obj=cast(SExpr, lhs), index=cast(SExpr, indices), value=cast(SExpr, rval)
                )
                return ctx.insert_io_node(setitem)
            if (
                len(targets) == 1
                and unpack_pyast_name(cast(SExpr, targets[0])) == internal_prefix("_")
                and (pystr := unpack_pystr(res))
                and (directive := parse_directive(pystr))
            ):
                ctx.read_directive(directive)
                return  # end early

            for tar in targets:
                name = unpack_pyast_name(cast(SExpr, tar))
                ctx.store_var(
                    name,
                    grm.write(
                        rg.DbgValue(
                            name=name,
                            value=cast(SExpr, res),
                            srcloc=ctx.write_src_loc(),
                            interloc=write_loc(cast(SExpr, interloc)),
                        )
                    ),
                )
            return

        case ("PyAst_AugAssign", (str(op), target, rhs, interloc)):
            assert isinstance(target, SExpr)
            match target._head, target._args:
                case ("PyAst_Name", (str(varname), "store", _)):
                    pass
                case _:
                    raise AssertionError(target)
            lhs = ctx.load_var(varname)
            rhs = yield rhs
            res = rg.PyInplaceBinOp(op=op, io=ctx.load_io(), lhs=lhs, rhs=cast(SExpr, rhs))
            ctx.store_var(varname, ctx.insert_io_node(res))
            return
        case ("PyAst_UnaryOp", (str(op), operand, interloc)):
            operandval = yield operand
            res = rg.PyUnaryOp(op=op, io=ctx.load_io(), operand=operandval)
            return ctx.insert_io_node(res)

        case ("PyAst_BinOp", (str(op), lhs, rhs, interloc)):
            lhsval = yield lhs
            rhsval = yield rhs
            res = rg.PyBinOp(op=op, io=ctx.load_io(), lhs=lhsval, rhs=rhsval)
            return ctx.insert_io_node(res)

        case ("PyAst_Compare", (str(op), lhs, rhs, interloc)):
            lhsval = yield lhs
            rhsval = yield rhs
            res = rg.PyBinOp(op=op, io=ctx.load_io(), lhs=lhsval, rhs=rhsval)
            return ctx.insert_io_node(res)

        case ("PyAst_Call", (SExpr() as func, SExpr() as posargs, interloc)):
            pos_args = []
            for arg in posargs._args:
                pos_args.append((yield arg))

            call = rg.PyCall(
                func=(yield func), io=ctx.load_io(), args=tuple(pos_args)
            )
            return ctx.insert_io_node(call)

        case (
            "PyAst_CallKwargs",
            (SExpr() as func, SExpr() as posargs, SExpr() as kwargs, interloc),
        ):
            pos_args = []
            for arg in posargs._args:
                pos_args.append((yield arg))
            kws = []
            for arg in kwargs._args:
                kws.append((yield arg))
            call_kwargs = rg.PyCallKwargs(
                func=cast(SExpr, (yield func)),
                io=ctx.load_io(),
                args=cast(rg.Posargs, grm.write(rg.Posargs(tuple(pos_args)))),
                kwargs=cast(rg.Kwargs, grm.write(rg.Kwargs(tuple(kws)))),
            )
            return ctx.insert_io_node(call_kwargs)

        case ("PyAst_Name", (str(name), "load", interloc)):
            return ctx.load_var(name)

        case ("PyAst_Constant_int", (int(value), interloc)):
            return grm.write(rg.PyInt(value))

        case ("PyAst_Constant_float", (float(value), interloc)):
            return grm.write(rg.PyFloat(value))

        case ("PyAst_Constant_bool", (bool(value), interloc)):
            return grm.write(rg.PyBool(value))

        case ("PyAst_Constant_str", (str(value), interloc)):
            return grm.write(rg.PyStr(value))

        case ("PyAst_Constant_complex", (float(real), float(imag), interloc)):
            return grm.write(rg.PyComplex(real=real, imag=imag))

        case ("PyAst_None", (interloc,)):
            return grm.write(rg.PyNone())

        case ("PyAst_Tuple", args):
            *elems, interloc = args
            proc_elems = []
            for el in elems:
                proc_elems.append((yield el))
            return grm.write(rg.PyTuple(elems=tuple(proc_elems)))

        case ("PyAst_List", args):
            *elems, interloc = args
            proc_elems = []
            for el in elems:
                proc_elems.append((yield el))
            return grm.write(rg.PyList(elems=tuple(proc_elems)))

        case ("PyAst_Attribute", (valuexpr, str(attr), interloc)):
            value = yield valuexpr
            return ctx.insert_io_node(
                rg.PyAttr(io=ctx.load_io(), value=cast(SExpr, value), attrname=attr)
            )

        case ("PyAst_Subscript", (valuexpr, indexexpr, interloc)):
            value = yield valuexpr
            index = yield indexexpr
            return ctx.insert_io_node(
                rg.PySubscript(io=ctx.load_io(), value=cast(SExpr, value), index=cast(SExpr, index))
            )

        case ("PyAst_Slice", (lower, upper, step, interloc)):
            lower_val = yield lower
            upper_val = yield upper
            step_val = yield step
            return ctx.insert_io_node(
                rg.PySlice(
                    io=ctx.load_io(),
                    lower=lower_val,
                    upper=upper_val,
                    step=step_val,
                )
            )

        case ("PyAst_Pass", (interloc,)):
            return

        case _:
            raise NotImplementedError(expr)


def format_rvsdg(prgm: SExpr, *, format_attrs=ase.pretty_str) -> str:

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
            case rg.Func(fname=str(fname), args=SExpr() as args, body=body):
                put(f"{fname} = Func {ase.pretty_str(args)}")
                (yield body)
            case rg.RegionBegin(inports=ins, attrs=rg.Attrs() as attrs):
                name = fresh_name()
                heading = f"{name} = Region[{expr._handle}] <- {' '.join(ins)}"
                if not attrs.attrs:
                    put(heading)
                else:
                    fmtattrs = format_attrs(attrs)
                    put(f"{heading}; #attrs {fmtattrs}")
                return name
            case rg.RegionEnd(begin=begin, ports=ports):
                (yield begin)
                put("{")
                outrefs = []
                with indent():
                    for p in ports:
                        ref = yield p.value
                        outrefs.append(f"{p.name}={ref}")
                put(f"}} [{expr._handle}] -> {' '.join(outrefs)}")
            case rg.IfElse(
                cond=cond, body=body, orelse=orelse, operands=operands
            ):
                ops = []
                for op in operands:
                    ops.append((yield op))
                condref = yield cond
                name = fresh_name()
                put(f"{name} = If {condref} <- {' '.join(ops)}")
                with indent():
                    (yield body)
                    put("Else")
                    (yield orelse)
                put(f"Endif")
                return name

            case rg.Loop(body=body, operands=operands):
                ops = []
                for op in operands:
                    ops.append((yield op))

                name = fresh_name()
                put(f"{name} = Loop [{expr._handle}] <- {' '.join(ops)}")
                with indent():
                    (yield body)
                put(f"EndLoop")
                return name

            case rg.PyForLoop(
                iter_arg_idx=int(iter_arg_idx),
                indvar_arg_idx=int(indvar_arg_idx),
                iterlast_arg_idx=int(iterlast_arg_idx),
                body=body,
                operands=operands,
            ):
                ops = []
                for op in operands:
                    ops.append((yield op))

                name = fresh_name()
                put(
                    f"{name} = PyForLoop [{expr._handle}] "
                    f"#iter_arg_idx={iter_arg_idx} "
                    f"#indvar_arg_idx={indvar_arg_idx} "
                    f"#iterlast_arg_idx={iterlast_arg_idx} "
                    f"<- {' '.join(ops)}"
                )
                with indent():
                    (yield body)
                put(f"EndPyForLoop")
                return name

            case rg.Unpack(val=source, idx=int(idx)):
                ref = yield source
                return f"{ref}[{idx}]"
            case rg.ArgRef(idx=int(idx), name=str(name)):
                return f"(ArgRef {idx} {name})"

            case rg.IO():
                return "IO"

            case rg.Undef(str(k)):
                name = fresh_name()
                put(f"{name} = Undef {k}")
                return name

            case rg.DbgValue(name=str(varname), value=value):
                valref = yield value
                name = fresh_name()
                put(f"{name} = DbgValue {varname!r} {valref}")
                return name

            case rg.PyNone():
                name = fresh_name()
                put(f"{name} = PyNone")
                return name

            case rg.PyBool(bool(v)):
                name = fresh_name()
                put(f"{name} = PyBool {v}")
                return name
            case rg.PyInt(int(v)):
                name = fresh_name()
                put(f"{name} = PyInt {v}")
                return name
            case rg.PyFloat(float(v)):
                name = fresh_name()
                put(f"{name} = PyFloat {v}")
                return name
            case rg.PyComplex(float(real), float(imag)):
                name = fresh_name()
                put(f"{name} = PyComplex {real} {imag}")
                return name
            case rg.PyStr(str(v)):
                name = fresh_name()
                put(f"{name} = PyStr {v!r}")
                return name

            case rg.PyTuple(elems):
                elem_args = []
                for el in elems:
                    elem_args.append((yield el))
                name = fresh_name()
                fmt = ", ".join(elem_args)
                put(f"{name} = PyTuple {fmt}")
                return name

            case rg.PyList(elems):
                elem_args = []
                for el in elems:
                    elem_args.append((yield el))
                name = fresh_name()
                fmt = ", ".join(elem_args)
                put(f"{name} = PyList {fmt}")
                return name

            case rg.PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
                ioref = yield io
                lhsref = yield lhs
                rhsref = yield rhs
                name = fresh_name()
                put(f"{name} = PyBinOp {op} {ioref} {lhsref}, {rhsref}")
                return name

            case rg.PyBinOpPure(op=op, lhs=lhs, rhs=rhs):
                lhsref = yield lhs
                rhsref = yield rhs
                name = fresh_name()
                put(f"{name} = PyBinOpPure {op} {lhsref}, {rhsref}")
                return name

            case rg.PyInplaceBinOp(op, io, lhs, rhs):
                ioref = yield io
                lhsref = yield lhs
                rhsref = yield rhs
                name = fresh_name()
                put(
                    f"{name} = PyInplaceBinOp {op} {ioref}, {lhsref}, {rhsref}"
                )
                return name

            case rg.PyUnaryOp(op=op, io=io, operand=operand):
                ioref = yield io
                operandref = yield operand
                name = fresh_name()
                put(f"{name} = PyUnaryOp {op} {ioref} {operandref}")
                return name

            case rg.PyCall(func=func, io=io, args=args):
                funcref = yield func
                ioref = yield io
                argrefs = []
                for arg in args:
                    argrefs.append((yield arg))
                fmtargs = ", ".join(argrefs)
                name = fresh_name()
                put(f"{name} = PyCall {funcref} {ioref} {fmtargs}")
                return name

            case rg.PyCallPure(func=func, args=args):
                funcref = yield func
                argrefs = []
                for arg in args:
                    argrefs.append((yield arg))
                fmtargs = ", ".join(argrefs)
                name = fresh_name()
                put(f"{name} = PyCall {funcref} {fmtargs}")
                return name

            case rg.PyLoadGlobal(io=io, name=str(varname)):
                ioref = yield io
                name = fresh_name()
                put(f"{name} = PyLoadGlobal {ioref} {varname!r}")
                return name

            case rg.PyAttr(io=io, value=value, attrname=attr):
                ioref = yield io
                valref = yield value
                name = fresh_name()
                put(f"{name} = PyAttr {ioref} {valref} {attr!r}")
                return name

            case rg.PySubscript(io=io, value=value, index=index):
                ioref = yield io
                valref = yield value
                indexref = yield index
                name = fresh_name()
                put(f"{name} = PySubscript {ioref} {valref} {indexref}")
                return name

            case rg.DbgValue(
                name=str(varname),
                value=value,
                srcloc=srcloc,
                interloc=interloc,
            ):
                valueref = yield value
                name = fresh_name()
                put(f"{name} = DbgValue {varname} {valueref}")  # TODO loc
                return name

            case rg.Attrs() as attrs:
                return "#attrs " + format_attrs(attrs)
            case _:
                argrefs = []
                for item in expr._args:
                    if isinstance(item, SExpr):
                        text = cast(str, (yield item))
                        if text is None:
                            raise ValueError("malformed for formatting")
                    else:
                        text = repr(item)
                    argrefs.append(text)
                name = fresh_name()
                put(f"{name} = {expr._head} {' '.join(argrefs)}")
                return name

    ase.traverse(prgm, formatter)

    return "\n".join(buffer)


def convert_to_rvsdg(grm: rg.Grammar, prgm: SExpr):
    # pp(prgm)

    state = RvsdgizeState(RvsdgizeCtx(grm=grm))
    memo: dict[SExpr, SExpr] = ase.traverse(prgm, rvsdgization, state) # type: ignore[arg-type]
    insert_metadata_map(memo, "rvsdgization")
    out = memo[prgm]

    # out._tape.render_dot(only_reachable=True).view()

    # print(out._tape.dump())
    # pp(out)

    if _DEBUG:
        print(format_rvsdg(out))
    return out
