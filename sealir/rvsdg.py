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

from numba_rvsdg.core.datastructures.ast_transforms import (
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
    unparse_code,
)

from sealir import ase, grammar, lam
from sealir.rewriter import TreeRewriter

_logger = logging.getLogger(__name__)


_DEBUG = False
_DEBUG_HTML = False

SExpr: TypeAlias = ase.SExpr


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


class ConvertToSExpr(ast.NodeTransformer):
    _tape: ase.Tape
    _first_line: int

    def __init__(self, tape, first_line):
        super().__init__()
        self._tape = tape
        self._first_line = first_line

    def generic_visit(self, node: ast.AST):
        raise NotImplementedError(ast.dump(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> SExpr:
        # attrs = ("name", "args", "body", "decorator_list", "returns", "type_comment", "type_params")
        fname = node.name
        args = self.visit(node.args)
        body = [self.visit(stmt) for stmt in node.body]
        # TODO: "decorator_list" "returns", "type_comment", "type_params",
        # decorator_list = [self.visit(x) for x in node.decorator_list]
        assert not node.decorator_list
        return self._tape.expr(
            "PyAst_FunctionDef",
            fname,
            args,
            self._tape.expr("PyAst_block", *body),
            self.get_loc(node),
        )

    def visit_Pass(self, node: ast.Return) -> SExpr:
        return self._tape.expr("PyAst_Pass", self.get_loc(node))

    def visit_Return(self, node: ast.Return) -> SExpr:
        return self._tape.expr(
            "PyAst_Return",
            self.visit(node.value),
            self.get_loc(node),
        )

    def visit_arguments(self, node: ast.arguments) -> SExpr:
        # TODO
        assert not node.posonlyargs
        assert not node.kwonlyargs
        assert not node.kw_defaults
        assert not node.defaults
        return self._tape.expr(
            "PyAst_arguments",
            *map(self.visit, node.args),
        )

    def visit_arg(self, node: ast.arg) -> SExpr:
        return self._tape.expr(
            "PyAst_arg",
            node.arg,
            (
                self.visit(node.annotation)
                if node.annotation is not None
                else self._tape.expr("PyAst_None", self.get_loc(node))
            ),
            self.get_loc(node),
        )

    def visit_Name(self, node: ast.Name) -> SExpr:
        match node.ctx:
            case ast.Load():
                ctx = "load"
            case ast.Store():
                ctx = "store"
            case _:
                raise NotImplementedError(node.ctx)
        return self._tape.expr("PyAst_Name", node.id, ctx, self.get_loc(node))

    def visit_Expr(self, node: ast.Expr) -> SExpr:
        return self._tape.expr(
            "PyAst_Assign",
            self.visit(node.value),
            # "!_" is a special name for unused variable
            self._tape.expr(
                "PyAst_Name",
                _internal_prefix("_"),
                "store",
                self.get_loc(node),
            ),
            self.get_loc(node),
        )

    def visit_Assign(self, node: ast.Assign) -> SExpr:
        return self._tape.expr(
            "PyAst_Assign",
            self.visit(node.value),
            *map(self.visit, node.targets),
            self.get_loc(node),
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> SExpr:
        return self._tape.expr(
            "PyAst_AugAssign",
            self.map_op(node.op),
            self.visit(node.target),
            self.visit(node.value),
            self.get_loc(node),
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> SExpr:
        return self._tape.expr(
            "PyAst_UnaryOp",
            self.map_op(node.op),
            self.visit(node.operand),
            self.get_loc(node),
        )

    def visit_BinOp(self, node: ast.BinOp) -> SExpr:
        return self._tape.expr(
            "PyAst_BinOp",
            self.map_op(node.op),
            self.visit(node.left),
            self.visit(node.right),
            self.get_loc(node),
        )

    def visit_Compare(self, node: ast.Compare) -> SExpr:
        [(op, comp)] = zip(node.ops, node.comparators, strict=True)  # TODO
        match op:
            case ast.NotEq():
                opname = "!="
            case ast.Lt():
                opname = "<"
            case ast.Gt():
                opname = ">"
            case ast.In():
                opname = "in"
            case _:
                raise NotImplementedError(op)
        return self._tape.expr(
            "PyAst_Compare",
            opname,
            self.visit(node.left),
            self.visit(comp),
            self.get_loc(node),
        )

    def visit_Attribute(self, node: ast.Attribute) -> SExpr:
        return self._tape.expr(
            "PyAst_Attribute",
            self.visit(node.value),
            node.attr,
            self.get_loc(node),
        )

    def visit_Subscript(self, node: ast.Subscript) -> SExpr:
        return self._tape.expr(
            "PyAst_Subscript",
            self.visit(node.value),
            self.visit(node.slice),
            self.get_loc(node),
        )

    def visit_Call(self, node: ast.Call) -> SExpr:
        # TODO
        assert not node.keywords
        posargs = self._tape.expr(
            "PyAst_callargs_pos", *map(self.visit, node.args)
        )
        return self._tape.expr(
            "PyAst_Call",
            self.visit(node.func),
            posargs,
            self.get_loc(node),
        )

    def visit_While(self, node: ast.While) -> SExpr:
        # ("test", "body", "orelse")
        test = self.visit(node.test)
        body = self._tape.expr(
            "PyAst_block",
            *map(self.visit, node.body),
        )
        assert not node.orelse
        return self._tape.expr(
            "PyAst_While",
            test,
            body,
            self.get_loc(node),
        )

    def visit_If(self, node: ast.If) -> SExpr:
        test = self.visit(node.test)
        body = self._tape.expr(
            "PyAst_block",
            *map(self.visit, node.body),
        )
        orelse = self._tape.expr(
            "PyAst_block",
            *map(self.visit, node.orelse),
        )
        return self._tape.expr(
            "PyAst_If",
            test,
            body,
            orelse,
            self.get_loc(node),
        )

    def visit_Constant(self, node: ast.Constant) -> SExpr:
        if node.kind is None:
            match node.value:
                case bool():
                    return self._tape.expr(
                        "PyAst_Constant_bool",
                        node.value,
                        self.get_loc(node),
                    )
                case int():
                    return self._tape.expr(
                        "PyAst_Constant_int",
                        node.value,
                        self.get_loc(node),
                    )
                case complex():
                    return self._tape.expr(
                        "PyAst_Constant_complex",
                        node.value.real,
                        node.value.imag,
                        self.get_loc(node),
                    )
                case None:
                    return self._tape.expr(
                        "PyAst_None",
                        self.get_loc(node),
                    )
                case str():
                    return self._tape.expr(
                        "PyAst_Constant_str",
                        node.value,
                        self.get_loc(node),
                    )
        raise NotImplementedError(ast.dump(node))

    def visit_Tuple(self, node: ast.Tuple) -> SExpr:
        match node:
            case ast.Tuple(elts=[*elements], ctx=ast.Load()):
                # e.g. Tuple(elts=[Constant(value=0)], ctx=Load())
                return self._tape.expr(
                    "PyAst_Tuple",
                    *map(self.visit, elements),
                    self.get_loc(node),
                )

        raise NotImplementedError(ast.dump(node))

    def visit_List(self, node: ast.List) -> SExpr:
        match node:
            case ast.List(elts=[*elements], ctx=ast.Load()):
                return self._tape.expr(
                    "PyAst_List",
                    *map(self.visit, elements),
                    self.get_loc(node),
                )

        raise NotImplementedError(ast.dump(node))

    def map_op(self, node: ast.operator) -> str:
        match node:
            case ast.Add():
                return "+"
            case ast.Sub():
                return "-"
            case ast.Mult():
                return "*"
            case ast.Div():
                return "/"
            case ast.FloorDiv():
                return "//"
            case ast.Not():
                return "not"
            case _:
                raise NotImplementedError(node)

    def get_loc(self, node: ast.AST) -> SExpr:
        return self._tape.expr(
            "PyAst_loc",
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
        )


def convert_to_sexpr(node: ast.AST, first_line: int):
    with ase.Tape() as stree:
        out = ConvertToSExpr(stree, first_line).visit(node)
    if _DEBUG:
        pprint(ase.as_tuple(out, depth=-1))
    return out


@dataclass(frozen=True)
class VariableInfo:
    nonlocals: set[str]
    args: tuple[str]
    loaded: set[str]
    stored: set[str]
    # expr_effect: dict
    # block_effect: dict
    # assign_map: dict


def find_variable_info(expr: SExpr):
    loaded: dict[str, SExpr] = {}
    stored: dict[str, SExpr] = {}

    def get_args(expr: SExpr) -> Iterator[str]:
        _, first_def = next(
            ase.search_descendants(
                expr, lambda x: x._head == "PyAst_FunctionDef"
            )
        )
        assert first_def == expr
        (fname, args, block, loc) = expr._args
        for arg in args._args:
            match arg:
                case ase.BasicSExpr("PyAst_arg", (name, *_)):
                    yield name
                case _:
                    raise AssertionError(arg)

    arguments = tuple(get_args(expr))

    class FindVariableUse(ase.TreeVisitor):
        def visit(self, expr: SExpr):
            match expr:
                case ase.BasicSExpr("PyAst_Name", (name, "load", loc)):
                    loaded[name] = expr
                case ase.BasicSExpr("PyAst_Name", (name, "store", loc)):
                    stored[name] = expr
                case ase.BasicSExpr("PyAst_arg", (name, anno, loc)):
                    stored[name] = expr
                case _ if expr._head in {"PyAst_Name", "PyAst_arg"}:
                    raise AssertionError(expr)

    ase.apply_bottomup(expr, FindVariableUse())

    nonlocals = set()
    for k in loaded:
        if k not in stored:
            nonlocals.add(k)

    @dataclass(frozen=True)
    class VarEffect:
        defs: set[str] = field(default_factory=set)
        uses: set[str] = field(default_factory=set)

        def define(self, name) -> None:
            self.defs.add(name)

        def use(self, name) -> None:
            self.uses.add(name)

        def merge(self, other: VarEffect) -> VarEffect:
            self.defs.update(other.defs)
            self.uses.update(other.uses)
            return self

    return VariableInfo(
        nonlocals=set(nonlocals),
        args=arguments,
        loaded=set(loaded),
        stored=set(stored),
    )


class _Root(grammar.Rule):
    pass


class Py_Return(_Root):
    retval: ase.SExpr


class Py_Args(_Root):
    args: tuple[ase.SExpr, ...]


class Py_Undef(_Root):
    pass


class Py_None(_Root):
    pass


class Py_Pass(_Root):
    pass


class Py_Tuple(_Root):
    elts: tuple[ase.SExpr, ...]


class Py_List(_Root):
    elts: tuple[ase.SExpr, ...]


class Py_GlobalLoad(_Root):
    name: str


class Py_Int(_Root):
    value: int


class Py_Bool(_Root):
    value: bool


class Py_Complex(_Root):
    real: float
    imag: float


class Py_Str(_Root):
    value: str


class Py_If(_Root):
    test: ase.SExpr
    then: ase.SExpr
    orelse: ase.SExpr


class Py_While(_Root):
    test: ase.SExpr
    body: ase.SExpr


class Py_UnaryOp(_Root):
    opname: str
    iostate: ase.SExpr
    arg: ase.SExpr


class Py_BinOp(_Root):
    opname: str
    iostate: ase.SExpr
    lhs: ase.SExpr
    rhs: ase.SExpr


class Py_InplaceBinOp(_Root):
    opname: str
    iostate: ase.SExpr
    lhs: ase.SExpr
    rhs: ase.SExpr


class Py_Call(_Root):
    iostate: ase.SExpr
    callee: ase.SExpr
    args: tuple[ase.SExpr, ...]


class Py_GetAttr(_Root):
    attr: str
    iostate: ase.SExpr
    value: ase.SExpr


class Py_GetItem(_Root):
    iostate: ase.SExpr
    value: ase.SExpr
    slice: ase.SExpr


class Py_Compare(_Root):
    opname: str
    iostate: ase.SExpr
    lhs: ase.SExpr
    rhs: ase.SExpr


class Py_Func(_Root):
    name: str
    args: ase.SExpr
    body: ase.SExpr


class VarLoad(_Root):
    name: str


class VarStore(_Root):
    name: str


class Assign(_Root):
    value: ase.SExpr
    targets: tuple[str, ...]


class Scfg_Pass(_Root):
    pass


class Scfg_While(_Root):
    test: ase.SExpr
    body: ase.SExpr


class Scfg_If(_Root):
    test: ase.SExpr
    then: ase.SExpr
    orelse: ase.SExpr


class Let(_Root):
    name: str
    value: ase.SExpr
    body: ase.SExpr


class Return(_Root):
    iostate: ase.SExpr
    retval: ase.SExpr


class BindArg(_Root):
    idx: int


class Grammar(grammar.Grammar):
    start = lam.LamGrammar.start | _Root


def convert_to_rvsdg(grm: Grammar, prgm: SExpr, varinfo: VariableInfo):
    tp = prgm._tape

    def get_block(
        expr: SExpr,
    ) -> SExpr:
        return next(
            ase.search_parents(expr, lambda x: x._head == "PyAst_block")
        )

    def uses_io(val: SExpr):
        return any(
            ase.search_descendants(
                val, lambda x: x._head == "VarLoad" and x._args == (".io",)
            )
        )

    def unpack_tuple(
        val: SExpr, targets: Sequence[str], *, force_io=False
    ) -> Iterable[tuple[SExpr, str]]:
        if (force_io or uses_io(val)) and ".io" not in targets:
            targets = (".io", *targets)
        if len(targets) == 1:
            yield (val, targets[0])
        else:
            packed_name = f".packed.{val._handle}"
            yield val, packed_name
            for i, target in enumerate(targets):
                packed = grm.write(VarLoad(packed_name))
                yield grm.write(lam.Unpack(idx=i, tup=packed)), target

    def unpack_assign(
        val: SExpr, targets: Sequence[str], *, force_io=False
    ) -> Iterable[tuple[SExpr, str]]:
        assert len(targets) > 0
        unpack_io = force_io or uses_io(val)
        if unpack_io:
            packed_name = f".val.{val._handle}"
            for val, packed_name in unpack_tuple(
                val, [packed_name], force_io=True
            ):
                yield val, packed_name
                get_val = lambda: grm.write(VarLoad(packed_name))
        elif len(targets) > 1:
            packed_name = f".val.{val._handle}"
            yield val, packed_name
            get_val = lambda: grm.write(VarLoad(packed_name))
        else:
            get_val = lambda: val
        # Unpack the value for all targets
        for target in targets:
            yield get_val(), target

    def lookup_defined(blk: ase.BasicSExpr):
        def get_varnames_defined_in_block(
            expr: ase.SExpr, state: ase.TraverseState
        ):
            match expr:
                case Let(
                    str(varname),
                    ase.SExpr() as value,
                    ase.SExpr() as inner,
                ):
                    return (yield inner) | {varname}
                case _:
                    return set()

        memo = ase.traverse(blk, get_varnames_defined_in_block)
        # return memo[blk]
        defs = memo[blk]
        return {k for k in defs if k != ".io" and not k.startswith(".")}

    # empty_set = frozenset()

    # def lookup_used(blk: ase.BasicSExpr):
    #     class FindUsed(TreeRewriter[set[str]]):
    #         flag_save_history = False

    #         def rewrite_var_load(
    #             self, orig: ase.BasicSExpr, varname: str
    #         ) -> set[str]:
    #             return {varname}

    #         def rewrite_generic(
    #             self, orig: ase.BasicSExpr, args: tuple[Any, ...], updated: bool
    #         ) -> set[str] | ase.BasicSExpr:
    #             return empty_set

    #     mypass = FindUsed()
    #     blk.apply_bottomup(mypass)
    #     memo = mypass.memo
    #     return reduce(operator.or_, memo.values())

    def lookup_input_vars(blk: ase.BasicSExpr):
        defined = set()
        external_references = set()

        def lookup_algo(expr: ase.BasicSExpr, state: ase.TraverseState):
            match expr:
                case Let(
                    str(varname),
                    ase.SExpr() as value,
                    ase.SExpr() as body,
                ):
                    yield value
                    defined.add(varname)
                    yield body
                case VarLoad(str(varname)):
                    if varname not in defined:
                        external_references.add(varname)
                case lam.Lam():
                    pass  # do not recurse into lambda
                case ase.SExpr():
                    assert expr._head != "VarLoad"
                    for child in expr._args:
                        yield child

        ase.traverse(blk, lookup_algo)
        return external_references

    @contextmanager
    def handle_chained_expr(*args: SExpr):
        argnames = [f".tmp.callarg.{i}" for i in range(len(args))]
        argloads = [grm.write(VarLoad(k)) for k in argnames]

        def gen(last: SExpr):
            for k, v in zip(argnames, args, strict=True):
                for unpacked, target in reversed(list(unpack_assign(v, [k]))):
                    last = grm.write(
                        Let(name=target, value=unpacked, body=last)
                    )
            return last

        yield argloads, gen

    def replace_scfg_pass(block_root: SExpr, all_names: list[str]):
        class ConvertSCFGPass(TreeRewriter[SExpr]):
            def rewrite_Scfg_Pass(self, orig: ase.BasicSExpr) -> SExpr:
                def put_load(varname):
                    if varname == ".io":
                        return grm.write(VarLoad(".io"))
                    else:
                        return grm.write(VarLoad(varname))

                args = [put_load(k) for k in all_names]
                # print("all_names", all_names)
                # print("defined_names", defined_names)
                # breakpoint()
                if len(args) > 1:
                    return grm.write(lam.Pack(tuple(args)))
                else:
                    return args[0]

        rewriter = ConvertSCFGPass()
        ase.apply_bottomup(block_root, rewriter)
        return rewriter.memo[block_root]

    class Convert2RVSDG(TreeRewriter[SExpr]):
        """Convert to RVSDG with Let-binding"""

        def rewrite_generic(
            self, orig: SExpr, args: tuple[Any, ...], updated: bool
        ) -> SExpr:
            raise NotImplementedError(orig._head)

        def rewrite_PyAst_Name(
            self, orig: ase.BasicSExpr, name: str, ctx: str, loc: SExpr
        ) -> SExpr:
            if name in varinfo.nonlocals:
                return grm.write(Py_GlobalLoad(name))
            else:
                match ctx:
                    case "store":
                        return grm.write(VarStore(name))
                    case "load":
                        return grm.write(VarLoad(name))
                    case _:
                        raise AssertionError(ctx)

        def rewrite_PyAst_loc(self, orig: ase.BasicSExpr, *args) -> SExpr:
            return orig

        def rewrite_PyAst_arg(
            self,
            orig: ase.BasicSExpr,
            name: str,
            annotation: SExpr,
            loc: SExpr,
        ) -> SExpr:
            return name

        def rewrite_PyAst_arguments(
            self, orig: ase.BasicSExpr, *args: SExpr
        ) -> SExpr:
            return grm.write(Py_Args(args))

        def rewrite_PyAst_Constant_bool(
            self, orig: ase.BasicSExpr, val: bool, loc: SExpr
        ) -> SExpr:
            return grm.write(Py_Bool(val))

        def rewrite_PyAst_Constant_int(
            self, orig: ase.BasicSExpr, val: int, loc: SExpr
        ) -> SExpr:
            return grm.write(Py_Int(val))

        def rewrite_PyAst_Constant_complex(
            self, orig: ase.BasicSExpr, real: float, imag: float, loc: SExpr
        ) -> SExpr:
            return grm.write(Py_Complex(real=real, imag=imag))

        def rewrite_PyAst_Constant_str(
            self, orig: ase.BasicSExpr, val: str, loc: SExpr
        ) -> SExpr:
            return grm.write(Py_Str(val))

        def rewrite_PyAst_None(
            self, orig: ase.BasicSExpr, loc: SExpr
        ) -> SExpr:
            return grm.write(Py_None())

        def rewrite_PyAst_Tuple(
            self, orig: ase.BasicSExpr, *rest: SExpr
        ) -> SExpr:
            [*elts, loc] = rest
            return grm.write(Py_Tuple(tuple(elts)))

        def rewrite_PyAst_List(
            self, orig: ase.BasicSExpr, *rest: SExpr
        ) -> SExpr:
            [*elts, loc] = rest
            return grm.write(Py_List(tuple(elts)))

        def rewrite_PyAst_Assign(
            self, orig: ase.BasicSExpr, val: SExpr, *rest: SExpr
        ) -> SExpr:
            [*targets, loc] = rest
            return grm.write(
                Assign(value=val, targets=tuple(t._args[0] for t in targets))
            )

        def rewrite_PyAst_AugAssign(
            self,
            orig: ase.BasicSExpr,
            opname: str,
            left: SExpr,
            right: SExpr,
            loc: SExpr,
        ) -> SExpr:
            [lhs_name] = left._args
            rhs_name = f".rhs.{orig._handle}"
            last = grm.write(
                Py_InplaceBinOp(
                    opname=opname,
                    iostate=self.get_io(),
                    lhs=grm.write(VarLoad(lhs_name)),
                    rhs=grm.write(VarLoad(rhs_name)),
                )
            )
            for val, name in reversed(
                list(unpack_assign(right, targets=[rhs_name]))
            ):
                last = grm.write(Let(name=name, value=val, body=last))
            out = grm.write(Assign(value=last, targets=(lhs_name,)))
            return out

        def rewrite_PyAst_callargs_pos(
            self, orig: ase.BasicSExpr, *args: SExpr
        ) -> SExpr:
            return args

        def rewrite_PyAst_Call(
            self,
            orig: ase.BasicSExpr,
            func: SExpr,
            posargs: tuple[SExpr, ...],
            loc: SExpr,
        ) -> SExpr:
            with handle_chained_expr(func, *posargs) as (args, finish):
                [func, *args] = args
                last = finish(
                    grm.write(
                        Py_Call(
                            iostate=self.get_io(),
                            callee=func,
                            args=tuple(args),
                        )
                    )
                )
                return last

        def rewrite_PyAst_Attribute(
            self,
            orig: ase.BasicSExpr,
            value: SExpr,
            attr: str,
            loc: SExpr,
        ) -> SExpr:
            with handle_chained_expr(value) as ((arg,), finish):
                return finish(
                    grm.write(
                        Py_GetAttr(attr=attr, iostate=self.get_io(), value=arg)
                    )
                )

        def rewrite_PyAst_Subscript(
            self,
            orig: ase.BasicSExpr,
            value: SExpr,
            slice: SExpr,
            loc: SExpr,
        ) -> SExpr:
            with handle_chained_expr(value, slice) as ((value, slice), finish):
                return finish(
                    grm.write(
                        Py_GetItem(
                            iostate=self.get_io(), value=value, slice=slice
                        )
                    )
                )

        def rewrite_PyAst_Compare(
            self,
            orig: ase.BasicSExpr,
            opname: str,
            left: SExpr,
            right: SExpr,
            loc: SExpr,
        ) -> SExpr:
            with handle_chained_expr(left, right) as ((lhs, rhs), finish):
                return finish(
                    grm.write(
                        Py_Compare(
                            opname=opname,
                            iostate=self.get_io(),
                            lhs=lhs,
                            rhs=rhs,
                        )
                    )
                )

        def rewrite_PyAst_block(
            self, orig: ase.BasicSExpr, *body: SExpr
        ) -> SExpr:
            last = grm.write(Scfg_Pass())
            for stmt in reversed(body):
                match stmt:
                    case Assign(value=val, targets=targets):
                        iterable = unpack_assign(val, targets)
                        for unpacked, target in reversed(list(iterable)):
                            last = grm.write(
                                Let(name=target, value=unpacked, body=last)
                            )

                    case Py_If(test=test, then=blk_true, orelse=blk_false):
                        # look up variable in each block to find variable
                        # definitions
                        defs_true = lookup_defined(blk_true) | {".io"}
                        defs_false = lookup_defined(blk_false) | {".io"}
                        defs = sorted(defs_true | defs_false)
                        blk_true = replace_scfg_pass(blk_true, defs)
                        blk_false = replace_scfg_pass(blk_false, defs)

                        if uses_io(test):
                            [
                                (unpack_cond, unpack_name),
                                (io, iovar),
                                (cond, _),
                            ] = unpack_tuple(test, [".cond"], force_io=True)
                            stmt = grm.write(
                                Scfg_If(
                                    test=cond, then=blk_true, orelse=blk_false
                                )
                            )
                            stmt = grm.write(
                                Let(name=iovar, value=io, body=stmt)
                            )
                            stmt = grm.write(
                                Let(
                                    name=unpack_name,
                                    value=unpack_cond,
                                    body=stmt,
                                )
                            )
                        else:
                            raise NotImplementedError

                        iterable = unpack_tuple(stmt, defs, force_io=True)
                        for unpacked, target in reversed(list(iterable)):
                            last = grm.write(
                                Let(name=target, value=unpacked, body=last)
                            )

                    case Py_While(
                        test=VarLoad() as test,
                        body=ase.SExpr() as loopblk,
                    ):
                        # look for defined names
                        defs = lookup_defined(loopblk) | {".io"}
                        # variables that loopback
                        backedge_var_prefix = "__scfg_backedge_var_"
                        loop_cont = "__scfg_loop_cont__"
                        loopback_vars = [loop_cont, *sorted(defs)]
                        # output variables
                        output_vars = [*loopback_vars]
                        loopblk = replace_scfg_pass(loopblk, output_vars)
                        pack_name = f".packed.{loopblk._handle}"
                        # create packing for mutated variables
                        for unpacked, target in reversed(
                            list(
                                unpack_tuple(
                                    grm.write(VarLoad(pack_name)),
                                    loopback_vars,
                                )
                            )
                        ):
                            loopblk = grm.write(
                                Let(name=target, value=unpacked, body=loopblk)
                            )
                        stmt = grm.write(Scfg_While(test=test, body=loopblk))

                        def prep_pack(k):
                            if k.startswith(backedge_var_prefix):
                                return grm.write(Py_Undef())
                            return grm.write(VarLoad(k))

                        packed = grm.write(
                            lam.Pack(elts=tuple(map(prep_pack, output_vars)))
                        )
                        stmt = grm.write(
                            Let(name=pack_name, value=packed, body=stmt)
                        )

                        iterable = unpack_tuple(stmt, output_vars)
                        for unpacked, target in reversed(list(iterable)):
                            last = grm.write(
                                Let(name=target, value=unpacked, body=last)
                            )

                    case Py_Return(retval=retval):
                        output = list(unpack_tuple(retval, [".retval"]))
                        match output:
                            case [
                                (unpack_cond, unpack_name),
                                (io, iovar),
                                (value, valuevar),
                            ]:
                                stmt = grm.write(
                                    Return(
                                        iostate=grm.write(VarLoad(iovar)),
                                        retval=grm.write(VarLoad(valuevar)),
                                    )
                                )
                                stmt = grm.write(
                                    Let(name=valuevar, value=value, body=stmt)
                                )
                                stmt = grm.write(
                                    Let(name=iovar, value=io, body=stmt)
                                )
                                stmt = grm.write(
                                    Let(
                                        name=unpack_name,
                                        value=unpack_cond,
                                        body=stmt,
                                    )
                                )
                                last = stmt
                            case [(value, valuevar)]:
                                stmt = grm.write(
                                    Return(iostate=self.get_io(), retval=value)
                                )
                                last = stmt
                            case _:
                                assert False
                    case Py_Pass():
                        pass
                    case _:
                        raise AssertionError(stmt)
            return last

        def rewrite_PyAst_If(
            self,
            orig: ase.BasicSExpr,
            test: SExpr,
            body: SExpr,
            orelse: SExpr,
            loc: SExpr,
        ) -> SExpr:
            return grm.write(Py_If(test=test, then=body, orelse=orelse))

        def rewrite_PyAst_While(
            self, orig: ase.BasicSExpr, *rest: SExpr
        ) -> SExpr:
            [test, body, loc] = rest
            match test:
                case VarLoad(name=str(testname)):
                    pass
                case _:
                    raise AssertionError(
                        f"unsupported while loop: {ase.pretty_str(test)}"
                    )
            return grm.write(Py_While(test=test, body=body))

        def rewrite_PyAst_UnaryOp(
            self, orig: ase.BasicSExpr, opname: str, val: SExpr, loc: SExpr
        ) -> SExpr:
            with handle_chained_expr(val) as ((val,), finish):
                return finish(
                    grm.write(
                        Py_UnaryOp(
                            opname=opname, iostate=self.get_io(), arg=val
                        )
                    )
                )

        def rewrite_PyAst_BinOp(
            self,
            orig: ase.BasicSExpr,
            opname: str,
            lhs: SExpr,
            rhs: SExpr,
            loc: SExpr,
        ) -> SExpr:
            with handle_chained_expr(lhs, rhs) as ((lhs, rhs), finish):
                return finish(
                    grm.write(
                        Py_BinOp(
                            opname=opname,
                            iostate=self.get_io(),
                            lhs=lhs,
                            rhs=rhs,
                        )
                    )
                )

        def rewrite_PyAst_Return(
            self, orig: ase.BasicSExpr, retval: SExpr, loc: SExpr
        ) -> SExpr:
            return grm.write(Py_Return(retval))

        def rewrite_PyAst_Pass(
            self, orig: ase.BasicSExpr, loc: SExpr
        ) -> SExpr:
            return grm.write(Py_Pass())

        def rewrite_PyAst_FunctionDef(
            self,
            orig: ase.BasicSExpr,
            fname: str,
            args: SExpr,
            body: SExpr,
            loc: SExpr,
        ) -> SExpr:
            return grm.write(Py_Func(name=fname, args=args, body=body))

        def get_io(self) -> SExpr:
            return grm.write(VarLoad(".io"))

    rewriter = Convert2RVSDG()
    with prgm._tape:
        ase.apply_bottomup(prgm, rewriter)
    prgm = rewriter.memo[prgm]

    pp(prgm)

    # Verify
    def verify(root: SExpr):
        # seen = set()
        for parents, cur in ase.walk_descendants_depth_first_no_repeat(root):
            # if cur not in seen:
            #     seen.add(cur)
            # else:
            #     continue
            match cur:
                case Scfg_While(VarLoad(str(testname))):
                    defn: ase.SExpr | None = None
                    for p in reversed(parents):
                        match p:
                            case Let(
                                name=str(x), value=ase.SExpr() as defn
                            ) if x == testname:
                                break

                    match defn:
                        case Py_Int(1):
                            break
                        case Py_Bool(True):
                            break
                        case _:
                            raise AssertionError(
                                f"unsupported loop: {cur}\ndefn {defn}"
                            )
                case ase.BasicSExpr("scfg_while"):
                    raise AssertionError(f"malformed scfg_while: {cur}")

    verify(prgm)

    return prgm


def convert_to_lambda(prgm: SExpr, varinfo: VariableInfo):
    grm = Grammar(prgm._tape)

    match prgm:
        case Py_Func(name=str(fname), args=Py_Args(args)):
            parameters = (".io", *args)
        case _:
            raise AssertionError(ase.as_tuple(prgm, 1))

    # Map var_load to parent
    parent_let_map: dict[SExpr, SExpr] = {}
    var_depth_map: dict[SExpr, int] = {}

    # Compute parent let of (var_load x)
    ts = time.time()

    @dataclass(frozen=True)
    class ParentLetInfo:
        var_load_map: dict[SExpr, int]

        def __or__(self, other) -> ParentLetInfo:
            if isinstance(other, self.__class__):
                new = self.var_load_map.copy()
                for k, v in other.var_load_map.items():
                    assert k not in new
                    new[k] = v
                return ParentLetInfo(new)
            else:
                return NotImplemented

        def drop(self, *varloads: SExpr) -> ParentLetInfo:
            new = self.var_load_map.copy()
            for vl in varloads:
                assert vl._head == "VarLoad"
                new.pop(vl)
            return ParentLetInfo(new)

        def up(self) -> ParentLetInfo:
            return ParentLetInfo(
                {k: v + 1 for k, v in self.var_load_map.items()}
            )

    class FindParentLet(grammar.TreeRewriter[ParentLetInfo]):
        """Bottom up pass"""

        def rewrite_VarLoad(self, orig: ase.SExpr, name: str) -> ParentLetInfo:
            # Propagate (var_load ) up the tree
            return ParentLetInfo({orig: 0})

        def rewrite_Let(
            self,
            orig: ase.SExpr,
            name: str,
            value: ParentLetInfo,
            body: ParentLetInfo,
        ) -> ParentLetInfo:
            # Find matching let
            matched = set()
            for varload, depth in body.var_load_map.items():
                if name == varload._args[0]:
                    parent_let_map[varload] = orig
                    var_depth_map[varload] = depth
                    matched.add(varload)

            # Propagate (var_load) from both value and body
            return value | body.drop(*matched).up()

        def rewrite_Py_Func(
            self,
            orig: ase.SExpr,
            name: str,
            args: ParentLetInfo,
            body: ParentLetInfo,
        ) -> ParentLetInfo:
            for varload, depth in body.var_load_map.items():
                var_depth_map[varload] = depth
            return body

        def rewrite_generic(
            self, orig: ase.SExpr, args: tuple[Any, ...], updated: bool
        ) -> ParentLetInfo:
            # Propagate the union of all sets in the args
            return reduce(
                operator.or_,
                filter(lambda x: isinstance(x, ParentLetInfo), args),
                ParentLetInfo({}),
            )

    reachable = ase.reachable_set(prgm)
    ase.apply_bottomup(prgm, FindParentLet(), reachable=reachable)

    _logger.debug("   var_load - let analysis", time.time() - ts)

    class RewriteToLambda(grammar.TreeRewriter[SExpr]):
        def rewrite_VarLoad(self, orig: SExpr, name: str) -> SExpr:
            blet = parent_let_map.get(orig)
            depth = var_depth_map.get(orig, 0)
            if blet is None:
                if name not in parameters:
                    return grm.write(Py_Undef())
                else:
                    return grm.write(
                        lam.Arg(
                            len(parameters)
                            - parameters.index(name)
                            - 1
                            + depth
                        )
                    )
            return grm.write(lam.Arg(depth))

        def rewrite_Let(
            self, orig: SExpr, name: str, value: SExpr, body: SExpr
        ) -> SExpr:
            return grm.write(lam.App(lam=grm.write(lam.Lam(body)), arg=value))

        def rewrite_Py_Func(
            self, orig: SExpr, name: str, args: SExpr, body: SExpr
        ) -> SExpr:
            # one extra for .io
            for _ in range(len(args._args) + 1):
                body = grm.write(lam.Lam(body))
            return body

    with prgm._tape:
        ts = time.time()
        rtl = RewriteToLambda()
        ase.apply_bottomup(prgm, rtl, reachable=reachable)
        memo = rtl.memo
        _logger.debug("   rewrite_let_into_lambda", time.time() - ts)
        prgm = memo[prgm]

    return prgm  # lb.run_abstraction_pass(prgm)


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

    prgm = convert_to_sexpr(transformed_ast, firstline)

    _logger.debug("convert_to_sexpr", time.time() - t_start)

    varinfo = find_variable_info(prgm)
    _logger.debug(varinfo)

    _logger.debug("find_variable_info", time.time() - t_start)

    grm = Grammar(prgm._tape)
    rvsdg = convert_to_rvsdg(grm, prgm, varinfo)

    _logger.debug("convert_to_rvsdg", time.time() - t_start)

    from sealir.prettyformat import html_format

    pp(rvsdg)

    lam_node = convert_to_lambda(rvsdg, varinfo)
    _logger.debug("convert_to_lambda", time.time() - t_start)
    if _DEBUG:
        pp(lam_node)
        print(ase.pretty_str(lam_node))

    if _DEBUG_HTML:
        # FIXME: This is currently slow due to inefficient metadata lookup.
        print("writing html...")
        ts = time.time()
        with open("debug.html", "w") as fout:
            html_format.write_html(
                fout,
                html_format.prepare_source(source_text),
                html_format.to_html(rvsdg),
                html_format.to_html(lam_node),
            )
        print("   took", time.time() - ts)
    return lam_node


######################


@dataclass(frozen=True)
class EvalLamState(ase.TraverseState):
    context: EvalCtx


class EvalIO:
    def __repr__(self) -> str:
        return "IO"


class EvalUndef:
    def __repr__(self) -> str:
        return "UNDEFINED"


class SExprValuePair(NamedTuple):
    expr: SExpr
    value: Any


@dataclass(frozen=True)
class EvalCtx:
    args: tuple[Any, ...]
    localscope: dict[str, Any]
    value_map: dict[SExpr, Any] = field(default_factory=dict)
    blam_stack: list[SExprValuePair] = field(default_factory=list)

    @classmethod
    def from_arguments(cls, *args):
        return cls.from_arguments_and_locals(args, {})

    @classmethod
    def from_arguments_and_locals(cls, args, locals):
        return EvalCtx(
            args=tuple(reversed([EvalIO(), *args])), localscope=locals
        )

    def make_arg_node(self, grm: Grammar):
        ba = []
        for i, v in enumerate(self.args):
            se = grm.write(BindArg(i))
            self.value_map[se] = v
            ba.append(se)
        return ba

    @contextmanager
    def bind_app(self, lam_expr: SExpr, argval: Any):
        self.blam_stack.append(SExprValuePair(lam_expr, argval))
        try:
            yield
        finally:
            self.blam_stack.pop()


def ensure_io(obj: Any) -> EvalIO:
    assert isinstance(obj, EvalIO)
    return obj


def lambda_evaluation(expr: ase.BasicSExpr, state: EvalLamState):
    DEBUG = _DEBUG

    indent = len(state.parents)
    ctx = state.context

    if DEBUG:
        dbg_print = print

        dbg_only = lambda f: lambda *args, **kwargs: f(*args, **kwargs)
    else:

        def dbg_print(*args):
            pass

        dbg_only = lambda f: lambda *args, **kwargs: None

    try:
        dbg_print(" " * indent, "EVAL", expr._handle, repr(expr))
        match expr:
            case lam.Lam(body):
                retval = yield body
                return retval
            case lam.App(arg=argval, lam=lam_func):
                with ctx.bind_app(lam_func, (yield argval)):
                    retval = yield lam_func
                    return retval
            case lam.Arg(int(argidx)):
                for i, v in enumerate(reversed(ctx.blam_stack)):
                    dbg_print(" " * indent, i, "--", v.value)
                retval = ctx.blam_stack[-argidx - 1].value
                return retval
            case lam.Unpack(idx=int(idx), tup=packed_expr):
                packed = yield packed_expr
                retval = packed[idx]
                return retval
            case lam.Pack(args):
                elems = []
                for arg in args:
                    elems.append((yield arg))
                retval = tuple(elems)
                return retval
            case BindArg():
                return ctx.value_map[expr]
            case Scfg_If(
                test=cond,
                then=br_true,
                orelse=br_false,
            ):
                condval = yield cond
                if condval:
                    retval = yield br_true
                else:
                    retval = yield br_false
                return retval
            case Scfg_While(body=loopblk):
                loop_cond = True
                while loop_cond:
                    memo = ase.traverse(loopblk, lambda_evaluation, state)
                    loop_end_vars = memo[loopblk]
                    # Update the loop condition (MUST BE FIRST ITEM)
                    loop_cond = loop_end_vars[0]
                    # Replace the top of stack with the out going value of the loop body
                    ctx.blam_stack[-1] = ctx.blam_stack[-1]._replace(
                        value=loop_end_vars
                    )
                return loop_end_vars
            case Return(iostate=iostate, retval=retval):
                ioval = ensure_io((yield iostate))
                retval = yield retval
                return ioval, retval
            case Py_Pass():
                return
            case Py_Tuple(args):
                elems = []
                for arg in args:
                    elems.append((yield arg))
                return tuple(elems)
            case Py_List(args):
                elems = []
                for arg in args:
                    elems.append((yield arg))
                return list(elems)
            case Py_Undef():
                return EvalUndef()
            case Py_None():
                return None
            case Py_Int(int(ival)):
                return ival
            case Py_Bool(bool(ival)):
                return ival
            case Py_Complex(real=float(freal), imag=float(fimag)):
                return complex(freal, fimag)
            case Py_Str(str(text)):
                return text
            case Py_GetAttr(
                attr=str(attrname),
                iostate=iostate,
                value=value,
            ):
                ioval = ensure_io((yield iostate))
                retval = getattr((yield value), attrname)
                return ioval, retval
            case Py_GetItem(
                iostate=iostate,
                value=value,
                slice=index,
            ):
                ioval = ensure_io((yield iostate))
                base_val = yield value
                index_val = yield index
                retval = base_val[index_val]
                return ioval, retval
            case Py_UnaryOp(
                opname=str(opname),
                iostate=iostate,
                arg=val,
            ):
                ioval = yield iostate
                match opname:
                    case "not":
                        retval = not (yield val)
                    case _:
                        raise NotImplementedError(opname)
                return ioval, retval
            case Py_BinOp(
                opname=str(op),
                iostate=iostate,
                lhs=lhs,
                rhs=rhs,
            ):
                ioval = ensure_io((yield iostate))
                lhsval = yield lhs
                rhsval = yield rhs
                match op:
                    case "+":
                        retval = lhsval + rhsval
                    case "-":
                        retval = lhsval - rhsval
                    case "*":
                        retval = lhsval * rhsval
                    case "/":
                        retval = lhsval / rhsval
                    case "//":
                        retval = lhsval // rhsval
                    case _:
                        raise NotImplementedError(op)
                return ioval, retval
            case Py_InplaceBinOp(
                opname=str(op),
                iostate=iostate,
                lhs=lhs,
                rhs=rhs,
            ):
                ioval = ensure_io((yield iostate))
                lhsval = yield lhs
                rhsval = yield rhs
                match op:
                    case "+":
                        lhsval += rhsval
                    case "*":
                        lhsval *= rhsval
                    case _:
                        raise NotImplementedError(op)
                retval = ioval, lhsval
                return retval
            case Py_Compare(
                opname=str(op),
                iostate=iostate,
                lhs=lhs,
                rhs=rhs,
            ):
                ioval = ensure_io((yield iostate))
                lhsval = yield lhs
                rhsval = yield rhs
                match op:
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
                return ioval, res
            case Py_Call(
                iostate=iostate,
                callee=callee,
                args=args,
            ):
                ioval = ensure_io((yield iostate))
                callee = yield callee
                argvals = []
                for arg in args:
                    argvals.append((yield arg))
                retval = callee(*argvals)
                return ioval, retval
            case Py_GlobalLoad(str(glbname)):
                return ChainMap(ctx.localscope, __builtins__)[glbname]
            case _:
                raise AssertionError(ase.as_tuple(expr))
    finally:
        if "retval" in locals():
            dbg_print(" " * indent, f"({expr._handle})=>", retval)
