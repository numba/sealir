from __future__ import annotations
import sys
import ast
import inspect
from typing import Type, TypeAlias, Any
from pprint import pprint
from dataclasses import dataclass, field
from collections import defaultdict, deque

from numba_rvsdg.core.datastructures.ast_transforms import (
    unparse_code,
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
)

from sealir import ase
from sealir.lam import LamBuilder
from sealir.rewriter import TreeRewriter


SExpr: TypeAlias = ase.Expr


class ConvertToSExpr(ast.NodeTransformer):
    _tape = ase.Tape

    def __init__(self, tape):
        super().__init__()
        self._tape = tape

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
        return ase.expr(
            "PyAst_FunctionDef",
            fname,
            args,
            ase.expr("PyAst_block", *body),
            self.get_loc(node),
        )

    def visit_Return(self, node: ast.Return) -> SExpr:
        return ase.expr(
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
        return ase.expr(
            "PyAst_arguments",
            *map(self.visit, node.args),
        )

    def visit_arg(self, node: ast.arg) -> SExpr:
        return ase.expr(
            "PyAst_arg",
            node.arg,
            self.visit(node.annotation),
            self.get_loc(node),
        )

    def visit_Name(self, node: ast.Name) -> SExpr:
        print(ast.dump(node))
        match node.ctx:
            case ast.Load():
                ctx = 'load'
            case ast.Store():
                ctx = 'store'
            case _:
                raise NotImplementedError(node.ctx)
        return ase.expr("PyAst_Name", node.id, ctx)

    def visit_Assign(self, node: ast.Assign) -> SExpr:
        return ase.expr(
            "PyAst_Assign",
            self.visit(node.value),
            *map(self.visit, node.targets),
            self.get_loc(node),
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> SExpr:
        return ase.expr(
            "PyAst_AugAssign",
            self.map_op(node.op),
            self.visit(node.value),
            self.visit(node.target),
            self.get_loc(node),
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> SExpr:
        return ase.expr(
            "PyAst_UnaryOp",
            self.map_op(node.op),
            self.visit(node.operand),
            self.get_loc(node),
        )

    def visit_BinOp(self, node: ast.BinOp) -> SExpr:
        return ase.expr(
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
            case _:
                raise NotImplementedError(op)
        return ase.expr(
            "PyAst_Compare",
            opname,
            self.visit(node.left),
            self.visit(comp),
            self.get_loc(node),
        )

    def visit_Call(self, node: ast.Call) -> SExpr:
        # TODO
        assert not node.keywords
        posargs = ase.expr("PyAst_callargs_pos", *map(self.visit, node.args))
        return ase.expr(
            "PyAst_Call",
            self.visit(node.func),
            posargs,
            self.get_loc(node),
        )

    def visit_While(self, node: ast.While) -> SExpr:
        # ("test", "body", "orelse")
        test = self.visit(node.test)
        body = ase.expr(
            "PyAst_block",
            *map(self.visit, node.body),
            self.get_loc(node),
        )
        assert not node.orelse
        return ase.expr(
            "PyAst_While",
            test,
            body,
            self.get_loc(node),
        )

    def visit_If(self, node: ast.If) -> SExpr:
        test = self.visit(node.test)
        body = ase.expr(
            "PyAst_block",
            *map(self.visit, node.body),
            self.get_loc(node),
        )
        orelse = ase.expr(
            "PyAst_block",
            *map(self.visit, node.orelse),
            self.get_loc(node),
        )
        return ase.expr(
            "PyAst_If",
            test,
            body,
            orelse,
            self.get_loc(node),
        )

    def visit_Constant(self, node: ast.Constant) -> SExpr:
        if node.kind is None:
            match node.value:
                case int():
                    return ase.expr(
                        "PyAst_Constant_int",
                        node.value,
                        self.get_loc(node),
                    )
                case None:
                    return ase.expr(
                        "PyAst_None",
                        self.get_loc(node),
                    )
                case str():
                    return ase.expr(
                        "PyAst_Constant_str",
                        node.value,
                        self.get_loc(node),
                    )
        raise NotImplementedError(ast.dump(node))

    def map_op(self, node: ast.operator) -> str:
        match node:
            case ast.Add():
                return "+"
            case ast.Not():
                return "not"
            case _:
                raise NotImplementedError(node)

    def get_loc(self, node: ast.AST) -> SExpr:
        return ase.expr(
            "PyAst_loc",
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
            None
        )


def convert_to_sexpr(node: ast.AST):
    with ase.Tape() as stree:
        out = ConvertToSExpr(stree).visit(node)
    pprint(out.as_tuple(depth=2**30))
    return out

@dataclass(frozen=True)
class VariableInfo:
    nonlocals: set[str]
    loaded: set[str]
    stored: set[str]
    expr_effect: dict
    block_effect: dict

def find_variable_info(expr: SExpr):
    loaded: dict[str, SExpr] = {}
    stored: dict[str, SExpr] = {}

    class FindVariableUse(ase.TreeVisitor):
        def visit(self, expr: SExpr):
            match expr:
                case ase.Expr("PyAst_Name", (name, "load")):
                    loaded[name] = expr
                case ase.Expr("PyAst_Name", (name, "store")):
                    stored[name] = expr
                case ase.Expr("PyAst_arg", (name,)):
                    stored[name] = expr

    expr.apply_bottomup(FindVariableUse())


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

    expr_effect: dict[SExpr, VarEffect] = defaultdict(VarEffect)

    # block_effect tracks the variables used by each statement in a block.
    # It shows which variable-uses are required for the remaining statements
    # in the block to execute. Variable-defs are local to that statement.
    block_effect: dict[SExpr, deque[VarEffect]] = defaultdict(deque)

    class FindDataDependence(ase.TreeVisitor):
        def visit(self, expr: SExpr):
            match expr:
                case ase.Expr("PyAst_block", body):
                    blkeff = block_effect[expr]
                    iter_stmt = iter(reversed(body))
                    last = next(iter_stmt)

                    blkeff.appendleft(VarEffect().merge(expr_effect[last]))
                    for stmt in iter_stmt:
                        lasteff = blkeff[0]
                        # Start with the expression level effect.
                        eff = VarEffect().merge(expr_effect[stmt])
                        blkeff.appendleft(eff)
                        # Keep track of future uses discounting newly defined
                        eff.uses.update(lasteff.uses - expr_effect[stmt].defs)
                        last = stmt
                    expr_effect[expr] = blkeff[0]

                case ase.Expr("PyAst_Name", (name, "load")):
                    if name not in nonlocals:
                        expr_effect[expr].use(name)
                case ase.Expr("PyAst_Name", (name, "store")):
                    expr_effect[expr].define(name)
                case ase.Expr("PyAst_arg", (name,)):
                    expr_effect[expr].define(name)
                case _:
                    for child in expr.args:
                        if isinstance(child, ase.Expr):
                            expr_effect[expr].merge(expr_effect[child])

    expr.apply_bottomup(FindDataDependence())

    return VariableInfo(
        nonlocals=set(nonlocals),
        loaded=set(loaded),
        stored=set(stored),
        expr_effect=expr_effect,
        block_effect=block_effect,
    )

def convert_to_rvsdg(prgm: SExpr, varinfo: VariableInfo):

    lb = LamBuilder(prgm.tape)

    def get_block(expr: SExpr) -> SExpr:
        return next(expr.search_parents(lambda x: x.head == "PyAst_block"))

    # XXX: THIS IS NOT THE ACTUAL ALGORITHM. JUST A MOCK
    class Convert2RVSDG(TreeRewriter[SExpr]):
        def rewrite_generic(self, orig: SExpr, args: tuple[Any, ...], updated: bool) -> SExpr:
            raise NotImplementedError(orig.head)

        def rewrite_PyAst_Name(self, orig: ase.Expr, name: str, ctx: str) -> SExpr:
            if name in varinfo.nonlocals:
                return lb.expr("py_load_global", name)
            else:
                match ctx:
                    case "store":
                        return name
                    case "load":
                        return lb.expr("py_load_var", name)
                    case _:
                        raise AssertionError(ctx)

        def rewrite_PyAst_loc(self, orig: ase.Expr, *args) -> SExpr:
            return orig

        def rewrite_PyAst_arg(self, orig: ase.Expr, name: str, annotation: SExpr, loc: SExpr) -> SExpr:
            return name

        def rewrite_PyAst_arguments(self, orig: ase.Expr, *args: SExpr) -> SExpr:
            return lb.expr("py_args", *args)

        def rewrite_PyAst_Constant_int(self, orig: ase.Expr, val: int, loc: SExpr) -> SExpr:
            return lb.expr("py_int", val)

        def rewrite_PyAst_Constant_str(self, orig: ase.Expr, val: int, loc: SExpr) -> SExpr:
            return lb.expr("py_str", val)

        def rewrite_PyAst_None(self, orig: ase.Expr, loc: SExpr) -> SExpr:
            return lb.expr("py_none")

        def rewrite_PyAst_Assign(self, orig: ase.Expr, val: SExpr, *rest: SExpr) -> SExpr:
            [*targets, loc] = rest
            return lb.expr("py_assign", val, *targets)

        def rewrite_PyAst_AugAssign(self, orig: ase.Expr, opname: str, left: SExpr, right: SExpr, loc: SExpr) -> SExpr:
            return lb.expr("py_assign", lb.expr("py_binop", opname, left, right), left)

        def rewrite_PyAst_callargs_pos(self, orig: ase.Expr, *args: SExpr) -> SExpr:
            return args

        def rewrite_PyAst_Call(self, orig: ase.Expr, func: SExpr, posargs: SExpr, loc: SExpr) -> SExpr:
            return lb.expr("py_call", func, *posargs)

        def rewrite_PyAst_Compare(self, orig: ase.Expr, opname: str, left: SExpr, right: SExpr, loc: SExpr) -> SExpr:
            return lb.expr("py_compare", opname, left, right)

        def rewrite_PyAst_block(self, orig: ase.Expr, *rest: SExpr) -> SExpr:
            [*body, loc] = rest
            return lb.expr("py_block", *body)

        def rewrite_PyAst_If(self, orig: ase.Expr, test: SExpr, body: SExpr, orelse: SExpr, loc: SExpr) -> SExpr:
            return lb.expr("py_if", test, body, orelse)

        def rewrite_PyAst_While(self, orig: ase.Expr, *rest: SExpr) -> SExpr:
            [*body, loc] = rest
            return lb.expr("py_while", *body)

        def rewrite_PyAst_UnaryOp(self, orig: ase.Expr, opname: str, val: SExpr, loc: SExpr) -> SExpr:
            return lb.expr("py_unaryop", opname, val)

        def rewrite_PyAst_Return(self, orig: ase.Expr, val: SExpr, loc: SExpr) -> SExpr:
            return lb.expr("py_return", val)

        def rewrite_PyAst_FunctionDef(self, orig: ase.Expr, fname: str, args: SExpr, body: SExpr, loc: SExpr) -> SExpr:
            return lb.expr("py_func", fname, args, body)



    rewriter = Convert2RVSDG()
    with prgm.tape:
        prgm.apply_bottomup(rewriter)

    rewritten = rewriter.memo[prgm]
    return rewritten




def restructure_source(function):
    ast2scfg_transformer = AST2SCFGTransformer(function)
    astcfg = ast2scfg_transformer.transform_to_ASTCFG()
    scfg = astcfg.to_SCFG()
    scfg.restructure()
    scfg2ast = SCFG2ASTTransformer()
    original_ast = unparse_code(function)[0]
    transformed_ast = scfg2ast.transform(original=original_ast, scfg=scfg)

    transformed_ast = ast.fix_missing_locations(transformed_ast)

    breakpoint()
    print(ast.unparse(transformed_ast))

    prgm = convert_to_sexpr(transformed_ast)

    varinfo = find_variable_info(prgm)

    rvsdg = convert_to_rvsdg(prgm, varinfo)

    print(rvsdg.str())
    # DEMO FIND ORIGINAL AST
    print('---py_range----')
    def find_py_range(rvsdg):
        for path, node in rvsdg.walk_descendants():
            if node.args and node.args[0] == "py_call":
                match node.args[1]:
                    case ase.Expr("expr", ("py_load_global", "range")):
                        return node

    py_range = find_py_range(rvsdg)
    print(py_range.str())

    print('---find md----')
    md = next(py_range.search_parents(lambda x: x.head == ".md.rewrite"))
    (_, _, orig) = md.args
    loc = orig.args[-1]

    lines, offset = inspect.getsourcelines(function)
    line_offset = loc.args[0]

    print(offset + line_offset, '|', lines[line_offset])
    rvsdg.tape.render_dot(show_metadata=True).view()

######################


def sum1d(n: int) -> int:
    c = 0
    for i in range(n):
        c += i
    return c

# def sum1d(n: int) -> int:
#     c = 0
#     for i in range(n):
#         for j in range(i):
#             c += i * j
#             if c > 100:
#                 break
#     return c


# def sum1d(n: int) -> int:
#     c = 0
#     for i in range(n):
#         for j in range(i):
#             c += i + j
#     return c


def main():
    source = restructure_source(sum1d)
    print(source)


main()
