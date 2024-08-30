from __future__ import annotations
import sys
import ast
from typing import Type, TypeAlias
from pprint import pprint

from numba_rvsdg.core.datastructures.ast_transforms import (
    unparse_code,
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
)

from sealir import ase
from sealir.lam import LamBuilder


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
            "Py_FunctionDef",
            fname,
            args,
            ase.expr("Py_block", *body),
            self.get_loc(node),
        )

    def visit_Return(self, node: ast.Return) -> SExpr:
        return ase.expr(
            "Py_Return",
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
            "Py_arguments",
            *map(self.visit, node.args),
        )

    def visit_arg(self, node: ast.arg) -> SExpr:
        return ase.expr(
            "Py_arg",
            node.arg,
            self.visit(node.annotation),
            self.get_loc(node),
        )

    def visit_Name(self, node: ast.Name) -> SExpr:
        print(ast.dump(node))
        match node.ctx:
            case ast.Load():
                return ase.expr("Py_Name_ld", node.id)
            case ast.Store():
                return ase.expr("Py_Name_st", node.id)
            case _:
                raise NotImplementedError(node.ctx)

    def visit_Assign(self, node: ast.Assign) -> SExpr:
        return ase.expr(
            "Py_Assign",
            self.visit(node.value),
            *map(self.visit, node.targets),
            self.get_loc(node),
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> SExpr:
        return ase.expr(
            "Py_AugAssign",
            self.map_op(node.op),
            self.visit(node.value),
            self.visit(node.target),
            self.get_loc(node),
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> SExpr:
        return ase.expr(
            "Py_UnaryOp",
            self.map_op(node.op),
            self.visit(node.operand),
            self.get_loc(node),
        )

    def visit_BinOp(self, node: ast.BinOp) -> SExpr:
        return ase.expr(
            "Py_BinOp",
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
            "Py_Compare",
            opname,
            self.visit(node.left),
            self.visit(comp),
            self.get_loc(node),
        )

    def visit_Call(self, node: ast.Call) -> SExpr:
        # TODO
        assert not node.keywords
        posargs = ase.expr("Py_callargs_pos", *map(self.visit, node.args))
        return ase.expr(
            "Py_Call",
            self.visit(node.func),
            posargs,
            self.get_loc(node),
        )

    def visit_While(self, node: ast.While) -> SExpr:
        # ("test", "body", "orelse")
        test = self.visit(node.test)
        body = ase.expr(
            "Py_block",
            *map(self.visit, node.body),
            self.get_loc(node),
        )
        assert not node.orelse
        return ase.expr(
            "Py_While",
            test,
            body,
            self.get_loc(node),
        )

    def visit_If(self, node: ast.If) -> SExpr:
        test = self.visit(node.test)
        body = ase.expr(
            "Py_block",
            *map(self.visit, node.body),
            self.get_loc(node),
        )
        orelse = ase.expr(
            "Py_block",
            *map(self.visit, node.orelse),
            self.get_loc(node),
        )
        return ase.expr(
            "Py_If",
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
                        "Py_Constant_int",
                        node.value,
                        self.get_loc(node),
                    )
                case None:
                    return ase.expr(
                        "Py_None",
                        self.get_loc(node),
                    )
                case str():
                    return ase.expr(
                        "Py_Constant_str",
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
            "Py_loc",
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


def find_variable_info(expr: SExpr):
    loaded = {}
    stored = {}

    class FindVariableUse(ase.TreeVisitor):
        def visit(self, expr: SExpr):
            match expr.as_tuple():
                case ("Py_Name_ld", name):
                    loaded[name] = expr
                case ("Py_Name_st", name):
                    stored[name] = expr
                case ("Py_arg", name, annotation):
                    stored[name] = expr

    expr.apply_bottomup(FindVariableUse())

    pprint(loaded)
    pprint(stored)
    nonlocals = set()
    for k in loaded:
        if k not in stored:
            nonlocals.add(k)
    pprint(nonlocals)


def restructure_source(function):
    ast2scfg_transformer = AST2SCFGTransformer(function)
    astcfg = ast2scfg_transformer.transform_to_ASTCFG()
    scfg = astcfg.to_SCFG()
    scfg.restructure()
    scfg2ast = SCFG2ASTTransformer()
    original_ast = unparse_code(function)[0]
    transformed_ast = scfg2ast.transform(original=original_ast, scfg=scfg)

    transformed_ast = ast.fix_missing_locations(transformed_ast)

    print(ast.unparse(transformed_ast))

    prgm = convert_to_sexpr(transformed_ast)

    find_variable_info(prgm)


######################


# def sum1d(n: int) -> int:
#     c = 0
#     for i in range(n):
#         c += i
#     return c

# def sum1d(n: int) -> int:
#     c = 0
#     for i in range(n):
#         for j in range(i):
#             c += i * j
#             if c > 100:
#                 break
#     return c


def sum1d(n: int) -> int:
    c = 0
    for i in range(n):
        for j in range(i):
            c += i + j
    return c


def main():
    source = restructure_source(sum1d)
    print(source)


main()
