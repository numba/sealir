from __future__ import annotations

import ast
import warnings
from pprint import pprint
from typing import TypeAlias, cast

from sealir import ase
from sealir.rvsdg import _DEBUG, internal_prefix

SExpr: TypeAlias = ase.SExpr


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
        if node.decorator_list:
            warnings.warn("decorators are not handled")
            # decorator_list = [self.visit(x) for x in node.decorator_list]
        return self._tape.expr(
            "PyAst_FunctionDef",
            fname,
            args,
            self._tape.expr("PyAst_block", *body),
            self.get_loc(node),
        )

    def visit_Pass(self, node: ast.Pass) -> SExpr:
        return self._tape.expr("PyAst_Pass", self.get_loc(node))

    def visit_Return(self, node: ast.Return) -> SExpr:
        return self._tape.expr(
            "PyAst_Return",
            self.visit(cast(ast.AST, node.value)),
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

    def visit_keyword(self, node: ast.keyword) -> SExpr:
        return self._tape.expr(
            "PyAst_keyword",
            node.arg,
            self.visit(node.value),
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
                internal_prefix("_"),
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
            self.map_op(node.op),   # type: ignore[arg-type]
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

    def visit_Slice(self, node: ast.Slice) -> SExpr:
        none = self._tape.expr("PyAst_None", self.get_loc(node))
        return self._tape.expr(
            "PyAst_Slice",
            self.visit(node.lower) if node.lower is not None else none,
            self.visit(node.upper) if node.upper is not None else none,
            self.visit(node.step) if node.step is not None else none,
            self.get_loc(node),
        )

    def visit_Call(self, node: ast.Call) -> SExpr:
        posargs = self._tape.expr(
            "PyAst_callargs_pos", *map(self.visit, node.args)
        )
        if node.keywords:
            kwargs = self._tape.expr(
                "PyAst_callargs_kw", *map(self.visit, node.keywords)
            )
            return self._tape.expr(
                "PyAst_CallKwargs",
                self.visit(node.func),
                posargs,
                kwargs,
                self.get_loc(node),
            )
        else:
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
                case float():
                    return self._tape.expr(
                        "PyAst_Constant_float",
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
            # binary
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
            case ast.Pow():
                return "**"
            case ast.MatMult():
                return "@"
            # unary
            case ast.Not():
                return "not"
            case ast.USub():
                return "-"
            case _:
                raise NotImplementedError(node)

    def get_loc(self, node: ast.AST) -> SExpr:
        return self._tape.expr(
            "PyAst_loc",
            node.lineno,            # type: ignore[attr-defined]
            node.col_offset,        # type: ignore[attr-defined]
            node.end_lineno,        # type: ignore[attr-defined]
            node.end_col_offset,    # type: ignore[attr-defined]
        )


def convert_to_sexpr(node: ast.AST, first_line: int):
    with ase.Tape() as stree:
        out = ConvertToSExpr(stree, first_line).visit(node)
    if _DEBUG:
        pprint(ase.as_tuple(out, depth=-1))
    return out
