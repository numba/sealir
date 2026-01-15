from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import TypedDict


class NodeDict(TypedDict):
    children: list[str]
    cost: float
    eclass: str
    op: str
    subsumed: bool


class ClassDataDict(TypedDict):
    type: str


class EGraphJsonDict(TypedDict):
    nodes: dict[str, NodeDict]
    root_eclasses: list[str]
    class_data: dict[str, ClassDataDict]


@dataclass
class Qualname:
    """
    Qualified name from Serialized EGraph JSON data.
    """

    prefix: str
    name: str
    param: Qualname | None = None

    def getattr(self, name: str) -> Qualname:
        assert self.param is None
        new_prefix = (
            self.prefix + "." + self.name if self.prefix else self.name
        )
        return Qualname(prefix=new_prefix, name=name)

    def subscript(self, tp_param: Qualname) -> Qualname:
        assert self.param is None
        return replace(self, param=tp_param)

    def get_fullname(self) -> str:
        new_prefix = (
            self.prefix + "." + self.name if self.prefix else self.name
        )
        if self.param is not None:
            return f"{new_prefix}[{self.param.get_fullname()}]"
        else:
            return new_prefix


def parse_type(typename: str) -> Qualname:
    typename = _normalize_no_prefix(typename)
    astree = ast.parse(typename, mode="eval")
    res = _ParseEGraphType().visit(astree)
    return res


def _normalize_no_prefix(name: str):
    if "." not in name:
        # assume they come from egglog.builtins
        return "egglog.builtins." + name
    return name


class _ParseEGraphType(ast.NodeTransformer):
    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Attribute(self, node: ast.Attribute):
        assert isinstance(node.ctx, ast.Load)
        value = self.visit(node.value)
        return value.getattr(node.attr)

    def visit_Name(self, node: ast.Name):
        assert isinstance(node.ctx, ast.Load)
        return Qualname(prefix="", name=node.id)

    def visit_Subscript(self, node: ast.Subscript):
        assert isinstance(node.ctx, ast.Load)
        value = self.visit(node.value)
        slice_ = self.visit(node.slice)
        return value.subscript(slice_)

    def generic_visit(self, node):
        raise NotImplementedError(type(node), ast.dump(node))
