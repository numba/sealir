from __future__ import annotations

from typing import TypeAlias

from sealir import ase, grammar
from sealir.grammar import SExprProto

SExpr: TypeAlias = ase.SExpr


class _Root(grammar.Rule):
    pass


class Rootset(_Root):
    roots: tuple[SExpr, ...]


class Generic(_Root):
    name: str
    children: tuple[SExpr, ...]


class GenericList(_Root):
    name: str
    children: tuple[SExpr, ...]


class Loc(_Root):
    filename: str
    line_first: int
    line_last: int
    col_first: int
    col_last: int


def unknown_loc() -> Loc:
    return Loc(filename="", line_first=0, line_last=0, col_first=0, col_last=0)


class Args(_Root):
    """Used for function signature"""

    arguments: tuple[SExpr, ...]


class Posargs(_Root):
    """Used for positional arguments"""

    args: tuple[SExpr, ...]


class Keyword(_Root):
    name: str
    value: SExpr


class Kwargs(_Root):
    """Used for keyword arguments"""

    kwargs: tuple[SExprProto[Keyword], ...]


class Port(_Root):
    name: str
    value: SExpr


class Attrs(_Root):
    attrs: tuple[SExpr, ...]


class RegionBegin(_Root):
    attrs: SExpr
    inports: tuple[str, ...]


class RegionEnd(_Root):
    begin: RegionBegin
    ports: tuple[SExprProto[Port], ...]


class Func(_Root):
    fname: str
    args: SExprProto[Args]
    body: SExprProto[RegionEnd]


class IfElse(_Root):
    cond: SExpr | None  # Can be None at runtime
    body: SExprProto[RegionEnd]
    orelse: SExprProto[RegionEnd]
    operands: tuple[SExpr, ...]


class Loop(_Root):
    body: SExprProto[RegionEnd]
    operands: tuple[SExpr, ...]


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


class PyFloat(_Root):
    value: float


class PyComplex(_Root):
    real: float
    imag: float


class PyBool(_Root):
    value: bool


class PyStr(_Root):
    value: str


class PyTuple(_Root):
    elems: tuple[SExpr, ...]


class PyList(_Root):
    elems: tuple[SExpr, ...]


class PyCall(_Root):
    func: SExpr
    io: SExpr
    args: tuple[SExpr, ...]


class PyCallKwargs(_Root):
    func: SExpr
    io: SExpr
    args: Posargs
    kwargs: Kwargs


class PyCallPure(_Root):
    func: SExpr
    args: tuple[SExpr, ...]


class PyAttr(_Root):
    io: SExpr
    value: SExpr
    attrname: str


class PySubscript(_Root):
    io: SExpr
    value: SExpr
    index: SExpr


class PySetItem(_Root):
    io: SExpr
    obj: SExpr
    index: SExpr
    value: SExpr


class PySlice(_Root):
    io: SExpr
    lower: SExpr
    upper: SExpr
    step: SExpr


class PyUnaryOp(_Root):
    op: str
    io: SExpr
    operand: SExpr


class PyBinOp(_Root):
    op: str
    io: SExpr
    lhs: SExpr
    rhs: SExpr


class PyBinOpPure(_Root):
    op: str
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


class PyForLoop(_Root):
    iter_arg_idx: int
    indvar_arg_idx: int
    iterlast_arg_idx: int
    body: SExpr
    operands: tuple[SExpr, ...]


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
    srcloc: SExprProto[Loc]
    interloc: SExprProto[Loc]


class ArgRef(_Root):
    idx: int
    name: str


class Grammar(grammar.Grammar):
    start = _Root
