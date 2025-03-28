from __future__ import annotations

from typing import TypeAlias

from sealir import ase, grammar

SExpr: TypeAlias = ase.SExpr


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
    body: RegionEnd
    orelse: RegionEnd
    operands: tuple[SExpr, ...]


class Loop(_Root):
    body: RegionEnd
    loopvar: str
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
