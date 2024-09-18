from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, NamedTuple, Type, cast, dataclass_transform

from sealir import ase


class Grammar:
    start: Type[Rule]

    def __init__(self, tape: ase.Tape) -> None:
        self._tape = tape

    def write(self, rule: Rule) -> ExprWithRule:
        if not isinstance(rule, self.start):
            raise ValueError(f"{type(rule)} is not a {self.start}")
        args = [getattr(rule, k.name) for k in rule._fields]
        expr = self._tape.expr(rule._sexpr_head, *args)
        return ExprWithRule(self, type(rule), expr)

    def downcast(self, expr: ase.BaseExpr) -> ExprWithRule:
        head = expr._head
        rulety = self.start._subtypes[head]
        return ExprWithRule(self, rulety, expr)


@dataclass_transform()
def rule(clsdef: Any) -> Type[Rule]:
    fields: dict[str, Any] = {}
    for k, v in clsdef.__annotations__.items():
        if not k.startswith("_"):
            fields[k] = v
    bases = tuple(t for t in clsdef.__bases__ if t is not object)
    if Rule not in bases:
        bases += (Rule,)
    for b in bases:
        if issubclass(b, Rule):
            for fd in b._fields:
                assert fd.name not in fields
                fields[fd.name] = fd.annotation

    newcls = type(
        clsdef.__name__,
        bases,
        {
            "_fields": tuple(_Field(k, v) for k, v in fields.items()),
            "__match_args__": tuple(fields.keys()),
            "_sexpr_head": clsdef.__name__,
        },
    )
    return newcls


class _Field(NamedTuple):
    name: str
    annotation: Any


class _MetaRule(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        if isinstance(instance, ase.BaseExpr):
            return instance._head == cls._sexpr_head
        else:
            return type.__instancecheck__(cls, instance)


class Rule(metaclass=_MetaRule):
    _sexpr_head: str
    _fields: tuple[_Field, ...] = ()
    _subtypes: dict[str, Type[Rule]] = {}
    __match_args__: tuple[str]

    def __init__(self, **kwargs):
        fields = dict(self._fields)
        for k, v in kwargs.items():
            setattr(self, k, v)
            fields.pop(k)
        if fields:
            raise TypeError(f"missing arguments: {fields}")

    def __init_subclass__(cls) -> None:
        if cls is not Rule:
            super().__init_subclass__()
            first_base = [c for c in cls.__mro__ if issubclass(c, Rule)][1]
            first_base._subtypes[cls.__name__] = cls


class ExprWithRule(ase.BaseExpr):
    def __init__(
        self, grammar: Grammar, rulety: Type[Rule], expr: ase.BaseExpr
    ) -> None:
        self._tape = expr._tape
        self._handle = expr._handle
        self._grammar = grammar
        self._rulety = rulety
        self._slots = {k: i for i, k in enumerate(rulety.__match_args__)}
        self._expr = expr
        self.__match_args__ = rulety.__match_args__

    def __getattr__(self, name: str) -> ase.value_type:
        idx = self._slots[name]
        return self._args[idx]

    def __repr__(self) -> str:
        inner = repr(self._expr)
        return f"{self._rulety._sexpr_head}!{inner}"

    @cached_property
    def _head(self) -> str:
        return self._expr._head

    @cached_property
    def _args(self) -> tuple[ase.value_type, ...]:
        g = self._grammar

        def cast(x: ase.value_type) -> ase.value_type:
            if isinstance(x, ase.BaseExpr):
                return g.downcast(x)
            else:
                return x

        return tuple(map(cast, self._expr._args))

    def _get_downcast(self) -> Callable[[ase.BaseExpr], ExprWithRule]:
        return self._grammar.downcast
