from __future__ import annotations

from collections import ChainMap
from collections.abc import MutableMapping
from functools import cached_property
from itertools import chain
from types import UnionType
from typing import (
    Any,
    Callable,
    NamedTuple,
    Self,
    Sequence,
    Type,
    TypeVar,
    dataclass_transform,
    get_type_hints,
)

from sealir import ase

Trule = TypeVar("Trule", bound="Rule")


class Grammar:
    start: Type[Rule]

    def __init__(self, tape: ase.Tape) -> None:
        self._tape = tape

    def __init_subclass__(cls) -> None:
        if isinstance(cls.start, UnionType):
            newcls: type[_CombinedRule] = type(
                "_CombinedRule",
                (_CombinedRule,),
                {
                    "_combined": tuple(cls.start.__args__),
                },
            )
            newcls._rules = ChainMap(*(t._rules for t in cls.start.__args__))
            cls.start = newcls

    def write(self, rule) -> ExprWithRule:
        if not isinstance(rule, self.start):
            raise ValueError(f"{type(rule)} is not a {self.start}")
        args = rule._get_sexpr_args()
        expr = self._tape.expr(rule._sexpr_head, *args)
        return ExprWithRule(self, type(rule), expr)

    def downcast(self, expr: ase.BaseExpr) -> ExprWithRule:
        head = expr._head
        rulety = self.start._rules[head]
        return ExprWithRule(self, rulety, expr)

    def __enter__(self) -> Self:
        self._tape.__enter__()
        return self

    def __exit__(self, exc_val, exc_typ, exc_tb) -> None:
        self._tape.__exit__(exc_val, exc_typ, exc_tb)


@dataclass_transform()
def rule(clsdef: type) -> Type[Rule]:
    fields: dict[str, Any] = {}

    hints = get_type_hints(clsdef)
    for k, v in hints.items():
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
            "_rules": {},
        },
    )
    return newcls


class _Field(NamedTuple):
    name: str
    annotation: Any

    def is_vararg(self) -> bool:
        return isinstance(self.annotation, type(tuple[Any]))


@dataclass_transform()
class _MetaRule(type):
    _fields: tuple[_Field, ...]
    __match_args__: tuple[str]
    _is_root: bool
    _sexpr_head: str
    _rules: MutableMapping[str, Type[Rule]]

    def __instancecheck__(cls, instance: Any) -> bool:
        if issubclass(cls, _CombinedRule):
            return any(isinstance(instance, t) for t in cls._combined)
        if isinstance(instance, ase.BaseExpr):
            return instance._head == cls._sexpr_head
        else:
            return type.__instancecheck__(cls, instance)


class Rule(metaclass=_MetaRule):

    def __init__(self, *args, **kwargs):
        fields = dict(self._fields)
        peel_args = chain(
            zip(self.__match_args__, args),
            kwargs.items(),
        )
        for k, v in peel_args:
            setattr(self, k, v)
            fields.pop(k)
        if fields:
            raise TypeError(f"missing arguments: {fields}")

    def __init_subclass__(cls) -> None:
        # Find the root for this grammar
        root_bases = [
            c
            for c in reversed(cls.__mro__[1:])
            if issubclass(c, Rule) and getattr(c, "_is_root", False)
        ]
        assert 0 <= len(root_bases) <= 1

        # init subclass fields
        fields: dict[str, Any] = {}

        hints = get_type_hints(cls)
        for k, v in hints.items():
            if not k.startswith("_"):
                fields[k] = v

        cls._fields = tuple(_Field(k, v) for k, v in fields.items())
        cls.__match_args__ = tuple(fields.keys())
        cls._sexpr_head = cls.__name__
        cls._is_root = not root_bases
        if cls._is_root:
            # only root has a fresh _rules
            cls._rules = {}
        # insert this class to _rules
        cls._rules[cls._sexpr_head] = cls

    def __repr__(self):
        name = self._sexpr_head
        args = [
            f"{fd.name}={val!r}"
            for fd, val in self._get_field_values().items()
        ]
        return f"{name}({', '.join(args)})"

    def _get_field_values(self) -> dict:
        return {fd: getattr(self, fd.name) for fd in self._fields}

    def _get_sexpr_args(self) -> Sequence[ase.value_type]:
        out: list[ase.value_type] = []
        for val in self._get_field_values().values():
            if isinstance(val, tuple):
                out.extend(val)
            else:
                out.append(val)
        return out


class _CombinedRule(Rule):
    _combined: tuple[type[Rule], ...]


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

    def __getattr__(
        self, name: str
    ) -> ase.value_type | tuple[ase.value_type, ...]:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            idx = self._slots[name]
        except IndexError:
            raise AttributeError(name)

        if idx + 1 == len(self._rulety._fields):
            last_fd = self._rulety._fields[-1]
            if last_fd.is_vararg():
                return tuple(self._args[idx:])

        return self._args[idx]

    def __repr__(self) -> str:
        inner = repr(self._expr)
        return inner

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

        values = tuple(map(cast, self._expr._args))
        return values

    def _get_downcast(self) -> Callable[[ase.BaseExpr], ExprWithRule]:
        return self._grammar.downcast
