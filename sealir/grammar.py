from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, MutableMapping
from functools import cached_property, lru_cache
from itertools import chain, cycle
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    NamedTuple,
    Protocol,
    Self,
    Sequence,
    Type,
    TypeVar,
    cast,
    dataclass_transform,
    get_type_hints,
    runtime_checkable,
)

from sealir import ase, rewriter

Trule = TypeVar("Trule", bound="Rule")
Tgrammar = TypeVar("Tgrammar", bound="Grammar")
T = TypeVar("T")
# Covariant type variable for protocols
Trule_co = TypeVar("Trule_co", bound="Rule", covariant=True)


class Grammar:
    start: type[Rule] | UnionType

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

    def write(self: Tgrammar, rule: Trule) -> NamedSExpr[Tgrammar, Trule]:
        bounded = NamedSExpr._subclass(type(self), type(rule))
        if not isinstance(rule, self.start):
            raise ValueError(f"{type(rule)} is not a {self.start}")
        args = rule._get_sexpr_args()
        expr = self._tape.expr(rule._sexpr_head, *args)
        return bounded._wrap(expr._tape, expr._handle)

    @classmethod
    @lru_cache(maxsize=4096)
    def downcast(
        cls: type[Tgrammar], expr: ase.SExpr
    ) -> NamedSExpr[Tgrammar, Trule]:
        head = expr._head
        try:
            rulety = cls.start._rules[head]     # type: ignore[union-attr]
        except KeyError:
            raise ValueError(f"{head!r} is not valid in the grammar")
        else:
            return cast(NamedSExpr[Tgrammar, Trule], NamedSExpr._subclass(cls, rulety)(expr))

    def __enter__(self) -> Self:
        self._tape.__enter__()
        return self

    def __exit__(self, exc_val, exc_typ, exc_tb) -> None:
        self._tape.__exit__(exc_val, exc_typ, exc_tb)


class _Field(NamedTuple):
    name: str
    annotation: Any

    def is_vararg(self) -> bool:
        return isinstance(self.annotation, type(tuple[Any]))


# Protocol definitions for mypy compatibility


@runtime_checkable
class SExprProto(Protocol, Generic[Trule_co]):
    """Generic protocol for SExpr and NamedSExpr to duck-type as Rule.
    """

    def _castable_to_SExprProto(self) -> None: ...

    def __getattr__(self, name: str) -> Any:
        """Provides dynamic field access (works for both Rule and NamedSExpr)"""
        ...


@dataclass_transform()
class _MetaRule(type):
    _fields: tuple[_Field, ...]
    __match_args__: tuple[str, ...]
    _is_root: bool
    _sexpr_head: str
    _rules: MutableMapping[str, Type[Rule]]

    def __instancecheck__(cls, instance: Any) -> bool:
        if issubclass(cls, _CombinedRule):
            return any(isinstance(instance, t) for t in cls._combined)
        if isinstance(instance, ase.SExpr):
            return instance._head == cls._sexpr_head  # type: ignore[attr-defined]
        else:
            return type.__instancecheck__(cls, instance)


class Rule(metaclass=_MetaRule):
    """Base class for all grammar rules.

    Implements SExprProto protocol through metaclass and duck typing.
    """

    # Explicit protocol satisfaction for mypy
    if TYPE_CHECKING:
        def __getattr__(self, name: str) -> Any:
            """Protocol method - satisfied by actual field access"""
            ...
        def _castable_to_SExprProto(self) -> None: ...

    def __init__(self, *args, **kwargs):
        if args:
            if len(self._fields) > 1:
                # NOTE: this restriction is to avoid silent error when fields
                #       are changed (esp. ordering of fields)
                raise TypeError(
                    "only Rule of a single field can use "
                    "positional argument"
                )

        fields = dict(self._fields)
        peel_args = chain(
            zip(chain(self.__match_args__, cycle([None])), args),
            kwargs.items(),
        )
        for k, v in peel_args:
            if k is None:
                raise TypeError("too many positional arguments")
            if hasattr(self, k):
                raise TypeError(f"duplicated keyword: {k}")
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
        cls.__match_args__ = tuple(fields.keys())  # type: ignore[misc]
        cls._sexpr_head = cls.__name__
        cls._is_root = not root_bases
        if cls._is_root:
            # only root has a fresh _rules
            cls._rules = {}
        # insert this class to _rules
        cls._rules[cls._sexpr_head] = cls
        cls._verify()

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

    @classmethod
    def _verify(cls):
        for fd in cls._fields[:-1]:
            if fd.is_vararg():
                raise TypeError(
                    f"Field {fd.name} must not be vararg. "
                    "Only the last field can be vararg."
                )

    @classmethod
    def _field_position(cls, attrname: str) -> int:
        for i, fd in enumerate(cls._fields):
            if fd.name == attrname:
                return i
        raise NameError(f"{attrname} not in {cls}")


class _CombinedRule(Rule):
    _combined: tuple[type[Rule], ...]


class NamedSExpr(ase.SExpr, Generic[Tgrammar, Trule]):
    _grammar: type[Grammar]
    _rulety: type[Trule]

    @classmethod
    def _subclass(
        cls, grammar: type[Tgrammar], rule: type[Trule]
    ) -> type[NamedSExpr[Tgrammar, Trule]]:
        assert issubclass(grammar, Grammar)
        assert issubclass(rule, Rule)
        return type(
            cls.__name__, (NamedSExpr,), dict(_grammar=grammar, _rulety=rule)
        )

    @classmethod
    def _wrap(cls, tape: ase.Tape, handle: ase.handle_type) -> Self:
        return cls(ase.BasicSExpr._wrap(tape, handle))

    def __init__(self, expr: ase.SExpr) -> None:
        assert self._grammar
        assert self._rulety
        self._tape = expr._tape
        self._handle = expr._handle
        rulety = self._rulety
        self._slots = {k: i for i, k in enumerate(rulety.__match_args__)}
        self._expr = expr
        self.__match_args__ = rulety.__match_args__  # type: ignore[misc]

    def __getattr__(
        self, name: str
    ) -> ase.value_type | tuple[ase.value_type, ...]:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            idx = self._slots[name]
        except KeyError:
            raise AttributeError(
                f"{self._head} doesn't have attribute {name!r}"
            )

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
            if isinstance(x, ase.SExpr):
                return g.downcast(x)
            else:
                return x

        values = tuple(map(cast, self._expr._args))
        return values

    def _get_downcast(self) -> Callable[[ase.SExpr], NamedSExpr]:
        return self._grammar.downcast

    def _replace(self, *args: ase.value_type) -> NamedSExpr[Tgrammar, Trule]:
        return self._grammar.downcast(self._expr._replace(*args))

    def _bind(self, *args) -> Mapping[str, Any]:
        npos = len(self.__match_args__)
        pack_last = False
        if self._rulety._fields[-1].is_vararg():
            npos -= 1
            pack_last = True
        out = {}
        i = 0
        for i in range(npos):
            out[self.__match_args__[i]] = args[i]

        if pack_last:
            out[self.__match_args__[-1]] = tuple(args[i:])
        return out

    def _serialize(self) -> tuple:
        """
        Returns an object that is suitable for serializing.
        """
        return self._grammar, self._tape, self._handle


def deserialize(payload) -> NamedSExpr:
    """Reconstruct a NamedSExpr given the output from `NamedSExpr._serialize()`."""
    grm, tape, handle = payload
    return grm.downcast(ase.BasicSExpr(tape, handle))


class TreeRewriter(rewriter.TreeRewriter[T]):
    grammar: Grammar | None = None

    def __init__(self, grammar: Grammar | None = None):
        super().__init__()
        if grammar is not None:
            self.grammar = grammar

    def _default_rewrite_dispatcher(
        self,
        orig: ase.SExpr,
        updated: bool,
        args: tuple[T | ase.value_type],
    ) -> T | ase.SExpr:

        if self.grammar is None:
            assert isinstance(orig, NamedSExpr), repr(orig)
        else:
            orig = self.grammar.downcast(orig)
        fname = f"rewrite_{orig._head}"
        fn = getattr(self, fname, None)
        if fn is not None:
            return fn(orig, **orig._bind(*args))
        else:
            return self.rewrite_generic(orig, args, updated)


def field_position(grm: type[Rule], attr: str) -> int:
    return grm._field_position(attr)
