"""
Condition-based dispatch table implementation.

Provides DispatchTableBuilder for construction and DispatchTable for runtime
dispatch. Cases are evaluated in order; first matching case executes.

Example:
    @dispatchtable
    def handler(builder):
        @builder.case(lambda x: x > 0)
        def positive(x): return f"pos: {x}"

        @builder.default
        def default(x): return f"default: {x}"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Self, Sequence

TCond = Callable[..., bool]
"""Type alias for condition functions that return boolean values."""


@dataclass(frozen=True)
class Case:
    """
    Dispatch case: handler function with condition predicates.

    Attributes:
        fn: Handler function executed when all conditions are True
        conditions: Condition predicates that must all return True
    """
    fn: Callable
    conditions: tuple[TCond, ...]


def _default_function(*args, **kwargs):
    """Default handler that raises NotImplementedError."""
    raise NotImplementedError(args, kwargs)


@dataclass
class DispatchTableBuilder:
    """
    Mutable builder for constructing dispatch tables.

    Accumulates cases and default handler, then builds immutable DispatchTable.

    Attributes:
        _cases: List of Case objects
        _default: Default handler function
    """
    _cases: list[Case] = field(default_factory=list)
    _default: Callable = _default_function

    @classmethod
    def get(cls, disp: DispatchTable) -> DispatchTableBuilder:
        """Copy cases and default handler from existing DispatchTable."""
        return DispatchTableBuilder(list(disp.cases), disp.default)

    def case(self, *conditions: TCond) -> Callable[[Callable], Callable]:
        """
        Append a handler for conditions.

        All conditions must return True for handler selection.
        """
        def wrap(fn: Callable) -> Callable:
            self._cases.append(Case(fn=fn, conditions=conditions))
            return fn
        return wrap

    def default(self, fn) -> Self:
        """Set default handler for unmatched cases."""
        self._default = fn
        return self

    def build(self) -> "DispatchTable":
        """Build immutable DispatchTable from current configuration."""
        return DispatchTable.get(self._cases, self._default)


@dataclass(frozen=True)
class DispatchTable:
    """
    Immutable dispatch table for runtime function routing.

    Evaluates cases in order; executes first handler where all conditions are
    True.

    Attributes:
        cases: Tuple of Case objects
        default: Default handler function
    """
    cases: tuple[Case, ...]
    default: Callable

    @classmethod
    def get(cls, cases: Sequence[Case], default: Callable):
        """Construct DispatchTable from cases and default handler."""
        return DispatchTable(cases=tuple(cases), default=default)

    def __call__(self, *args, **kwargs):
        """Execute first handler where all conditions are True, else default."""
        for case in self.cases:
            if all(cond(*args, **kwargs) for cond in case.conditions):
                return case.fn(*args, **kwargs)
        return self.default(*args, **kwargs)

    def extend(
        self, def_func: Callable[[DispatchTableBuilder], None]
    ) -> "DispatchTable":
        """Create new DispatchTable with additional cases from def_func."""
        builder = DispatchTableBuilder.get(self)
        def_func(builder)
        return builder.build()


def dispatchtable(
    init_fn: Callable[[DispatchTableBuilder], None],
) -> DispatchTable:
    """
    Create dispatch table from configuration function.

    Args:
        init_fn: Function that configures DispatchTableBuilder

    Returns:
        Immutable DispatchTable
    """
    builder = DispatchTableBuilder()
    init_fn(builder)
    return builder.build()
