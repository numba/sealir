import copy
from dataclasses import dataclass
from typing import Any, Callable, Self

TCond = Callable[[Any], bool]


@dataclass(frozen=True)
class Case:
    fn: Callable
    conditions: tuple[TCond, ...]


class DispatchTable:
    _cases: list[Case]
    _default: Callable

    def __init__(self):
        self._cases = []
        self._default = self._default_function

    def case(self, *conditions: TCond) -> Callable[[Callable], Callable]:
        def wrap(fn: Callable) -> Callable:
            self._cases.append(Case(fn=fn, conditions=conditions))
            return fn
        return wrap

    def default(self, fn) -> Self:
        self._default = fn
        return self

    def __call__(self, *args, **kwargs):
        for case in self._cases:
            if all(cond(*args, **kwargs) for cond in case.conditions):
                return case.fn(*args, **kwargs)
        return self._default(*args, **kwargs)

    @staticmethod
    def _default_function(*args, **kwargs):
        raise NotImplementedError(args, kwargs)

    def copy(self) -> "DispatchTable":
        dt = DispatchTable()
        dt._cases = self._cases.copy()
        dt._default = self._default
        return dt



def dispatchtable(default_fn):
    disp = DispatchTable()
    disp.default(default_fn)
    return disp
