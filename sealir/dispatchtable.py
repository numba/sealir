import inspect
import operator
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Self, get_type_hints

TCond = Callable[[Any], bool]


@dataclass(frozen=True)
class Case:
    fn: Callable
    conditions: tuple[TCond, ...]


# class Matcher:
#     Eq = staticmethod(lambda x: partial(operator.eq, x))

#     _cases: list[Case]
#     _default: Callable | None

#     def __init__(self):
#         self._cases = []
#         self._default = None

#     def case(self, *conditions: TCond) -> Callable[[Callable], Callable]:
#         if self._default is not None:
#             raise ValueError("cannot add more cases after 'default' is added")

#         def wrap(fn: Callable) -> Callable:
#             self._cases.append(Case(fn=fn, conditions=conditions))
#             return fn

#         return wrap

#     def default(self, fn) -> Self:
#         assert fn is not None
#         self._default = fn
#         return self

#     def __call__(self, *args, **kwargs):
#         for case in self._cases:
#             sig = inspect.signature(case.fn)
#             bound = sig.bind(*args, **kwargs)

#             not_match = False

#             # match by fixed conditions
#             nc = len(case.conditions)
#             for arg, cond in zip(
#                 bound.args[:nc], case.conditions, strict=True
#             ):
#                 if not cond(arg):
#                     not_match = True

#             if not_match:
#                 continue

#             # # do runtime type checking
#             # annotations = get_type_hints(case.fn)
#             # for k, v in bound.arguments.items():
#             #     anno = annotations[k]
#             #     if anno is not inspect._empty and not isinstance(v, anno):
#             #         not_match = True
#             #         break
#             if not_match:
#                 continue
#             return case.fn(**bound.arguments)
#         if self._default is None:
#             raise NotImplementedError(f"cannot match for ({args}, {kwargs})")
#         return self._default(*args, **kwargs)


# def test_usecase():

#     matcher = Matcher()

#     Eq = matcher.Eq

#     @matcher.case(Eq("123"))
#     def _(a: str, b: int):
#         assert isinstance(b, int)
#         assert a == "123"
#         print("Select 1", a, b)
#         return 1

#     @matcher.case(Eq("123"))
#     def _(a: str, b: float):
#         assert isinstance(b, float)
#         assert a == "123"
#         print("Select 2", a, b)
#         return 2

#     @matcher.default
#     def _(a, b):
#         print("Select otherwise", a, b)
#         return 3

#     assert matcher(a="123", b=0) == 1
#     assert matcher(a="123", b=0.0) == 2
#     assert matcher(a="12", b=0.0) == 3


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
