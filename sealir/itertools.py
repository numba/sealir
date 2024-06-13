from typing import Iterator, TypeVar

T = TypeVar("T")


def first(iterator: Iterator[T]) -> T:
    return next(iter(iterator))
