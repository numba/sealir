from typing import Iterator, TypeVar

T = TypeVar("T")


def first(iterator: Iterator[T]) -> T:
    return next(iter(iterator))


def maybe_first(iterator: Iterator[T]) -> T | None:
    try:
        return first(iterator)
    except StopIteration:
        return None
