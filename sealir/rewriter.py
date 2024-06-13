from __future__ import annotations

from typing import TypeVar, Generic, Any, Union

from sealir import ase


T = TypeVar("T")


class _PassThru:
    def __repr__(self) -> str:
        return "<PassThru>"


class TreeRewriter(Generic[T], ase.TreeVisitor):

    PassThru = _PassThru()

    memo: dict[ase.Expr, Union[T, ase.Expr]]

    def __init__(self):
        self.memo = {}

    def visit(self, expr: ase.Expr) -> None:
        with expr.tree:
            res = self._dispatch(expr)
            self.memo[expr] = res

    def _dispatch(self, old: ase.Expr) -> Union[T, ase.Expr]:
        head = old.head
        args = old.args
        updated = False

        def _lookup(val):
            nonlocal updated
            if isinstance(val, ase.Expr):
                updated = True
                return self.memo[val]
            else:
                return val

        args = tuple(_lookup(arg) for arg in args)
        fname = f"rewrite_{head}"
        fn = getattr(self, fname, None)
        if fn is not None:
            return fn(*args)
        else:
            return self.rewrite_generic(old, args, updated)

    def rewrite_generic(
        self, old: ase.Expr, args: tuple[Any, ...], updated: bool
    ) -> Union[T, ase.Expr]:
        """Default implementation will automatically create a new node if
        children are updated; otherwise, returns the original expression if
        its children are unmodified.
        """
        if updated:
            return ase.expr(old.head, *args)
        else:
            return old
