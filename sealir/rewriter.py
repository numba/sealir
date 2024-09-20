from __future__ import annotations

from typing import Any, Generic, TypeVar, Union

from sealir import ase

T = TypeVar("T")


class TreeRewriter(Generic[T], ase.TreeVisitor):

    memo: dict[ase.BaseExpr, Union[T, ase.BaseExpr]]

    flag_save_history = True

    def __init__(self):
        self.memo = {}

    def visit(self, expr: ase.BaseExpr) -> None:
        if expr in self.memo:
            return
        res = self._dispatch(expr)
        self.memo[expr] = res
        # Logic for save history
        if self.flag_save_history:
            if res != expr and isinstance(res, ase.BaseExpr):
                # Insert code that maps replacement back to old
                cls = type(self)
                tp = expr._tape
                tp.expr(
                    ".md.rewrite",
                    f"{cls.__module__}.{cls.__qualname__}",
                    res,
                    expr,
                )

    def _dispatch(self, orig: ase.BaseExpr) -> T | ase.BaseExpr:
        args = orig._args
        updated = False

        def _lookup(val):
            nonlocal updated
            if isinstance(val, ase.BaseExpr):
                updated = True
                return self.memo[val]
            else:
                return val

        args = tuple(_lookup(arg) for arg in args)

        if updated:
            self._passthru_state = lambda: orig._replace(*args)
        else:
            self._passthru_state = lambda: orig

        res = self._default_rewrite_dispatcher(orig, updated, args)

        return res

    def passthru(self) -> ase.BaseExpr:
        return self._passthru_state()

    def rewrite_generic(
        self, orig: ase.BaseExpr, args: tuple[Any, ...], updated: bool
    ) -> T | ase.BaseExpr:
        """Default implementation will automatically create a new node if
        children are updated; otherwise, returns the original expression if
        its children are unmodified.
        """
        if updated:
            return orig._replace(*args)
        else:
            return orig

    def _default_rewrite_dispatcher(
        self,
        orig: ase.BaseExpr,
        updated: bool,
        args: tuple[T | ase.value_type],
    ) -> T | ase.BaseExpr:
        fname = f"rewrite_{orig._head}"
        fn = getattr(self, fname, None)
        if fn is not None:
            return fn(orig, *args)
        else:
            return self.rewrite_generic(orig, args, updated)
