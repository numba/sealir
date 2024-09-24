from __future__ import annotations

from collections import deque
from typing import Any, Generic, NamedTuple, TypeVar, Union

from sealir import ase

T = TypeVar("T")


class MdRewriteLayout(NamedTuple):
    rewriter: str
    repl: ase.SExpr
    orig: ase.SExpr


def insert_metadata(rewriter: str, repl: ase.SExpr, orig: ase.SExpr):
    assert orig._tape == repl._tape
    tp = orig._tape
    tp.expr(".md.rewrite", *MdRewriteLayout(rewriter, repl, orig))


def insert_metadata_map(memo: dict[ase.SExpr, ase.SExpr]):
    """Helper to write metadata from a memo dictionary."""
    for orig, repl in memo.items():
        if isinstance(repl, ase.SExpr):
            insert_metadata(
                ".md.rewrite",
                repl,
                orig,
            )


def metadata_find_original(node: ase.SExpr, loc_test) -> ase.SExpr | None:
    def iter_md_orig(node: ase.SExpr):
        stack = deque([node])
        while stack:
            node = stack.pop()
            for parent in ase.walk_parents(node):
                if parent._head == ".md.rewrite":
                    md = MdRewriteLayout(*parent._args)
                    if md.orig != node:
                        if loc_test(md.orig):
                            yield md.orig
                        else:
                            stack.append(md.orig)
                else:
                    stack.appendleft(parent)

    mds = iter_md_orig(node)
    try:
        return next(iter(mds))
    except StopIteration:
        return None


class TreeRewriter(Generic[T], ase.TreeVisitor):

    memo: dict[ase.SExpr, Union[T, ase.SExpr]]

    flag_save_history = True

    def __init__(self):
        self.memo = {}

    def visit(self, expr: ase.SExpr) -> None:
        if expr in self.memo:
            return
        res = self._dispatch(expr)
        self.memo[expr] = res
        # Logic for save history
        if self.flag_save_history:
            if (
                isinstance(res, ase.SExpr)
                and res != expr
                and res._tape == expr._tape
            ):
                # Insert code that maps replacement back to old
                cls = type(self)
                insert_metadata(
                    f"{cls.__module__}.{cls.__qualname__}",
                    res,
                    expr,
                )

    def _dispatch(self, orig: ase.SExpr) -> T | ase.SExpr:
        args = orig._args
        updated = False

        def _lookup(val):
            nonlocal updated
            if isinstance(val, ase.SExpr):
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

    def passthru(self) -> ase.SExpr:
        return self._passthru_state()

    def rewrite_generic(
        self, orig: ase.SExpr, args: tuple[Any, ...], updated: bool
    ) -> T | ase.SExpr:
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
        orig: ase.SExpr,
        updated: bool,
        args: tuple[T | ase.value_type],
    ) -> T | ase.SExpr:
        fname = f"rewrite_{orig._head}"
        fn = getattr(self, fname, None)
        if fn is not None:
            return fn(orig, *args)
        else:
            return self.rewrite_generic(orig, args, updated)
