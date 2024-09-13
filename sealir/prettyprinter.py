from __future__ import annotations

from collections import Counter
from typing import Any

from sealir.ase import Expr

from .rewriter import TreeRewriter


def pretty_print(expr: Expr) -> str:
    if expr.is_metadata:
        return str(expr)

    oc = Occurrences()
    expr.apply_bottomup(oc)
    ctr = oc.memo[expr]
    assert isinstance(ctr, Counter)

    formatted: dict[Expr, str] = {}
    ident: dict[Expr, int] = {}
    seen = set()

    def fmt(arg):
        if not isinstance(arg, Expr):
            return repr(arg)
        elif arg.is_simple:
            return formatted.get(arg)

        if arg in seen:
            h = ident[arg]
            return f"${h}"

        seen.add(arg)
        ret = formatted.get(arg)
        if ret is None:
            ret = str(arg)

        if ctr[arg] > 1:
            h = ident.setdefault(arg, len(ident))
            ret += f"[${h}]"

        return ret

    for key in sorted(ctr, key=lambda k: k._handle):
        items = [key.head, *map(fmt, key.args)]
        text = " ".join(items)
        formatted[key] = f"({text})"

    return formatted[expr]


class Occurrences(TreeRewriter[Counter]):
    """Count occurrences of expression used in arguments."""

    def rewrite_generic(
        self, old: Expr, args: tuple[Any, ...], updated: bool
    ) -> Counter:
        ctr = Counter([old])
        for arg in args:
            if isinstance(arg, Counter):
                ctr.update(arg)
        return ctr
