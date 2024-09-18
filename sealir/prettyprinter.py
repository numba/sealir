from __future__ import annotations

from collections import Counter
from typing import Any

from sealir import ase
from sealir.ase import BaseExpr

from .rewriter import TreeRewriter


def pretty_print(expr: BaseExpr) -> str:
    if ase.is_metadata(expr):
        return str(expr)

    oc = Occurrences()
    ase.apply_bottomup(expr, oc)
    ctr = oc.memo[expr]
    assert isinstance(ctr, Counter)

    formatted: dict[BaseExpr, str] = {}
    ident: dict[BaseExpr, int] = {}
    seen = set()

    def fmt(arg):
        if not isinstance(arg, BaseExpr):
            return repr(arg)
        elif ase.is_simple(arg):
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

    for key in sorted(ctr, key=lambda k: ase.get_handle(k)):
        items = [ase.get_head(key), *map(fmt, ase.get_args(key))]
        text = " ".join(items)
        formatted[key] = f"({text})"

    return formatted[expr]


class Occurrences(TreeRewriter[Counter]):
    """Count occurrences of expression used in arguments."""

    def rewrite_generic(
        self, old: BaseExpr, args: tuple[Any, ...], updated: bool
    ) -> Counter:
        ctr = Counter([old])
        for arg in args:
            if isinstance(arg, Counter):
                ctr.update(arg)
        return ctr
