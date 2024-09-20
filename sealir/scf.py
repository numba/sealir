from __future__ import annotations

import inspect
from sealir import lam, grammar

def region(grm: grammar.Grammar):
    """Helper for defining regions that must be a 1-arity lambda."""

    def outer(fn):
        sig = inspect.signature(fn)
        arity = len(sig.parameters)

        @lam.lam_func(grm)
        def body(tup):
            args = lam.unpack(grm, tup, arity)
            bound = sig.bind(*args)
            return fn(*bound.args, **bound.kwargs)

        return body

    return outer
