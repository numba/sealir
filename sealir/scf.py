from __future__ import annotations

import inspect


def region(lambar):
    """Helper for defining regions that must be a 1-arity lambda."""

    def outer(fn):
        sig = inspect.signature(fn)
        arity = len(sig.parameters)

        @lambar.lam_func
        def body(tup):
            args = lambar.unpack(tup, arity)
            bound = sig.bind(*args)
            return fn(*bound.args, **bound.kwargs)

        return body

    return outer
