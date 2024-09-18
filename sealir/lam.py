from __future__ import annotations

import inspect
import types
from collections import Counter, defaultdict
from typing import Any, NamedTuple

from sealir import ase
from sealir.itertools import first
from sealir.rewriter import TreeRewriter


class _app(NamedTuple):
    arg: ase.BaseExpr
    body: ase.BaseExpr


class LamBuilder:
    """Helper for building lambda calculus expression"""

    _tape: ase.Tape

    def __init__(self, tape: ase.Tape | None = None):
        self._tape = tape or ase.Tape()

    @property
    def tape(self) -> ase.Tape:
        return self._tape

    def get_root(self) -> ase.BaseExpr:
        return ase.SimpleExpr(self._tape, self._tape.last())

    def expr(self, head: str, *args) -> ase.BaseExpr:
        """Makes a user defined expression"""
        with self._tape as tp:
            return tp.expr("expr", head, *args)

    def lam(self, body_expr: ase.BaseExpr) -> ase.BaseExpr:
        """Makes a lambda abstraction"""
        with self._tape as tp:
            return tp.expr("lam", body_expr)

    def lam_func(self, fn):
        """Decorator to help build lambda expressions from a Python function.
        Takes multiple arguments and uses De Bruijn index.
        """
        argspec = inspect.getfullargspec(fn)
        argnames = argspec.args
        assert not argspec.varargs
        assert not argspec.kwonlyargs

        with self._tape:
            # use De Bruijn index starting from 0
            # > lam (lam arg0 arg1)
            #         ^---|     |
            #    ^--------------|
            args = [self.arg(i) for i in range(len(argnames))]
            # intercept cell variables
            fn = self._intercept_cells(fn)
            # Reverse arguments so that arg0 refers to the innermost lambda.
            # And, it is the rightmost argument in Python.
            # That way, first `(app )` will replace the leftmost Python argument
            expr = fn(*reversed(args))
            for _ in args:
                expr = self.lam(expr)
            return expr

    # Helper API

    def unpack(
        self, tup: ase.BaseExpr, nelem: int
    ) -> tuple[ase.BaseExpr, ...]:
        """Unpack a tuple with known size with `(tuple.getitem )`"""
        return tuple(
            map(lambda i: self.expr("tuple.getitem", tup, i), range(nelem))
        )

    def _intercept_cells(self, fn):
        if fn.__closure__ is None:
            return fn
        changed = False
        new_closure = []
        for i, cell in enumerate(fn.__closure__):
            val = cell.cell_contents
            if isinstance(val, ase.BaseExpr):
                if ase.get_head(val) == "arg":
                    # increase de bruijn index
                    [idx] = ase.get_args(val)
                    new_closure.append(types.CellType(self.arg(idx + 1)))
                    changed = True
                elif ase.get_head(val) != "lam":
                    # cannot refer to Expr by freevars
                    raise ValueError(
                        "cannot refer to non-argument Expr by freevars"
                    )
            else:
                new_closure.append(cell)

        if not changed:
            return fn

        return types.FunctionType(
            fn.__code__,
            fn.__globals__,
            fn.__name__,
            fn.__defaults__,
            tuple(new_closure),
        )

    def arg(self, index: int) -> ase.BaseExpr:
        with self._tape as tp:
            return tp.expr("arg", index)

    def app(self, lam, arg0, *more_args) -> ase.BaseExpr:
        """Makes an apply expression."""
        args = (arg0, *more_args)
        with self._tape as tp:
            stack = list(args)
            out = lam
            while stack:
                arg = stack.pop()
                out = tp.expr("app", *_app(body=out, arg=arg))
        return out

    def beta_reduction(self, app_expr: ase.BaseExpr) -> ase.BaseExpr:
        """Reduces a (app (lam ...) arg)"""
        assert ase.get_head(app_expr) == "app"

        target = app_expr

        app_exprs = []
        while ase.get_head(app_expr) == "app":
            app_exprs.append(app_expr)
            app_expr = _app(*ase.get_args(app_expr)).body
        napps = len(app_exprs)

        arg2repl = {}
        drops = set(app_exprs)
        for parents, child in ase.walk_descendants(app_expr):
            if ase.get_head(child) == "arg":
                lams = [x for x in parents if ase.get_head(x) == "lam"]
                if len(lams) <= napps:  # don't go deeper
                    [debruijn] = ase.get_args(child)
                    # in range?
                    if isinstance(debruijn, int) and debruijn < len(lams):
                        arg2repl[debruijn] = _app(
                            *ase.get_args(app_exprs[-debruijn - 1]),
                        ).arg
                        drops.add(lams[-debruijn - 1])

        assert arg2repl
        br = BetaReduction(drops, arg2repl)
        with self._tape:
            ase.apply_bottomup(target, br)
        out = br.memo[target]
        return out

    def format(self, expr: ase.BaseExpr) -> str:
        """Multi-line formatting of the S-expression"""
        return format_lambda(expr).get()

    def simplify(self) -> LamBuilder:
        """Make a copy and remove dead node. Last node is assumed to be root."""
        last = ase.SimpleExpr(self._tape, self._tape.last())
        new_tree = ase.Tape()
        ase.copy_tree_into(last, new_tree)
        return LamBuilder(new_tree)

    def render_dot(self, **kwargs):
        return self._tape.render_dot(**kwargs)

    def run_abstraction_pass(self, root_expr: ase.BaseExpr):
        """Convert expressions that would otherwise require a let-binding
        to use lambda-abstraction with an application.
        """

        while True:
            ctr: dict[ase.BaseExpr, int] = Counter()
            seen = set()

            class Occurrences(ase.TreeVisitor):
                # TODO: replace prettyprinter one with this
                def visit(self, expr: ase.BaseExpr):
                    if expr not in seen:
                        seen.add(expr)
                        for arg in ase.get_args(expr):
                            if isinstance(arg, ase.BaseExpr) and not is_simple(
                                arg
                            ):
                                ctr.update([arg])

            ase.apply_topdown(root_expr, Occurrences())

            multi_occurring = [
                expr for expr, freq in reversed(ctr.items()) if freq > 1
            ]
            if not multi_occurring:
                break

            repl = {}
            # Handle the oldest (outermost) node first
            expr = min(multi_occurring)

            # find parent lambda
            def parent_containing_this(x: ase.BaseExpr):
                return ase.get_head(x) == "lam"

            host_lam = first(
                ase.search_ancestors(expr, parent_containing_this)
            )
            # find remaining expressions in the lambda
            old_node, repl_node = replace_by_abstraction(self, host_lam, expr)
            repl[old_node] = repl_node

            # Rewrite the rest
            class RewriteProgram(TreeRewriter):
                def rewrite_generic(
                    self,
                    old: ase.BaseExpr,
                    args: tuple[Any, ...],
                    updated: bool,
                ) -> Any | ase.BaseExpr:
                    if old in repl:
                        return repl[old]
                    return super().rewrite_generic(old, args, updated)

            rewriter = RewriteProgram()
            with self.tape:
                ase.apply_bottomup(root_expr, rewriter)
            root_expr = rewriter.memo[root_expr]

        return root_expr


def replace_by_abstraction(
    lambar: LamBuilder, lamexpr: ase.BaseExpr, anchor: ase.BaseExpr
) -> tuple[ase.BaseExpr, ase.BaseExpr]:
    """
    Returns a `( old_node, (app (lam rewritten_old_node) anchor) )` tuple.
    """
    # Find lambda depth of children in the lambda node
    lam_depth_map = {}
    for parents, child in ase.walk_descendants_depth_first_no_repeat(lamexpr):
        lam_depth_map[child] = len(
            [p for p in parents if ase.get_head(p) == "lam"]
        )

    # rewrite these children nodes into a new lambda abstraction
    # replacing the `anchor` node with an arg node.
    [old_node] = ase.get_args(lamexpr)
    new_node = rewrite_into_abstraction(
        lambar, old_node, anchor, lam_depth_map
    )
    repl = lambar.app(lambar.lam(new_node), anchor)
    return old_node, repl


def rewrite_into_abstraction(
    lambar: LamBuilder,
    root: ase.BaseExpr,
    anchor: ase.BaseExpr,
    lam_depth_map: dict[ase.BaseExpr, int],
) -> ase.BaseExpr:

    arg_index = lam_depth_map[anchor] - 1

    # replace remaining program as (app (lam body) expr)
    # in body, shift de bruijin index + 1 for those referring to
    # outer lambdas before introducing new (arg 0)
    class RewriteAddArg(TreeRewriter):
        def rewrite_arg(self, orig: ase.BaseExpr, index: int):
            if orig not in lam_depth_map:
                return self.PassThru
            depth_offset = lam_depth_map[orig] - lam_depth_map[anchor]
            if index - depth_offset < arg_index:
                return self.PassThru
            else:
                return lambar.arg(index + 1)

        def rewrite_generic(
            self, orig: ase.BaseExpr, args: tuple[Any, ...], updated: bool
        ) -> Any | ase.BaseExpr:
            if orig == anchor:
                return lambar.arg(arg_index)
            return super().rewrite_generic(orig, args, updated)

    rewrite = RewriteAddArg()
    with lambar.tape:
        print("running RewriteAddArg", len(lambar.tape))
        ase.apply_bottomup(root, rewrite)
    out_expr = rewrite.memo[root]
    return out_expr


def format_lambda(expr: ase.BaseExpr):
    class Writer:
        def __init__(self):
            self.items = []

        def write(self, *args):
            self.items.extend(args)

        def extend(self, args):
            self.items.extend(args)

        def get(self) -> str:
            import io

            lines = []

            cur = []
            depth = [0]

            def indent():
                return "  " * (len(depth) - 1)

            for it in self.items:

                if it == "{":
                    cur.append(it)
                    lines.append(indent() + " ".join(cur))
                    depth.append(len(lines[-1]))
                    cur = []
                elif it == "}":
                    lines.append(indent() + " ".join(cur))
                    depth.pop()
                    cur = [it]
                elif it == "let":
                    if cur:
                        lines.append(indent() + " ".join(cur))
                    cur = [it]
                else:
                    cur.append(it)
            if cur:
                lines.append(indent() + " ".join(cur))

            return "\n".join(lines)

    descendants = list(ase.walk_descendants(expr))

    def first_lam_parent(parents) -> ase.BaseExpr | None:
        for p in reversed(parents):
            if ase.get_head(p) == "lam":
                return p
        return None

    def fmt(node):
        ret = formatted.get(node)
        if ret is None:
            return repr(node)
        return ret

    class LamScope:
        def __init__(self):
            self.formatted = dict()
            self.lambda_depth = 1
            self.writer = Writer()

    def compute_lam_depth(expr):
        depth = 0
        while ase.get_head(expr) == "lam":
            depth += 1
            expr = ase.get_args(expr)[0]
        return depth

    def flush(child_scope, wr):
        assert child_scope.formatted
        assert len(child_scope.formatted) == 1
        [out] = child_scope.formatted.values()
        wr.write(out)

    grouped: defaultdict[ase.BaseExpr | None, LamScope]
    grouped = defaultdict(LamScope)

    for ident, (parents, child) in enumerate(reversed(descendants)):
        lam_parent = first_lam_parent(parents)
        scope = grouped[lam_parent]
        formatted, wr = scope.formatted, scope.writer
        if child not in formatted:
            if is_simple(child) and parents:
                parts = [
                    f"{ase.get_head(child)}",
                    *map(fmt, ase.get_args(child)),
                ]
                formatted[child] = f"({" ".join(parts)})"
            elif ase.get_head(child) == "lam":
                child_scope = grouped[child]
                child_scope.lambda_depth = compute_lam_depth(child)

                if not parents or ase.get_head(parents[-1]) != "lam":
                    # top-level lambda in this chain
                    wr.write(
                        "let",
                        f"${ident}",
                        "=",
                        "λ" * (child_scope.lambda_depth),
                    )
                    wr.write("{")
                    if child_scope.writer.items:
                        wr.extend(child_scope.writer.items)
                    else:
                        flush(child_scope, wr)
                    wr.write("}")
                    formatted[child] = f"${ident}"
                else:
                    if not child_scope.writer.items:
                        flush(child_scope, wr)
                    wr.extend(child_scope.writer.items)
            else:
                wr.write(
                    "let",
                    f"${ident}",
                    "=",
                    ase.get_head(child),
                    *map(fmt, ase.get_args(child)),
                )
                formatted[child] = f"${ident}"
    assert None in grouped
    return scope.writer


def is_simple(expr):
    return not any(isinstance(x, ase.BaseExpr) for x in ase.get_args(expr))


class BetaReduction(TreeRewriter[ase.BaseExpr]):
    """A tree rewriter implementing beta-reduction logic"""

    def __init__(self, drops, repl):
        super().__init__()
        self._drops = drops
        self._repl = repl

    def rewrite_generic(
        self, old: ase.BaseExpr, args: tuple[Any, ...], updated: bool
    ) -> ase.BaseExpr:
        if ase.get_head(old) == "arg":
            # Replace argument
            return self._repl.get(ase.get_args(old)[0], old)
        elif old in self._drops:
            # Drop the lambda
            assert ase.get_head(old) in {"app", "lam"}
            match ase.get_head(old):
                case "app":
                    return _app(*args).body
                case "lam":
                    return args[0]

        return super().rewrite_generic(old, args, updated)
