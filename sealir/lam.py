from __future__ import annotations

from typing import Any
import types
import inspect
from collections import defaultdict, Counter

from sealir import ase
from sealir.rewriter import TreeRewriter
from sealir.itertools import first


class LamBuilder:
    """Helper for building lambda calculus expression"""

    _tree: ase.Tree

    def __init__(self, tree: ase.Tree | None = None):
        self._tree = tree or ase.Tree()

    def expr(self, head: str, *args) -> ase.Expr:
        """Makes a user defined expression"""
        with self._tree:
            return ase.expr("expr", head, *args)

    def lam(self, body_expr: ase.Expr) -> ase.Expr:
        """Makes a lambda abstraction"""
        with self._tree:
            return ase.expr("lam", body_expr)

    def lam_func(self, fn):
        """Decorator to help build lambda expressions from a Python function.
        Takes multiple arguments and uses De Bruijn index.
        """
        argspec = inspect.getfullargspec(fn)
        argnames = argspec.args
        assert not argspec.varargs
        assert not argspec.kwonlyargs

        with self._tree:
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

    def unpack(self, tup: ase.Expr, nelem: int) -> tuple[ase.Expr, ...]:
        """Unpack a tuple with known size with `(tuple.getitem )`
        """
        return tuple(map(lambda i: self.expr("tuple.getitem", tup, i),
                         range(nelem)))

    def _intercept_cells(self, fn):
        if fn.__closure__ is None:
            return fn
        changed = False
        new_closure = []
        for i, cell in enumerate(fn.__closure__):
            val = cell.cell_contents
            if isinstance(val, ase.Expr):
                if val.head == "arg":
                    # increase de bruijn index
                    [idx] = val.args
                    new_closure.append(types.CellType(self.arg(idx + 1)))
                    changed = True
                elif val.head != "lam":
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

    def arg(self, index: int) -> ase.Expr:
        return ase.expr("arg", index)

    def app(self, lam, arg0, *more_args) -> ase.Expr:
        """Makes an apply expression."""
        args = (arg0, *more_args)
        with self._tree:
            stack = list(args)
            out = lam
            while stack:
                arg = stack.pop()
                out = ase.expr("app", out, arg)
        return out

    def beta_reduction(self, app_expr: ase.Expr) -> ase.Expr:
        """Reduces a (app (lam ...) arg)"""
        assert app_expr.head == "app"

        target = app_expr

        app_exprs = []
        while app_expr.head == "app":
            app_exprs.append(app_expr)
            assert len(app_expr.args) == 2
            assert isinstance(app_expr.args[0], ase.Expr)
            app_expr = app_expr.args[0]
        napps = len(app_exprs)

        arg2repl = {}
        drops = set(app_exprs)
        for parents, child in app_expr.walk_descendants():
            if child.head == "arg":
                lams = [x for x in parents if x.head == "lam"]
                if len(lams) <= napps:  # don't go deeper
                    debruijn = child.args[0]
                    # in range?
                    if isinstance(debruijn, int) and debruijn < len(lams):
                        arg2repl[debruijn] = app_exprs[-debruijn - 1].args[1]
                        drops.add(lams[-debruijn - 1])

        assert arg2repl
        br = BetaReduction(drops, arg2repl)
        target.apply_bottomup(br)
        out = br.memo[target]
        return out

    def format(self, expr: ase.Expr) -> str:
        """Multi-line formatting of the S-expression"""
        return format_lambda(expr).get()

    def run_abstraction_pass(self, root_expr: ase.Expr):
        """Convert expressions that would otherwise require a let-binding
        to use lambda-abstraction with an application.
        """

        while True:
            ctr: dict[ase.Expr, int] = Counter()
            seen = set()

            class Occurrences(ase.TreeVisitor):
                # TODO: replace prettyprinter one with this
                def visit(self, expr: ase.Expr):
                    if expr not in seen:
                        seen.add(expr)
                        for arg in expr.args:
                            if isinstance(arg, ase.Expr) and not is_simple(arg):
                                ctr.update([arg])

            root_expr.apply_topdown(Occurrences())

            multi_occurring = [
                expr for expr, freq in reversed(ctr.items()) if freq > 1
            ]
            if not multi_occurring:
                break

            repl = {}
            # Handle the oldest (outermost) node first
            expr = min(multi_occurring)

            # find parent lambda
            def parent_containing_this(x: ase.Expr):
                return x.head == "lam" and x.contains(expr)

            host_lam = first(expr.search_parents(parent_containing_this))
            # find remaining expressions in the lambda
            old_node, repl_node = replace_by_abstraction(self, host_lam, expr)
            repl[old_node] = repl_node

            # Rewrite the rest
            class RewriteProgram(TreeRewriter):
                def rewrite_generic(
                    self, old: ase.Expr, args: tuple[Any, ...], updated: bool
                ) -> Any | ase.Expr:
                    if old in repl:
                        return repl[old]
                    return super().rewrite_generic(old, args, updated)

            rewriter = RewriteProgram()
            root_expr.apply_bottomup(rewriter)
            root_expr = rewriter.memo[root_expr]

        return root_expr


def replace_by_abstraction(
    lambar: LamBuilder, lamexpr: ase.Expr, anchor: ase.Expr
) -> tuple[ase.Expr, ase.Expr]:
    """
    Returns a `( old_node, (app (lam rewritten_old_node) anchor) )` tuple.
    """
    # Find lambda depth of children in the lambda node,
    # where the children are younger than the anchor node
    lam_depth = 0
    for parents, child in lamexpr.walk_descendants():
        if parents and child > anchor:
            lam_depth = max(
                lam_depth, len([p for p in parents if p.head == "lam"])
            )

    # rewrite these children nodes into a new lambda abstraction
    # replacing the `anchor` node with an arg node.
    [old_node] = lamexpr.args
    new_node = rewrite_into_abstraction(
        lambar, old_node, anchor, lam_depth - 1
    )
    repl = lambar.app(lambar.lam(new_node), anchor)
    return old_node, repl


def rewrite_into_abstraction(
    lambar: LamBuilder,
    root: ase.Expr,
    anchor: ase.Expr,
    arg_index: int,
) -> ase.Expr:
    # replace remaining program as (app (lam body) expr)
    # in body, shift de bruijin index + 1 for those referring to
    # outer lambdas before introducing new (arg 0)
    class RewriteAddArg(TreeRewriter):
        def rewrite_arg(self, index: int):
            if index < arg_index:
                return self.PassThru
            else:
                return lambar.arg(index + 1)

        def rewrite_generic(
            self, old: ase.Expr, args: tuple[Any, ...], updated: bool
        ) -> Any | ase.Expr:
            if old == anchor:
                return lambar.arg(arg_index)
            return super().rewrite_generic(old, args, updated)

    rewrite = RewriteAddArg()
    root.apply_bottomup(rewrite)
    out_expr = rewrite.memo[root]
    return out_expr


def format_lambda(expr: ase.Expr):
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

    descendants = list(expr.walk_descendants())

    def first_lam_parent(parents) -> ase.Expr | None:
        for p in reversed(parents):
            if p.head == "lam":
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
        while expr.head == "lam":
            depth += 1
            expr = expr.args[0]
        return depth

    grouped: defaultdict[ase.Expr | None, LamScope]
    grouped = defaultdict(LamScope)

    for ident, (parents, child) in enumerate(reversed(descendants)):
        lam_parent = first_lam_parent(parents)
        scope = grouped[lam_parent]
        formatted, wr = scope.formatted, scope.writer
        if child not in formatted:
            if is_simple(child) and parents:
                parts = [f"{child.head}", *map(fmt, child.args)]
                formatted[child] = f"({" ".join(parts)})"
            elif child.head == "lam":
                child_scope = grouped[child]
                child_scope.lambda_depth = compute_lam_depth(child)

                if not parents or parents[-1].head != "lam":
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
                        # empty child?
                        assert len(child_scope.formatted) == 1
                        [out] = child_scope.formatted.values()
                        wr.write(out)
                    wr.write("}")
                    formatted[child] = f"${ident}"
                else:
                    wr.extend(child_scope.writer.items)
            else:
                wr.write(
                    "let", f"${ident}", "=", child.head, *map(fmt, child.args)
                )
                formatted[child] = f"${ident}"
    assert None in grouped
    return scope.writer


def is_simple(expr):
    return not any(isinstance(x, ase.Expr) for x in expr.args)


class BetaReduction(TreeRewriter[ase.Expr]):
    """A tree rewriter implementing beta-reduction logic"""

    def __init__(self, drops, repl):
        super().__init__()
        self._drops = drops
        self._repl = repl

    def rewrite_generic(
        self, old: ase.Expr, args: tuple[Any, ...], updated: bool
    ) -> ase.Expr:
        if old.head == "arg":
            # Replace argument
            return self._repl.get(old.args[0], old)
        elif old in self._drops:
            # Drop the lambda
            assert old.head in {"app", "lam"}
            return args[0]
        return super().rewrite_generic(old, args, updated)
