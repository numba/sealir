from __future__ import annotations

import inspect
import types
from collections import Counter, defaultdict
from typing import Annotated, Any, NamedTuple

from sealir import ase, grammar
from sealir.itertools import first
from sealir.rewriter import TreeRewriter


@grammar.rule
class _Value(grammar.Rule):
    pass


@grammar.rule
class Lam(_Value):
    body: ase.BaseExpr


@grammar.rule
class Expr(_Value):
    head: str
    args: tuple[ase.value_type, ...]


@grammar.rule
class Arg(_Value):
    index: int


@grammar.rule
class App(_Value):
    arg: ase.BaseExpr
    lam: ase.BaseExpr


class LamGrammar(grammar.Grammar):
    start = _Value


class LamBuilder:
    """Helper for building lambda calculus expression"""

    _tape: ase.Tape

    def __init__(self, tape: ase.Tape | None = None):
        self._tape = tape or ase.Tape()
        self._grm = LamGrammar(self._tape)

    @property
    def tape(self) -> ase.Tape:
        return self._tape

    def get_root(self) -> ase.BaseExpr:
        return ase.SimpleExpr(self._tape, self._tape.last())

    def expr(self, head: str, *args) -> ase.BaseExpr:
        """Makes a user defined expression"""
        with self._tape:
            return self._grm.write(Expr(head=head, args=args))

    def lam(self, body_expr: ase.BaseExpr) -> ase.BaseExpr:
        """Makes a lambda abstraction"""
        with self._tape:
            return self._grm.write(Lam(body=body_expr))

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
                if val._head == "Arg":
                    # increase de Lruijn index
                    [idx] = val._args
                    new_closure.append(types.CellType(self.arg(idx + 1)))
                    changed = True
                elif val._head != "Lam":
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
        with self._tape:
            return self._grm.write(Arg(index=index))

    def app(self, lam, arg0, *more_args) -> ase.BaseExpr:
        """Makes an apply expression."""
        args = (arg0, *more_args)
        with self._tape:
            stack = list(args)
            out = lam
            while stack:
                arg = stack.pop()
                out = self._grm.write(App(lam=out, arg=arg))
        return out

    def beta_reduction(self, app_expr: ase.BaseExpr) -> ase.BaseExpr:
        """Reduces a (App arg (Lam ...))"""
        assert app_expr._head == "App"

        target = app_expr

        app_exprs = []
        while isinstance(app_expr, App):
            app_exprs.append(app_expr)
            app_expr = app_expr.lam

        napps = len(app_exprs)

        arg2repl = {}
        drops = set(app_exprs)
        for parents, child in ase.walk_descendants(app_expr):
            if child._head == "Arg":
                lams = [x for x in parents if isinstance(x, Lam)]
                if len(lams) <= napps:  # don't go deeper
                    [debruijn] = child._args
                    # in range?
                    if isinstance(debruijn, int) and debruijn < len(lams):
                        arg2repl[debruijn] = app_exprs[-debruijn - 1].arg
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
                        for arg in expr._args:
                            if isinstance(
                                arg, ase.BaseExpr
                            ) and not ase.is_simple(arg):
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
                return x._head == "Lam"

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
        lam_depth_map[child] = len([p for p in parents if p._head == "Lam"])

    # rewrite these children nodes into a new lambda abstraction
    # replacing the `anchor` node with an arg node.
    [old_node] = lamexpr._args
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
        def rewrite_Arg(self, orig: ase.BaseExpr, index: int):
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
            if p._head == "Lam":
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
        while expr._head == "Lam":
            depth += 1
            expr = expr._args[0]
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
            if ase.is_simple(child) and parents:
                parts = [
                    f"{child._head}",
                    *map(fmt, child._args),
                ]
                formatted[child] = f"({" ".join(parts)})"
            elif child._head == "Lam":
                child_scope = grouped[child]
                child_scope.lambda_depth = compute_lam_depth(child)

                if not parents or parents[-1]._head != "Lam":
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
                    child._head,
                    *map(fmt, child._args),
                )
                formatted[child] = f"${ident}"
    assert None in grouped
    return scope.writer


class BetaReduction(TreeRewriter[ase.BaseExpr]):
    """A tree rewriter implementing beta-reduction logic"""

    def __init__(self, drops, repl):
        super().__init__()
        self._drops = drops
        self._repl = repl

    def rewrite_generic(
        self, old: ase.BaseExpr, args: tuple[Any, ...], updated: bool
    ) -> ase.BaseExpr:
        if isinstance(old, Arg):
            # Replace argument
            return self._repl.get(Arg(*args).index, old)
        elif old in self._drops:
            # Drop the lambda
            match old:
                case App():
                    return App(*args).lam
                case Lam():
                    return Lam(*args).body
                case _:
                    raise TypeError(f"must be App | Lam; got: {old}")

        return super().rewrite_generic(old, args, updated)
