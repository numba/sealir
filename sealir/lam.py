from __future__ import annotations

import inspect
import types
from collections import Counter, defaultdict
from typing import Any

from sealir import ase, grammar
from sealir.itertools import first
from sealir.rewriter import TreeRewriter


class _Value(grammar.Rule):
    pass


class Lam(_Value):
    body: ase.SExpr


class Arg(_Value):
    index: int


class App(_Value):
    arg: ase.SExpr
    lam: ase.SExpr


class Unpack(_Value):
    idx: int
    tup: ase.SExpr


class Pack(_Value):
    elts: tuple[ase.SExpr, ...]


class LamGrammar(grammar.Grammar):
    start = _Value


def _intercept_cells(grm, fn, arity):
    if fn.__closure__ is None:
        return fn
    changed = False
    new_closure = []
    for i, cell in enumerate(fn.__closure__):
        val = cell.cell_contents
        if isinstance(val, ase.SExpr):
            if val._head == "Arg":
                # increase de Lruijn index
                [idx] = val._args
                new_closure.append(types.CellType(grm.write(Arg(arity - 1 - idx + 1))))
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


def lam_func(grm: grammar.Grammar):
    """Decorator to help build lambda expressions from a Python function.
    Takes multiple arguments and uses De Bruijn index.
    """

    def wrap(fn):
        argspec = inspect.getfullargspec(fn)
        argnames = argspec.args
        assert not argspec.varargs
        assert not argspec.kwonlyargs

        # use De Bruijn index starting from 0
        # > lam (lam arg0 arg1)
        #         ^---|     |
        #    ^--------------|
        args = [grm.write(Arg(i)) for i in range(len(argnames))]
        # intercept cell variables
        fn = _intercept_cells(grm, fn, len(argnames))
        # Reverse arguments so that arg0 refers to the innermost lambda.
        # And, it is the rightmost argument in Python.
        # That way, first `(app )` will replace the leftmost Python argument
        expr = fn(*reversed(args))
        for _ in args:
            expr = grm.write(Lam(expr))
        return expr

    return wrap


def app_func(grm, lam, arg0, *more_args) -> ase.SExpr:
    """Makes an apply expression."""
    args = (arg0, *more_args)

    out = lam
    for arg in args:
        out = grm.write(App(lam=out, arg=arg))
    return out


def unpack(
    grm: grammar.Grammar, tup: ase.SExpr, nelem: int
) -> tuple[ase.SExpr, ...]:
    """Unpack a tuple with known size with `(Unpack )`"""
    return tuple(
        map(lambda i: grm.write(Unpack(tup=tup, idx=i)), range(nelem))
    )


def format(expr: ase.SExpr) -> str:
    """Multi-line formatting of the S-expression"""
    return format_lambda(expr).get()


def simplify(grm: grammar.Grammar) -> grammar.Grammar:
    """Make a copy and remove dead node. Last node is assumed to be root."""
    last = ase.BasicSExpr(grm._tape, grm._tape.last())
    new_tree = ase.Tape()
    ase.copy_tree_into(last, new_tree)
    return type(grm)(new_tree)


def beta_reduction(app_expr: ase.SExpr) -> ase.SExpr:
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
        if isinstance(child, Arg):
            lams = [x for x in parents if isinstance(x, Lam)]
            if len(lams) >= napps:  # don't go deeper
                debruijn = child.index
                # in range?
                if debruijn <= len(app_exprs) - 1:
                    arg2repl[child] = app_exprs[::-1][debruijn - 1].arg
                    drops.add(lams[:napps][-debruijn])

    assert arg2repl
    br = BetaReduction(drops, arg2repl)

    ase.apply_bottomup(target, br)
    out = br.memo[target]
    return out


def run_abstraction_pass(grm: grammar.Grammar, root_expr: ase.SExpr):
    """Convert expressions that would otherwise require a let-binding
    to use lambda-abstraction with an application.
    """

    while True:
        ctr: dict[ase.SExpr, int] = Counter()
        seen = set()

        class Occurrences(ase.TreeVisitor):
            # TODO: replace prettyprinter one with this
            def visit(self, expr: ase.SExpr):
                if expr not in seen:
                    seen.add(expr)
                    for arg in expr._args:
                        if isinstance(arg, ase.SExpr) and not ase.is_simple(
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
        def parent_containing_this(x: ase.SExpr):
            return x._head == "Lam"

        host_lam = first(ase.search_ancestors(expr, parent_containing_this))
        # find remaining expressions in the lambda
        old_node, repl_node = replace_by_abstraction(grm, host_lam, expr)
        repl[old_node] = repl_node

        # Rewrite the rest
        class RewriteProgram(TreeRewriter):
            def rewrite_generic(
                self,
                old: ase.SExpr,
                args: tuple[Any, ...],
                updated: bool,
            ) -> Any | ase.SExpr:
                if old in repl:
                    return repl[old]
                return super().rewrite_generic(old, args, updated)

        rewriter = RewriteProgram()
        ase.apply_bottomup(root_expr, rewriter)
        root_expr = rewriter.memo[root_expr]

    return root_expr


def replace_by_abstraction(
    grm: grammar.Grammar, lamexpr: ase.SExpr, anchor: ase.SExpr
) -> tuple[ase.SExpr, ase.SExpr]:
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
    new_node = rewrite_into_abstraction(grm, old_node, anchor, lam_depth_map)
    repl = grm.write(App(lam=grm.write(Lam(new_node)), arg=anchor))
    return old_node, repl


def rewrite_into_abstraction(
    grm: grammar.Grammar,
    root: ase.SExpr,
    anchor: ase.SExpr,
    lam_depth_map: dict[ase.SExpr, int],
) -> ase.SExpr:

    arg_index = lam_depth_map[anchor] - 1

    # replace remaining program as (app (lam body) expr)
    # in body, shift de bruijin index + 1 for those referring to
    # outer lambdas before introducing new (arg 0)
    class RewriteAddArg(grammar.TreeRewriter):

        def rewrite_Arg(self, orig: ase.SExpr, index: int):
            if orig not in lam_depth_map:
                return orig._replace(index)
            depth_offset = lam_depth_map[orig] - lam_depth_map[anchor]
            if index - depth_offset < arg_index:
                return orig._replace(index)
            else:
                return orig._replace(index + 1)

        def rewrite_generic(
            self, orig: ase.SExpr, args: tuple[Any, ...], updated: bool
        ) -> Any | ase.SExpr:
            if orig == anchor:
                return grm.write(Arg(arg_index))
            return super().rewrite_generic(orig, args, updated)

    rewrite = RewriteAddArg(grm)
    ase.apply_bottomup(root, rewrite)
    out_expr = rewrite.memo[root]
    return out_expr


def format_lambda(expr: ase.SExpr):
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

    def first_lam_parent(parents) -> ase.SExpr | None:
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

    grouped: defaultdict[ase.SExpr | None, LamScope]
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


class BetaReduction(grammar.TreeRewriter[ase.SExpr]):
    """A tree rewriter implementing beta-reduction logic"""

    def __init__(self, drops, repl):
        super().__init__()
        self._drops = drops
        self._repl = repl

    def rewrite_Arg(self, orig: ase.SExpr, index: int) -> ase.SExpr:
        return self._repl.get(orig, orig)

    def rewrite_App(
        self, orig: ase.SExpr, lam: ase.SExpr, **kwargs
    ) -> ase.SExpr:
        if orig in self._drops:
            return lam
        else:
            return self.passthru()

    def rewrite_Lam(self, orig: ase.SExpr, body: ase.SExpr) -> ase.SExpr:
        if orig in self._drops:
            return body
        else:
            return self.passthru()
