"""
Array-based S-Expression
------------------------

For a immutable flat storage of S-expression and fast search in the
expression-tree.
S-expression are stored as integer in a append-only tape.
The number points to other S-expression nodes in the heap.
Tokens are stored in a side table with mapping from the token
to a negative integer index. Negative indices to differentiate from S-expression
position indices.

The design is to make it so that there are no need for recursive function
to process such a S-expression tree.
"""

from __future__ import annotations

import abc
import html
import sys
from collections import Counter, deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from pprint import pformat
from typing import (
    Any,
    Callable,
    Iterator,
    LiteralString,
    Self,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

from .graphviz_support import graphviz_function

T = TypeVar("T")


class MalformedContextError(RuntimeError):
    pass


class HeapOverflow(RuntimeError):
    pass


class NotFound(ValueError):
    pass


class HandleSentry(IntEnum):
    BEGIN = 0xFFFF_FFFE
    END = BEGIN + 1

    def __repr__(self):
        return f"<{self.name}>"


class Tape:
    """
    An append-only Tape for storing all S-expressions.
    """

    _heap: list[handle_type]
    """The main storage for the S-expressions.
    """
    _tokens: list[token_type]
    """Store tokens.
    """
    _tokenmap: dict[tuple[Type, token_type], int]
    """
    Auxiliary. Map tokens to their position in `_tokens`.
    """
    _num_records: int
    """
    Auxiliary. Number of records (S-expressions).
    """

    def __init__(self):
        # First item on the heap is the None token
        self._heap = [0]
        self._tokens = [None]
        self._tokenmap = {(type(None), None): 0}
        self._num_records = 0
        self._open_counter = 0
        self._downcast = lambda x: x

    def __len__(self) -> int:
        return self._num_records

    @property
    def heap_size(self) -> int:
        return len(self._heap)

    def __enter__(self):
        self._open_counter += 1
        return self

    def __exit__(self, exc_val, exc_typ, exc_tb):
        if self._open_counter <= 0:
            raise MalformedContextError("malformed stack: top is not self")
        self._open_counter -= 1

    def iter_expr(self) -> Iterator[SExpr]:
        crawler = TapeCrawler(self, self._downcast)
        crawler.move_to_first_record()
        for rec in crawler.walk():
            yield rec.to_expr()

    # Write API

    def expr(self, head: str, *args: value_type) -> SExpr:
        """The main API for creating an `Expr`."""
        return BasicSExpr._write(self, head, args)

    # Debug API

    def dump_raw(self) -> str:
        buf = []
        buf.append("\n")
        buf.append("Heap:\n")
        for i, h in enumerate(self._heap):
            buf.append(f"{i:6} | {repr(h)}\n")
        buf.append("\n")
        buf.append("Tokens:\n")
        buf.append(pformat(self._tokens))
        return "".join(buf)

    def dump(self) -> str:
        buf = []
        buf.append("\n")
        buf.append(f"Tape @{hex(id(self))}\n")

        crawler = TapeCrawler(self, self._downcast)
        buf.append(f"| {crawler.pos:6} | <{repr(crawler.get())}: none>\n")
        crawler.step()

        def fixup(x):
            if isinstance(x, SExpr):
                return f"<{x._handle}>"
            else:
                return repr(x)

        for rec in crawler.walk():
            args = ", ".join(fixup(x) for x in rec.read_args())
            buf.append(f"| {rec.handle:6} | {rec.read_head()} {args}\n")
        buf.append("\n")
        return "".join(buf)

    @graphviz_function
    def render_dot(self, *, gv, show_metadata: bool = False,
                   only_reachable: bool = False):
        def make_label(i, x):
            if isinstance(x, SExpr):
                return f"<{i}> [{x._handle}]"
            else:
                return html.escape(f"{x!r} :{type(x).__name__}")

        g = gv.Digraph(node_attr={"shape": "record"})

        crawler = TapeCrawler(self, self._downcast)

        # Records that are children of the last node
        crawler.seek(self.last())

        # Seek to the first non-metadata in the back
        while crawler.pos > 0:
            # Search for the first non-metadata from the back
            rec = crawler.read_surrounding_record()
            if rec.read_head().startswith(metadata_prefix):
                crawler.move_to_previous_record()
            else:
                break

        reachable = set()
        for _, child_rec in crawler.walk_descendants():
            reachable.add(child_rec.handle)

        crawler.move_to_first_record()
        edges = []

        g.node(
            "start",
            label="start",
            rank="min",
            shape="doublecircle",
            root="true",
        )
        lastname = None
        maxrank = len(self._heap)
        for record in crawler.walk():
            idx = record.handle
            head = record.read_head()
            args = record.read_args()
            if only_reachable and idx not in reachable:
                continue # skip
            if not show_metadata and head.startswith(metadata_prefix):
                # Skip metadata
                continue
            label = "|".join(make_label(i, v) for i, v in enumerate(args))
            node_kwargs = {}
            edge_kwargs = {}
            if idx not in reachable:
                node_kwargs["color"] = "lightgrey"
                edge_kwargs["color"] = "lightgrey"
                edge_kwargs["weight"] = "1"
            else:
                edge_kwargs["weight"] = "5"
            nodename = f"node{idx}"
            g.node(
                nodename,
                label=f"[{idx}] {head}|{label}",
                rank=str(maxrank - record.handle + 1),
                **node_kwargs,
            )
            for i, arg in enumerate(args):
                if isinstance(arg, SExpr):
                    args = (f"{nodename}:{i}", f"node{arg._handle}")
                    kwargs = {**edge_kwargs}
                    edges.append((args, kwargs))
            if not head.startswith(metadata_prefix):
                lastname = nodename
        # emit start
        if lastname:
            edges.append((("start", lastname), {}))
        # emit edges
        for args, kwargs in edges:
            g.edge(*args, **kwargs)

        return g

    # Raw Access API

    def get(self, pos: handle_type) -> handle_type:
        assert pos >= 0
        return self._heap[pos]

    def load(
        self, start: handle_type, stop: handle_type
    ) -> tuple[handle_type, ...]:
        return tuple(self._heap[start:stop])

    def last(self) -> handle_type:
        return self.rindex(HandleSentry.BEGIN, len(self._heap) - 1)

    # Search API

    def index(self, target: handle_type, startpos: handle_type) -> handle_type:
        try:
            return self._heap.index(target, startpos)
        except ValueError:
            raise NotFound

    def rindex(
        self, target: handle_type, startpos: handle_type
    ) -> handle_type:
        pos = startpos
        heap = self._heap
        while pos >= 0 and heap[pos] != target:
            pos -= 1
        if pos < 0:
            raise NotFound
        return pos

    # Read API

    def read_head(self, handle: handle_type) -> str:
        out = self._read_token(self._heap[handle + 1])
        assert isinstance(out, str)
        return out

    def read_args(self, handle: handle_type) -> tuple[value_type, ...]:
        offset = handle + 2
        buf = []
        while True:
            i = self._heap[offset]
            if i < HandleSentry.END:
                buf.append(self.read_value(i))
            elif i == HandleSentry.END:
                break
            else:
                raise HeapOverflow
            offset += 1
        return tuple(buf)

    def read_value(self, handle: handle_type) -> value_type:
        if handle <= 0:
            return self._read_token(handle)
        elif handle < HandleSentry.BEGIN:
            return BasicSExpr(self, handle)
        else:
            raise MalformedContextError

    def _read_token(self, token_index: handle_type) -> token_type:
        assert token_index <= 0
        return self._tokens[-token_index]

    # Write API

    def write(self, head: str, args: tuple[value_type, ...]) -> handle_type:
        handle = self.write_begin()
        self.write_token(head)
        for a in args:
            if isinstance(a, SExpr):
                if a._tape is not self:
                    raise ValueError(
                        f"invalid to assign Expr({repr(a)}) to a different tape"
                    )
                self._heap.append(a._handle)
            else:
                self.write_token(a)
        self.write_end()
        return handle

    def write_ref(self, ref: handle_type):
        assert 0 < ref < HandleSentry.BEGIN
        self._heap.append(ref)

    def write_token(self, token: token_type) -> None:
        if token is not None and not isinstance(token, (int, str, float)):
            raise TypeError(f"invalid token type for {type(token)}")
        last = -len(self._tokens)
        handle = self._tokenmap.get((type(token), token), last)
        if handle == last:
            self._tokens.append(token)
            self._tokenmap[type(token), token] = handle
        self._heap.append(handle)

    def write_begin(self) -> handle_type:
        self._guard()
        handle = len(self._heap)
        self._heap.append(HandleSentry.BEGIN)
        return handle

    def write_end(self) -> None:
        self._guard()
        self._heap.append(HandleSentry.END)
        self._num_records += 1

    def _guard(self) -> None:
        n = len(self._heap)
        if n >= HandleSentry.BEGIN:
            raise HeapOverflow


@dataclass(frozen=True, kw_only=True)
class TraverseState:
    parents: list[SExpr] = field(default_factory=list)


class SExpr(abc.ABC):
    _tape: Tape
    _handle: handle_type

    @property
    @abc.abstractmethod
    def _head(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def _args(self) -> tuple[value_type, ...]:
        pass

    @abc.abstractmethod
    def _get_downcast(self) -> Callable[[SExpr], SExpr]:
        pass

    @classmethod
    @abc.abstractmethod
    def _wrap(cls, tape: Tape, handle: handle_type) -> Self:
        pass

    @classmethod
    def _write(
        cls, tape: Tape, head: str, args: tuple[value_type, ...]
    ) -> SExpr:
        handle = tape.write(head, args)
        return cls._wrap(tape, handle)

    def _replace(self, *args: value_type) -> Self:
        if args == self._args:
            return self
        else:
            return self._write(self._tape, self._head, args)

    # Comparison API

    def __lt__(self, other) -> bool:
        if isinstance(other, SExpr):
            return self._handle < other._handle
        else:
            return NotImplemented

    def __eq__(self, value) -> bool:
        if isinstance(value, SExpr):
            return self._tape == value._tape and self._handle == value._handle
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((id(self._tape), self._handle))

    def __str__(self):
        head = self._head
        args = self._args
        return (
            f"{self.__class__.__name__}({head}, {', '.join(map(repr, args))})"
        )


metadata_prefix = "."


def pretty_str(expr: SExpr) -> str:
    from .prettyprinter import pretty_print

    return pretty_print(expr)


def is_metadata(sexpr: SExpr) -> bool:
    return sexpr._head.startswith(metadata_prefix)


def is_simple(sexpr: SExpr) -> bool:
    """
    Checks if the expression is a simple expression,
    where all arguments are not `Expr` objects.
    """
    return all(not isinstance(a, SExpr) for a in sexpr._args)


# Search API


def walk_parents(self: SExpr) -> Iterator[SExpr]:
    """A Iterator that yields Expr that immediately contains this
    object.
    Returned values follow the order of occurrence.
    """
    crawler = TapeCrawler(self._tape, self._get_downcast())
    crawler.seek(self._handle)
    crawler.skip_to_record_end()
    while crawler.move_to_pos_of(self._handle):
        yield crawler.read_surrounding_record().to_expr()
        crawler.skip_to_record_end()


def search_parents(
    self: SExpr, pred: Callable[[SExpr], bool]
) -> Iterator[SExpr]:
    for p in walk_parents(self):
        if pred(p):
            yield p


def search_ancestors(
    self: SExpr, pred: Callable[[SExpr], bool]
) -> Iterator[SExpr]:
    """
    Recursive breath-first search for parent expressions that match the
    given predicate.

    NOTE: Implementation is recursion-free.

    This method performs a breath-first search starting from the current
    expression, following the parent links up the expression tree. It yields
    all parent expressions that satisfy the given predicate function.
    """
    candidates = deque(walk_parents(self))
    seen = set(candidates)
    while candidates:
        p = candidates.popleft()
        if pred(p):
            yield p
        for new_p in walk_parents(p):
            if new_p not in seen:
                candidates.append(new_p)
                seen.add(new_p)


def walk_descendants(
    self: SExpr,
) -> Iterator[tuple[tuple[SExpr, ...], SExpr]]:
    """Walk descendants of this Expr node.
    Breath-first order. Left to right.
    """
    oldtree = self._tape
    crawler = TapeCrawler(oldtree, self._get_downcast())
    crawler.seek(self._handle)
    for parents, desc in crawler.walk_descendants():
        parent_exprs = tuple(map(lambda x: x.to_expr(), parents))
        yield parent_exprs, desc.to_expr()


def walk_descendants_depth_first_no_repeat(
    self: SExpr,
) -> Iterator[tuple[tuple[SExpr, ...], SExpr]]:
    """Walk descendants of this Expr node.
    Depth-first order. Left to right. Avoid duplication.
    """
    stack = [(self, ())]
    visited = set()
    while stack:
        node, parents = stack.pop()
        if node not in visited:
            visited.add(node)
            yield parents, node
            for arg in reversed(node._args):
                if isinstance(arg, SExpr):
                    stack.append((arg, (*parents, node)))


def walk_descendants_depth_first(
    self: SExpr,
) -> Iterator[tuple[tuple[SExpr, ...], SExpr]]:
    """Walk descendants of this Expr node.
    Depth-first order. Left to right.
    """
    stack = [(self, ())]
    while stack:
        node, parents = stack.pop()
        yield parents, node
        for arg in reversed(node._args):
            if isinstance(arg, SExpr):
                stack.append((arg, (*parents, node)))


def search_descendants(
    self: SExpr, pred: Callable[[SExpr], bool]
) -> Iterator[tuple[tuple[SExpr, ...], SExpr]]:
    for parents, cur in walk_descendants(self):
        if pred(cur):
            yield parents, cur


def traverse(
    self: SExpr,
    corofunc: Callable[[SExpr, TraverseState], Coroutine[SExpr, T, T]],
    state: TraverseState | None = None,
) -> dict[SExpr, T]:
    """Traverses the expression tree rooted at the current node, applying
    the provided coroutine function to each node in a depth-first order.
    The traversal is memoized, so that if a node is encountered more than
    once, the result from the first visit is reused. The function returns
    a dictionary mapping each visited node to the value returned by the
    coroutine function for that node.
    """
    stack: list[tuple[Coroutine[SExpr, T, T], SExpr, SExpr]]
    stack = []
    memo = {}
    cur_node = self
    state = state or TraverseState()
    coro = corofunc(cur_node, state)

    def handle_coro(to_send):
        try:
            item = coro.send(to_send)
        except StopIteration as stopped:
            memo[cur_node] = stopped.value
            if stack:
                state.parents.pop()
                return True, stack.pop()
            else:
                return False, (None, None, None)
        else:
            return True, (coro, cur_node, item)

    status, (coro, cur_node, item) = handle_coro(None)
    while status:
        # Loop starts with a fresh item
        if item in memo:
            # Already processed?
            status, (coro, cur_node, item) = handle_coro(memo[item])
        else:
            # Not yet seen?
            stack.append((coro, cur_node, item))
            state.parents.append(cur_node)
            coro, cur_node = corofunc(item, state), item
            status, (coro, cur_node, item) = handle_coro(None)

    return memo


def reachable_set(self: SExpr) -> set[SExpr]:
    return {node for _, node in walk_descendants_depth_first_no_repeat(self)}


def contains(self: SExpr, test: SExpr) -> bool:
    """Is `test` part of this expression tree."""
    for _, child in walk_descendants(self):
        if child == test:
            return True
    return False


# Apply API


def apply_bottomup(
    self: SExpr,
    visitor: TreeVisitor,
    *,
    reachable: set[SExpr] | None | LiteralString = "compute",
) -> None:
    """
    Apply the TreeVisitor to every sexpr bottom up. When a sexpr is visited,
    its children must have been visited prior.

    Args:
        visitor:
            The visitor to apply to each expression.
        reachable (optional):
            The set of reachable expressions.
            If "compute", the reachable set will be computed.
            Defaults to "compute".
    """
    # normalize reachable
    match reachable:
        case "compute":
            reachable = reachable_set(self)
        case str():
            raise ValueError(f"invalid input for `reachable`: {reachable!r}")

    crawler = TapeCrawler(self._tape, self._get_downcast())
    match reachable:
        case set():
            first_reachable = min(reachable)
            crawler.seek(first_reachable._handle)
        case None:
            crawler.move_to_first_record()
        case _:
            raise AssertionError("unreachable")
    for rec in crawler.walk():
        if rec.handle <= self._handle:  # visit all younger
            ex = rec.to_expr()
            if not reachable or ex in reachable:
                if not is_metadata(ex):
                    visitor.visit(ex)


def apply_topdown(self: SExpr, visitor: TreeVisitor) -> None:
    """
    Apply the TreeVisitor to every sexpr under `self` subtree.
    """
    for _, node in walk_descendants_depth_first_no_repeat(self):
        if not is_metadata(node):
            visitor.visit(node)


# Conversion API


def as_tuple(self: SExpr, depth: int = 1, dedup=False) -> tuple[Any, ...]:
    """
    Converts an Expr object to a tuple, with a specified depth limit.

    Args:
        depth (int): The maximum depth to traverse the Expr object.
                        Set to -1 for unbounded.
    """
    # nothing is actually going to be sys.maxsize big,
    # so that's what we meant by unbounded.
    depth = sys.maxsize if depth == -1 else depth

    pending = []

    occurrences: Counter[SExpr] = Counter()
    for parents, cur in walk_descendants_depth_first_no_repeat(self):
        if len(parents) >= depth:
            break
        else:
            pending.append(cur)

    class CountOccurrences(TreeVisitor):
        def visit(self, node: SExpr):
            if not is_metadata(node):
                occurrences.update([cur])
                occurrences.update(
                    (x for x in node._args if isinstance(x, SExpr))
                )

    apply_bottomup(self, CountOccurrences())

    dupset = {x for x, ct in occurrences.items() if ct > 1}
    working_set = set(pending)

    def multi_parents(expr: SExpr) -> bool:
        it = search_parents(expr, lambda x: x in working_set)
        try:
            next(it)
            next(it)
        except StopIteration:
            return False
        else:
            return True

    memo: dict[SExpr, tuple] = {}

    for cur in reversed(pending):
        args = []
        for arg in cur._args:
            if dedup and arg in dupset and occurrences[arg] > 1:
                val = f"${arg._handle}"
            else:
                val = memo.get(arg, arg)
            occurrences[arg] -= 1
            args.append(val)
        if dedup and cur in dupset and multi_parents(cur):
            out = (f"[${cur._handle}]", cur._head, *args)
        else:
            out = (cur._head, *args)
        memo[cur] = out

    return memo[self]


def as_dict(self) -> dict[str, dict]:
    """
    Converts the expression tree into a dictionary representation.

    The `as_dict` method takes an Expr object and returns a dictionary
    representation of the object. It uses a memoization technique to avoid
    duplicate references.

    The method also handles simple expressions (where all arguments are not
    Expr objects) differently, by directly including the argument values in
    the dictionary.
    """
    memo: dict[SExpr, dict[str, dict]] = {}
    seen_once = set()

    class AsDict(TreeVisitor):
        def visit(self, expr: SExpr) -> None:
            parts: list[dict | token_type] = []
            for arg in expr._args:
                match arg:
                    case SExpr():
                        if arg in seen_once:
                            # This handles duplicated references
                            parts.append(
                                dict(ref=f"{arg._head}-{arg._handle}")
                            )
                        else:
                            ref = memo[arg]
                            if not is_simple(arg):
                                seen_once.add(arg)
                            parts.append(ref)
                    case _:
                        parts.append(arg)
            k = f"{expr._head}-{expr._handle}"
            memo[expr] = {k: dict(head=expr._head, args=tuple(parts))}

    self.apply_bottomup(AsDict())
    return memo[self]


# Copy API


def copy_tree_into(self: SExpr, tape: Tape) -> SExpr:
    """Copy all the expression tree starting with this node into the given
    tape.
    Returns a fresh Expr in the new tape.
    """
    oldtree = self._tape
    crawler = TapeCrawler(oldtree, self._get_downcast())
    crawler.seek(self._handle)
    liveset = set(_select(crawler.walk_descendants(), 1))
    surviving = sorted(liveset)
    mapping = {}
    for oldrec in surviving:
        head = oldtree.read_head(oldrec.handle)
        args = oldtree.read_args(oldrec.handle)

        mapping[oldrec.handle] = tape.write_begin()
        tape.write_token(head)

        for arg in args:
            if isinstance(arg, SExpr):
                tape.write_ref(mapping[arg._handle])
            else:
                tape.write_token(arg)

        tape.write_end()

    out = tape.read_value(mapping[self._handle])
    assert isinstance(out, SExpr)
    return out


class BasicSExpr(SExpr):
    """S-expression reference

    Default comparison is by identity (the handle).
    """

    _tape: Tape
    _handle: handle_type
    __match_args__ = "_head", "_args"

    @classmethod
    def _wrap(cls, tape: Tape, handle: handle_type) -> Self:
        return cls(tape, handle)

    def __init__(self, tape: Tape, handle: handle_type) -> None:
        self._tape = tape
        self._handle = handle

    @cached_property
    def _head(self) -> str:
        tape = self._tape
        return tape.read_head(self._handle)

    @cached_property
    def _args(self) -> tuple[value_type, ...]:
        tape = self._tape
        return tape.read_args(self._handle)

    def __repr__(self):
        start = f"<Expr {self._head!r} [{self._handle}]"
        end = f" @{hex(id(self._tape))}>"
        return start + end

    def _get_downcast(self) -> Callable[[SExpr], BasicSExpr]:
        def downcast(expr):
            if isinstance(expr, BasicSExpr):
                return expr
            else:
                return BasicSExpr(expr._tape, expr._handle)

        return downcast


class TreeVisitor:
    def visit(self, expr: SExpr):
        pass


class TapeCrawler:
    """Provides a file-like API to read the `Tape`."""

    _tape: Tape
    _pos: handle_type

    def __init__(self, tape: Tape, downcast: Callable[[SExpr], SExpr]) -> None:
        self._tape = tape
        self._pos = 0
        self._downcast = downcast

    def move_to_first_record(self) -> None:
        self._pos = 1

    @property
    def pos(self) -> handle_type:
        return self._pos

    def get(self) -> handle_type:
        return self._tape.get(self._pos)

    def seek(self, pos: handle_type) -> None:
        start_handle = self._tape.get(pos)
        assert start_handle == HandleSentry.BEGIN
        self._pos = pos

    def step(self) -> None:
        self._pos += 1

    def skip_to_record_end(self):
        self._pos = self._tape.index(HandleSentry.END, self._pos + 1)

    def walk(self) -> Iterator[Record]:
        while self._pos < self._tape.heap_size:
            yield self.read_record()

    def walk_descendants(self) -> Iterator[tuple[tuple[Record, ...], Record]]:
        """Walk all descendants starting from the current position.
        Breath-first order.

        Yields (parents, record)
        """
        rec = self.read_record()
        todos: deque[tuple[tuple[Record, ...], Record]]
        todos = deque([((), rec)])
        while todos:
            parents, rec = todos.popleft()
            yield parents, rec
            for child in rec.children():
                todos.append(((*parents, rec), child))

    def read_record(self) -> Record:
        assert self._tape.get(self._pos) == HandleSentry.BEGIN
        begin = self._pos
        end = self._tape.index(HandleSentry.END, begin)
        rec = Record(self._tape, begin, end, self._downcast)
        self._pos = end + 1
        return rec

    def read_surrounding_record(self) -> Record:
        start = self._tape.rindex(HandleSentry.BEGIN, self._pos)
        stop = self._tape.index(HandleSentry.END, self._pos)
        return Record(self._tape, start, stop, self._downcast)

    def move_to_pos_of(self, target: handle_type) -> bool:
        try:
            self._pos = self._tape.index(target, self._pos)
        except NotFound:
            return False
        else:
            return True

    def move_to_previous_record(self, startpos=None) -> None:
        startpos = startpos or self._pos
        # Move to start of current
        self._pos = self._tape.rindex(HandleSentry.BEGIN, startpos)
        # Move to start of previous
        self._pos = self._tape.rindex(HandleSentry.BEGIN, self._pos - 1)


@dataclass(frozen=True, order=True)
class Record:
    tape: Tape
    handle: handle_type
    end_handle: handle_type
    downcast: Callable[[SExpr], SExpr]

    def children(self) -> Iterator[Record]:
        """Return child records. Cannot be tokens."""
        body = self.tape.load(self.handle + 1, self.end_handle)
        for h in body:
            if h > 0:  # don't include tokens
                end = self.tape.index(HandleSentry.END, h)
                yield type(self)(self.tape, h, end, self.downcast)

    def read_head(self):
        return self.tape.read_head(self.handle)

    def read_args(self):
        return self.tape.read_args(self.handle)

    def to_expr(self) -> SExpr:
        base = BasicSExpr(self.tape, self.handle)
        if is_metadata(base):
            return base
        return self.downcast(base)

    def __repr__(self):
        return f"<Record {self.handle}:{self.end_handle} tape@{hex(id(self.tape))} >"


def _select(iterable, idx: int):
    for args in iterable:
        yield args[idx]


token_type: TypeAlias = Union[int, float, str, None]
value_type: TypeAlias = Union[token_type, SExpr]
handle_type: TypeAlias = int
