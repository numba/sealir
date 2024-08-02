"""
Array-based S-Expression
------------------------

For a immutable flat storage of S-expression and fast search in the tree.
S-expression are stored as integer in a "heap". The number points to other
S-expression nodes in the heap.
Tokens are stored in a side table (dict) with mapping from the token
to a negative integer index. Negative to differentiate from S-expression
position indices.

The design is to make it so that there are no need for recursive function
to process such a S-expression tree.

"""

from __future__ import annotations
import threading
from typing import TypeAlias, Union, Iterator, Callable

from dataclasses import dataclass
from enum import IntEnum
from pprint import pformat
from functools import cached_property
from collections import deque
import html

from .graphviz_support import graphviz_function

class Context:
    _tls_context = threading.local()

    @classmethod
    def get_stack(cls) -> list[Tree]:
        try:
            return cls._tls_context.stack
        except AttributeError:
            stk = cls._tls_context.stack = []
            return stk

    @classmethod
    def push(cls, tree: Tree):
        cls.get_stack().append(tree)

    @classmethod
    def pop(cls) -> Tree:
        return cls.get_stack().pop()

    @classmethod
    def top(cls) -> Tree:
        out = cls.top_or_none()
        if out is None:
            raise MalformedContextError("no active Tree")
        else:
            return out

    @classmethod
    def top_or_none(cls) -> Tree | None:
        stk = cls.get_stack()
        if not stk:
            return None
        return stk[-1]


class MalformedContextError(RuntimeError):
    pass


class HeapOverflow(RuntimeError):
    pass


class NotFound(ValueError):
    pass


token_type: TypeAlias = Union[int, float, str, None]
value_type: TypeAlias = Union[token_type, "Expr"]
handle_type: TypeAlias = int


class HandleSentry(IntEnum):
    BEGIN = 0xFFFF_FFFE
    END = BEGIN + 1

    def __repr__(self):
        return f"<{self.name}>"


class Tree:
    _heap: list[handle_type]
    _tokens: list[token_type]

    def __init__(self):
        # First item on the heap is the None token
        self._heap = [0]
        self._tokens = [None]
        self._tokenmap = {None: 0}
        self._num_records = 0

    def __len__(self) -> int:
        return self._num_records

    @property
    def heap_size(self) -> int:
        return len(self._heap)

    def __enter__(self):
        Context.push(self)
        return self

    def __exit__(self, exc_val, exc_typ, exc_tb):
        out = Context.pop()
        if out is not self:
            raise MalformedContextError("malformed stack: top is not self")

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
        buf.append(f"Tree @{hex(id(self))}\n")

        crawler = TreeCrawler(self)
        buf.append(f"| {crawler.pos:6} | <{repr(crawler.get())}: bottom>\n")
        crawler.step()

        def fixup(x):
            if isinstance(x, Expr):
                return f"<{x._handle}>"
            else:
                return repr(x)

        for rec in crawler.walk():
            args = ", ".join(fixup(x) for x in rec.read_args())
            buf.append(f"| {rec.handle:6} | {rec.read_head()} {args}\n")
        buf.append("\n")
        return "".join(buf)

    @graphviz_function
    def render_dot(self, *, gv):
        def make_label(i, x):
            if isinstance(x, Expr):
                return f"<{i}> [{x._handle}]"
            else:
                return html.escape(f"{x!r} :{type(x).__name__}")
        g = gv.Digraph(node_attr={'shape': 'record'})

        crawler = TreeCrawler(self)

        # Records that are children of the last node
        crawler.seek(self.last())
        reachable = set()
        for _, child_rec in crawler.walk_descendants():
            reachable.add(child_rec.handle)

        crawler.move_to_first_record()
        edges = []
        for record in crawler.walk():
            idx = record.handle
            head = record.read_head()
            args = record.read_args()
            label = '|'.join(make_label(i, v) for i, v in enumerate(args))
            kwargs = {}
            if idx in reachable:
                kwargs['color'] = 'red'
            g.node(f"node{idx}", label=f"[{idx}] {head}|{label}",
                   rank=str(record.handle), **kwargs)
            for i, arg in enumerate(args):
                if isinstance(arg, Expr):
                    edges.append((f"node{arg._handle}", f"node{idx}:{i}"))
        for edge in edges:
            g.edge(*edge)

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

    def rindex(self, target: handle_type, startpos: handle_type) -> handle_type:
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
            return Expr(self, handle)
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
            if isinstance(a, Expr):
                self._heap.append(a._handle)
            else:
                self.write_token(a)
        self.write_end()
        return handle

    def write_ref(self, ref: handle_type):
        assert 0 < ref < HandleSentry.BEGIN
        self._heap.append(ref)

    def write_token(self, token: token_type) -> None:
        assert isinstance(token, (int, str, float))
        last = -len(self._tokens)
        handle = self._tokenmap.get(token, last)
        if handle == last:
            self._tokens.append(token)
            self._tokenmap[token] = handle
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


class Expr:
    """S-expression reference

    Default comparison is by identity (the handle).
    """

    _tree: Tree
    _handle: handle_type

    def __init__(self, tree: Tree, handle: handle_type) -> None:
        self._tree = tree
        self._handle = handle

    def __eq__(self, value: object) -> bool:
        if isinstance(value, type(self)):
            return self._tree == value._tree and self._handle == value._handle
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((id(self._tree), self._handle))

    @classmethod
    def write(cls, tree: Tree, head: str, args: tuple[value_type, ...]) -> Expr:
        handle = tree.write(head, args)
        return cls(tree, handle)

    @property
    def tree(self) -> Tree:
        return self._tree

    @cached_property
    def head(self) -> str:
        tree = self._tree
        return tree.read_head(self._handle)

    @cached_property
    def args(self) -> tuple[value_type, ...]:
        tree = self._tree
        return tree.read_args(self._handle)

    def __str__(self):
        return f"Expr({self.head}, {', '.join(map(repr, self.args))})"

    def str(self) -> str:
        from .prettyprinter import pretty_print

        return pretty_print(self)

    def __repr__(self):
        active_tree = Context.top_or_none()
        start = f"<Expr {self.head!r} [{self._handle}]"
        if active_tree is not self._tree:
            end = f" @{hex(id(self._tree))}>"
        else:
            end = ">"
        return start + end

    # Comparison API

    def __lt__(self, other) -> bool:
        if isinstance(other, type(self)):
            return self._handle < other._handle
        else:
            return NotImplemented

    # Search API

    def find_parents(self) -> Iterator[Expr]:
        """A Iterator that yields Expr that immediately contains this
        object.
        Returned values follow the order of occurrence.
        """
        crawler = TreeCrawler(self._tree)
        crawler.seek(self._handle)
        crawler.skip_to_record_end()
        while crawler.move_to_pos_of(self._handle):
            yield crawler.read_surrounding_record().to_expr()
            crawler.skip_to_record_end()

    def find_children(self) -> Iterator[Expr]:
        crawler = TreeCrawler(self._tree)
        crawler.seek(self._handle)
        record = crawler.read_record()
        for child in record.children():
            yield child.to_expr()

    def search_parents(self, pred: Callable[[Expr], bool]) -> Iterator[Expr]:
        """Yields all Expr node `e` in parent nodes of `self` and
        that `pred(e)` is True
        """
        candidates = deque(self.find_parents())
        seen = set(candidates)
        while candidates:
            p = candidates.popleft()
            if pred(p):
                yield p
            for new_p in p.find_parents():
                if new_p not in seen:
                    candidates.append(new_p)
                    seen.add(new_p)

    def walk_descendants(self) -> Iterator[tuple[tuple[Expr, ...], Expr]]:
        """Walk descendants of this Expr node.
        Breath-first order.
        """
        oldtree = self._tree
        crawler = TreeCrawler(oldtree)
        crawler.seek(self._handle)
        for parents, desc in crawler.walk_descendants():
            parent_exprs = tuple(map(lambda x: x.to_expr(), parents))
            yield parent_exprs, desc.to_expr()

    def contains(self, expr: Expr) -> bool:
        """Is `expr` part of this expression tree."""
        for _, child in expr.walk_descendants():
            if child == expr:
                return True
        return False

    # Copy API

    def copy_tree_into(self, tree: Tree) -> Expr:
        """Copy all descendants into the given tree.
        Returns a fresh Expr in the new tree.
        """
        oldtree = self._tree
        crawler = TreeCrawler(oldtree)
        crawler.seek(self._handle)
        liveset = set(_select(crawler.walk_descendants(), 1))
        surviving = sorted(liveset)
        mapping = {}
        for oldrec in surviving:
            head = oldtree.read_head(oldrec.handle)
            args = oldtree.read_args(oldrec.handle)

            mapping[oldrec.handle] = tree.write_begin()
            tree.write_token(head)

            for arg in args:
                if isinstance(arg, Expr):
                    tree.write_ref(mapping[arg._handle])
                else:
                    tree.write_token(arg)

            tree.write_end()

        out = tree.read_value(mapping[self._handle])
        assert isinstance(out, Expr)
        return out

    # Apply API

    def apply_bottomup(self, visitor: TreeVisitor) -> None:
        """
        Apply the TreeVisitor to every sexpr bottom up.
        When a sexpr is visited, it's children must have been visited prior.
        It will visit more that the subtree under `self`.
        """
        crawler = TreeCrawler(self._tree)
        crawler.move_to_first_record()
        for rec in crawler.walk():
            if rec.handle <= self._handle:  # visit all younger
                visitor.visit(rec.to_expr())

    def apply_topdown(self, visitor: TreeVisitor) -> None:
        """
        Apply the TreeVisitor to every sexpr under `self` subtree.
        """
        for _, node in self.walk_descendants():
            visitor.visit(node)


class TreeVisitor:
    def visit(self, expr: Expr):
        pass


class TreeCrawler:
    _tree: Tree
    _pos: handle_type

    def __init__(self, tree: Tree) -> None:
        self._tree = tree
        self._pos = 0

    def move_to_first_record(self) -> None:
        self._pos = 1

    @property
    def pos(self) -> handle_type:
        return self._pos

    def get(self) -> handle_type:
        return self._tree.get(self._pos)

    def seek(self, pos: handle_type) -> None:
        start_handle = self._tree.get(pos)
        assert start_handle == HandleSentry.BEGIN
        self._pos = pos

    def step(self) -> None:
        self._pos += 1

    def skip_to_record_end(self):
        self._pos = self._tree.index(HandleSentry.END, self._pos + 1)

    def walk(self) -> Iterator[Record]:
        while self._pos < self._tree.heap_size:
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
        assert self._tree.get(self._pos) == HandleSentry.BEGIN
        begin = self._pos
        end = self._tree.index(HandleSentry.END, begin)
        rec = Record(self._tree, begin, end)
        self._pos = end + 1
        return rec

    def read_surrounding_record(self) -> Record:
        start = self._tree.rindex(HandleSentry.BEGIN, self._pos)
        stop = self._tree.index(HandleSentry.END, self._pos)
        return Record(self._tree, start, stop)

    def move_to_pos_of(self, target: handle_type) -> bool:
        try:
            self._pos = self._tree.index(target, self._pos)
        except NotFound:
            return False
        else:
            return True


@dataclass(frozen=True, order=True)
class Record:
    tree: Tree
    handle: handle_type
    end_handle: handle_type

    def children(self) -> Iterator[Record]:
        """Return child records. Cannot be tokens."""
        body = self.tree.load(self.handle + 1, self.end_handle)
        for h in body:
            if h > 0:  # don't include tokens
                end = self.tree.index(HandleSentry.END, h)
                yield Record(self.tree, h, end)

    def read_head(self):
        return self.tree.read_head(self.handle)

    def read_args(self):
        return self.tree.read_args(self.handle)

    def to_expr(self) -> Expr:
        return Expr(self.tree, self.handle)

    def __repr__(self):
        return f"<Record {self.handle}:{self.end_handle} tree@{hex(id(self.tree))} >"


def expr(head: str, *args: value_type) -> Expr:
    tree = Context.top()
    return Expr.write(tree, head, args)


def _select(iterable, idx: int):
    for args in iterable:
        yield args[idx]
