from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Iterable, Iterator, Sequence

from sealir import ase, lam, rvsdg
from sealir.graphviz_support import graphviz_function
from sealir.rvsdg import restructure_source


@dataclass(frozen=True)
class GraphNode:
    is_origin = False
    enclosing_subgraph: Subgraph | None = field(
        kw_only=True, default=None, repr=False
    )

    def get_id(self) -> str:
        return repr(self)

    def get_label_text(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class Subgraph(GraphNode):
    expr: ase.SExpr
    desc: str


@dataclass(frozen=True)
class Argument(GraphNode):
    is_origin = True
    name: str
    argidx: int


@dataclass(frozen=True)
class Phi(GraphNode):
    expr: ase.SExpr
    port_index: int

    def get_label_text(self) -> str:
        return (
            f"Phi({self.expr._head} [{self.expr._handle}], "
            f"port={self.port_index})"
        )


@dataclass(frozen=True)
class OriginIO(GraphNode):
    is_origin = True


@dataclass(frozen=True)
class Sink(GraphNode):
    pass


@dataclass(frozen=True)
class ExprNode(GraphNode):
    expr: ase.SExpr

    def get_label_text(self) -> str:
        expr = self.expr
        args = [
            (f"[{x._handle}]" if isinstance(x, ase.SExpr) else repr(x))
            for x in expr._args
        ]
        args = [f"[{expr._handle}]", expr._head, *args]
        return " ".join(args)


@dataclass(frozen=True)
class Packed(GraphNode):
    values: tuple[GraphNode, ...]

    def get_label_text(self) -> str:
        return f"Packed"


@dataclass(frozen=True)
class Edge:
    src: GraphNode
    dst: GraphNode
    dst_port: int


def ensure_io(obj: Any) -> rvsdg.EvalIO:
    # assert isinstance(obj, (rvsdg.EvalIO, lam.Unpack))
    return obj


def build_value_state_connection(
    lam_node: ase.SExpr, argnames: Sequence[str]
) -> list[Edge]:
    """
    Builds a value-state connection for a lambda node in the RVSDG form.

    This function uses partial evaluation to build up the value-state edges.
    """
    edges = set()
    subgraph_stack: list[Subgraph] = []

    sink = Sink()

    def tos_subgraph() -> Subgraph | None:
        tos = subgraph_stack[-1] if subgraph_stack else None
        return tos

    @contextmanager
    def subgraph_context(*args, **kwargs):
        tos = tos_subgraph()
        subgraph = Subgraph(*args, **kwargs, enclosing_subgraph=tos)
        subgraph_stack.append(subgraph)
        try:
            yield
        finally:
            subgraph_stack.pop()

    def wrap(expr: ase.SExpr | GraphNode) -> GraphNode:
        match expr:
            case ase.SExpr():
                return ExprNode(expr, enclosing_subgraph=tos_subgraph())
            case GraphNode():
                return expr
            case rvsdg.EvalIO():
                return OriginIO()
            case tuple():
                elts = map(wrap, expr)
                ret = Packed(elts, enclosing_subgraph=tos_subgraph())
                for i, elt in enumerate(elts):
                    edges.add(Edge(src=elt, dst=ret, dst_port=i))
                return ret
            case _:
                raise AssertionError(type(expr))

    def partial_eval(expr: ase.SExpr, state: rvsdg.EvalLamState):
        ctx = state.context
        match expr:
            case lam.Lam(body):
                retval = yield body
                return retval
            case lam.App(arg=argval, lam=lam_func):
                with ctx.bind_app(lam_func, (yield argval)):
                    retval = yield lam_func
                    return retval
            case lam.Arg(int(argidx)):
                retval = ctx.blam_stack[-argidx - 1].value
                return retval
            case lam.Unpack(idx=int(idx), tup=packed_expr):
                packed = yield packed_expr
                match packed:
                    case tuple():
                        retval = packed[idx]
                        return retval
                    case ase.SExpr() as arg:
                        edges.add(
                            Edge(src=wrap(arg), dst=wrap(expr), dst_port=0)
                        )
                        return expr
                    case GraphNode() as arg:
                        edges.add(Edge(src=arg, dst=wrap(expr), dst_port=0))
                        return expr
                    case _:
                        raise NotImplementedError
            case lam.Pack(args):
                elems = []
                for arg in args:
                    elems.append((yield arg))
                retval = tuple(elems)
                return retval
            case rvsdg.BindArg():
                return ctx.value_map[expr]
            case rvsdg.Scfg_If(
                test=cond,
                then=br_true,
                orelse=br_false,
            ):
                cond_node = wrap((yield cond))

                with subgraph_context(expr, "if"):
                    with subgraph_context(expr, "then"):
                        br_true_node = yield br_true
                    with subgraph_context(expr, "else"):
                        br_false_node = yield br_false
                    phis = []
                    for i, (left, right) in enumerate(
                        zip(br_true_node, br_false_node, strict=True)
                    ):
                        phi = Phi(
                            expr,
                            port_index=i,
                            enclosing_subgraph=tos_subgraph(),
                        )
                        phis.append(phi)
                        edges.add(Edge(src=wrap(left), dst=phi, dst_port=0))
                        edges.add(Edge(src=wrap(right), dst=phi, dst_port=1))
                return tuple(phis)
            case rvsdg.Scfg_While(body=loopblk):
                tos = ctx.blam_stack[-1]
                assert isinstance(tos.value, tuple)

                with subgraph_context(expr, "loop"):
                    phis_mut: list[Phi] = []
                    for i, v in enumerate(tos.value):
                        phi = Phi(
                            expr,
                            port_index=i,
                            enclosing_subgraph=tos_subgraph(),
                        )
                        edges.add(Edge(src=wrap(v), dst=phi, dst_port=0))
                        phis_mut.append(phi)
                    phis = tuple(phis_mut)
                    # Replace the top of stack with the out going value of the
                    # loop body
                    ctx.blam_stack[-1] = ctx.blam_stack[-1]._replace(
                        value=phis
                    )

                    loop_end_vars = yield loopblk
                    for phi, lev in zip(phis, loop_end_vars, strict=True):
                        edges.add(Edge(src=wrap(lev), dst=phi, dst_port=0))
                return tuple(phis)
            case rvsdg.Return(iostate=iostate, retval=retval):
                ioval = ensure_io((yield iostate))
                retval = yield retval
                edges.add(Edge(src=wrap(ioval), dst=sink, dst_port=0))
                edges.add(Edge(src=wrap(retval), dst=sink, dst_port=1))
                return ioval, retval
            case _:
                for i, arg in enumerate(expr._args):
                    if isinstance(arg, ase.SExpr):
                        argval = yield arg
                        edges.add(
                            Edge(src=wrap(argval), dst=wrap(expr), dst_port=i)
                        )
                return expr

    args = [Argument(name=k, argidx=i) for i, k in enumerate(argnames)]
    grm = rvsdg.Grammar(lam_node._tape)
    ctx = rvsdg.EvalCtx.from_arguments_and_locals(args, {})
    with grm:
        app_root = lam.app_func(grm, lam_node, *ctx.make_arg_node(grm))

    ase.traverse(app_root, partial_eval, rvsdg.EvalLamState(context=ctx))
    return edges


@graphviz_function
def render_dot(edges: Sequence[Edge], *, gv: Any) -> Any:
    g = gv.Digraph(node_attr={"shape": "rect"})

    def make_id(node: GraphNode):
        return node.get_id()

    def make_label(expr: GraphNode):
        return expr.get_label_text()

    def make_node(g, node: GraphNode):
        if isinstance(node, Sink):
            pass
        else:
            g.node(make_id(node), label=make_label(node))

    origin = g.node("origin", shape="doublecircle", label="origin", rank="min")
    sink = g.node(
        make_id(Sink()), label="sink", rank="max", shape="doublecircle"
    )

    subgraph_map: dict[Subgraph | None, Any]

    subgraph_map = {}

    def draw_nodes_in_subgraph(nodes: Iterable[GraphNode]):
        for node in nodes:
            parent = node.enclosing_subgraph
            if parent not in subgraph_map:
                subgraph_map[parent] = gv.Digraph(
                    name=f"cluster_{id(parent)}",
                    graph_attr=dict(
                        color="grey", label=parent.desc if parent else ""
                    ),
                )

            if not isinstance(node, Subgraph):
                subg = subgraph_map[parent]
                make_node(subg, node)

    nodes: set[GraphNode] = set()
    for edge in edges:
        nodes.add(edge.src)
        nodes.add(edge.dst)

    draw_nodes_in_subgraph(nodes)

    # find subgraph nesting depths
    subgraph_depth: dict[Subgraph | None, int] = {None: 0}

    def find_depth_of_subgraph(subgraph: Subgraph):
        parent = subgraph.enclosing_subgraph
        if parent is not None:
            if parent not in subgraph_depth:
                find_depth_of_subgraph(parent)
        ret = subgraph_depth[subgraph] = 1 + subgraph_depth[parent]
        return ret

    for subgraph in subgraph_map.keys():
        if subgraph is not None:
            find_depth_of_subgraph(subgraph)

    ranked = sorted(subgraph_depth.items(), key=lambda x: x[1], reverse=True)
    for subgraph, _ in ranked:
        if subgraph is not None:
            parent = subgraph.enclosing_subgraph
            subgraph_map[parent].subgraph(subgraph_map[subgraph])

    g.subgraph(subgraph_map[None])

    for edge in edges:
        g.edge(
            make_id(edge.src), make_id(edge.dst), headlabel=str(edge.dst_port)
        )

    for n in nodes:
        if n.is_origin:
            g.edge("origin", make_id(n), weight="10")
    return g
