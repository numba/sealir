from __future__ import annotations

import os
from pprint import pprint
from contextlib import contextmanager
from dataclasses import dataclass, field

from sealir import rvsdg, ase, lam

from egglog import Expr, i64, i64Like, Vec, String, EGraph, function


DEBUG = bool(os.environ.get("DEBUG", False))

# Borrowed deBruijn handling from normalization-by-evaluation


class Env(Expr):
    @classmethod
    def nil(cls) -> Env: ...

    @classmethod
    def loop(cls, uid: i64) -> Env: ...

    @classmethod
    def cons(cls, val: Value, env: Env) -> Env: ...

    @classmethod
    def named(cls, i: i64) -> Env: ...


# class VClosure(Expr):
#     def __init__(self, env: Env, term: STerm) -> VClosure: ...


class Value(Expr):
    @classmethod
    def bound(cls, val: STerm) -> Value:
        pass

    @classmethod
    def true(cls) -> Value:
        pass
    @classmethod
    def false(cls) -> Value:
        pass

class Scope(Expr):
    def __init__(self, uid: i64Like): ...


class STerm(Expr):

    @classmethod
    def lam(cls, scope: Scope, body: STerm) -> STerm: ...

    @classmethod
    def app(cls, lam: STerm, arg: STerm) -> STerm: ...

    @classmethod
    def var(cls, scope: Scope, debruijn: i64Like) -> STerm: ...

    @classmethod
    def unpack(cls, idx: i64Like, tup: STerm) -> STerm: ...

    @classmethod
    def pack(cls, tup: Vec[STerm]) -> STerm: ...

    # Others
    @classmethod
    def param(cls, val: i64Like) -> STerm: ...
    @classmethod
    def const_i64(cls, val: i64Like) -> STerm: ...
    @classmethod
    def undef(cls) -> STerm: ...

    @classmethod
    def ret(cls, iostate: STerm, retval: STerm) -> STerm: ...

    @classmethod
    def func(cls, name: String) -> STerm: ...

    @classmethod
    def if_else(cls, test: STerm, then: STerm, orelse: STerm) -> STerm: ...

    @classmethod
    def do_while(cls, uid: i64Like, body: STerm) -> STerm: ...


def make_call(fn, *args):
    for arg in args:
        fn = STerm.app(fn, arg)
    return fn

def PyUnop(opname: String, iostate: STerm, arg: STerm) -> STerm:
    return make_call(STerm.func(f"PyUnOp::{opname}"), iostate, arg)


def PyBinop(opname: String, iostate: STerm, lhs: STerm, rhs: STerm) -> STerm:
    return make_call(STerm.func(f"PyBinOp::{opname}"), iostate, lhs, rhs)


def PyCmpop(opname: String, iostate: STerm, lhs: STerm, rhs: STerm) -> STerm:
    return make_call(STerm.func(f"PyCmpOp::{opname}"), iostate, lhs, rhs)


def Pack(*args: STerm) -> STerm:
    return STerm.pack(Vec(*args))


@function
def eval(env: Env, expr: STerm) -> Value: ...


@function
def VClosure(env: Env, expr: STerm) -> Value: ...


@function
def VLoop(body: Value) -> Value: ...


@function
def Lookup(env: Env, debruijn: i64Like) -> Value: ...


@function
def ParentEnv(env: Env) -> Env: ...


@function
def Depth(env: Env) -> i64: ...


@function
def VLam(cl: Value) -> Value: ...


@function
def VApp(f: Value, x: Value) -> Value: ...
@function
def VRet(f: Value, x: Value) -> Value: ...
@function
def VUnpack(i: i64Like, x: Value) -> Value: ...

@function
def VFunc(fname: String) -> Value: ...
@function
def VBranch(cond: Value, then: Value, orelse: Value) -> Value: ...

def vcall(fn: Value, *args: Value) -> Value:
    expr = fn
    for arg in args:
        expr = VApp(expr, arg)
    return expr

@function
def is_pure(call: Value) -> Value: ...

@function
def VBinOp(opname: String, lhs: Value, rhs: Value) -> Value: ...

@dataclass(frozen=True)
class EnvCtx:
    blam_stack: list[ase.SExpr] = field(default_factory=list)
    lam_map: dict[ase.SExpr, Env] = field(default_factory=dict)

    @contextmanager
    def bind_lam(self, lam_expr: ase.SExpr, scope: Scope):
        self.blam_stack.append(lam_expr)
        scope = self.lam_map.setdefault(lam_expr, scope)
        try:
            yield scope
        finally:
            self.blam_stack.pop()

    def get_parent_lambda(self) -> ase.SExpr:
        return self.blam_stack[-1]

    def get_parent_scope(self) -> Scope:
        return self.lam_map[self.get_parent_lambda()]


@dataclass(frozen=True)
class EggConvState(ase.TraverseState):
    context: EnvCtx


def convert_tuple_to_egglog(root, assume):
    from egglog import birewrite, rewrite, set_, rule, eq, ne, union, set_

    def conversion(expr: ase.BasicSExpr, state: EggConvState):
        ctx = state.context
        match expr:
            # Basic Lambda Calculus
            case lam.Lam(body):
                with ctx.bind_lam(expr, Scope(expr._handle)) as env:
                    return STerm.lam(env, (yield body))
            case lam.App(arg=argval, lam=lam_func):
                # flipped args
                return STerm.app((yield lam_func), (yield argval))
            case lam.Arg(int(argidx)):
                return STerm.var(ctx.get_parent_scope(), argidx)
            case lam.Unpack(idx=int(idx), tup=packed_expr):
                return STerm.unpack(idx, (yield packed_expr))
            case lam.Pack(args):
                elems = []
                for arg in args:
                    elems.append((yield arg))
                return Pack(*elems)
            case rvsdg.Return(iostate=iostate, retval=retval):
                return STerm.ret((yield iostate), (yield retval))
            case rvsdg.BindArg(val):
                return STerm.param(val)
            # SCFG extensions
            case rvsdg.Scfg_If(test=test, then=then, orelse=orelse):
                return STerm.if_else((yield test), (yield then), (yield orelse))
            case rvsdg.Scfg_While(body=body):
                return STerm.do_while(expr._handle, (yield body))
            # Py extensions
            case rvsdg.Py_UnaryOp(
                opname=str(op),
                iostate=iostate,
                arg=arg,
            ):
                return PyUnop(op, (yield iostate), (yield arg))
            case rvsdg.Py_BinOp(
                opname=str(op),
                iostate=iostate,
                lhs=lhs,
                rhs=rhs,
            ):
                return PyBinop(op, (yield iostate), (yield lhs), (yield rhs))
            case rvsdg.Py_Compare(
                opname=str(op),
                iostate=iostate,
                lhs=lhs,
                rhs=rhs,
            ):
                return PyCmpop(op, (yield iostate), (yield lhs), (yield rhs))
            case rvsdg.Py_Int(value=int(arg)):
                return make_call(STerm.func(f"PyInt"), STerm.const_i64(int(arg)))
            case rvsdg.Py_Undef():
                return STerm.undef()
            case _:
                raise ValueError(f"? {expr}")

    memo = ase.traverse(root, conversion, state=EggConvState(context=EnvCtx()))
    sterm = memo[root]

    egraph = EGraph()

    @egraph.register
    def _nbe(
        expr: STerm,
        term: STerm,
        val: STerm,
        val2: STerm,
        env: Env,
        scope: Scope,
        x: Value,
        i: i64,
        text: String,
        vec_terms: Vec[STerm],
        vec_vals: Vec[Value],
    ):
        # Uses NbE logic from https://github.com/egraphs-good/egglog/pull/28

        Var = STerm.var
        Lam = STerm.lam
        App = STerm.app

        # --- Lambda Evaluation ---
        # create VClosure from Lam
        yield rewrite(eval(env, Lam(scope, expr))).to(
            VLam(VClosure(env, expr))
        )

        # create VApp from App
        yield rewrite(eval(env, App(term, val))).to(
            VApp(eval(env, term), eval(env, val))
        )

        # reduce VApp(VLam(VClosure))
        yield rewrite(VApp(VLam(VClosure(env, term)), x)).to(
            eval(Env.cons(x, env), term)
        )

        # --- Lookup logic ---
        # fmt: off
        yield rewrite(   eval( env, Var(scope, i)  ) ).to( Lookup( env, i )    )
        yield rewrite( Lookup( Env.cons(x, env), 0 ) ).to( x                   )
        yield rewrite( Lookup( Env.cons(x, env), i ) ).to( Lookup( env, i - 1),
            # given
            ne(i).to(0)
        )
        # fmt: on

        # --- Value bound ---

        # eval of (param) -> Value
        yield rewrite(eval(env, STerm.param(i))).to(
            Value.bound(STerm.param(i))
        )

        # --- VRet ---
        # eval of (ret)
        yield rewrite(eval(env, STerm.ret(val, val2))).to(
            VRet(eval(env, val), eval(env, val2))
        )

        # --- VUnpack ---
        # eval of (unpack)
        yield rewrite(eval(env, STerm.unpack(i, val))).to(
            VUnpack(i, eval(env, val))
        )
        # --- Pack ---
        # eval of (pack vec...)
        yield rewrite(
            VUnpack(i, eval(env, STerm.pack(vec_terms)))
        ).to(
            eval(env, vec_terms[i])
        )
        # --- VFunc ---
        yield rewrite(
            eval(env, STerm.func(text))
        ).to(
            VFunc(text)
        )


    @egraph.register
    def _nbe_scfg_branch(
        expr: STerm,
        expr2: STerm,
        val: STerm,
        env: Env,
    ):
        # --- if_else ---
        yield rewrite(
            eval(env, STerm.if_else(val, expr, expr2))
        ).to(
            VBranch(eval(env, val), eval(env, expr), eval(env, expr2))
        )
        # simplify true
        yield rewrite(
            VBranch(Value.true(), eval(env, expr), eval(env, expr2))
        ).to(
            eval(env, expr)
        )
        # simplify false
        yield rewrite(
            VBranch(Value.false(), eval(env, expr), eval(env, expr2))
        ).to(
            eval(env, expr2)
        )

    @egraph.register
    def _nbe_scfg_loop(
        expr: STerm,
        expr2: STerm,
        term: STerm,
        val: STerm,
        val2: STerm,
        env: Env,
        scope: Scope,
        x: Value,
        y: Value,
        i: i64,
        uid: i64,
        text: String,
    ):
        # --- do_while ---
        yield rewrite(
            eval(env, STerm.do_while(uid, expr))
        ).to(
            VLoop(eval(Env.cons(Lookup(env, 0), Env.loop(uid)), expr)),
        )

        """

        Phi(inc1, loopback1), .. Phi(incN, loopbackN)
        """

    @egraph.register
    def _py(
        x: Value,
        y: Value,
        z: Value,
    ):
        def binop_rewrite(prefix: str, opname: str):
            call = vcall(VFunc(f"{prefix}::{opname}"), x, y, z)
            yield rewrite(
                VUnpack(0, call)
            ).to(
                x,
                # given
                is_pure(call)
            )
            yield rewrite(
                VUnpack(1, call)
            ).to(
                VBinOp(opname, y, z),
                # given
                is_pure(call)
            )

        # --- PyBinOp ---
        yield from binop_rewrite("PyBinOp", "+")
        # --- PyCmpOp ---
        yield from binop_rewrite("PyCmpOp", ">")


    if assume:
        assume(egraph)

    env = Env.nil()
    rootexpr = egraph.let("root", eval(env, sterm))
    # print(str(sterm).replace("STerm.", ""))

    saturate(egraph)
    print("output".center(80, "-"))

    out = egraph.simplify(rootexpr, 1)

    # for extracted in egraph.extract_multiple(rootexpr, 10):
    #     print("---", extracted)
    print(str(out).replace("STerm.", ""))

    return egraph, rootexpr, out


def saturate(egraph, limit=1_000):
    if DEBUG:
        egraph.saturate(limit=limit)
    else:
        # workaround egraph.saturate() is always opening the viz.
        i = 0
        while egraph.run(1).updated and i < limit:
            i += 1

def serialize_to_viz(egraph):
    from sealir.egg_utils import extract_eclasses, ECTree
    from sealir.eggview import write_page
    ecdata = extract_eclasses(egraph)

    ectree = ECTree(ecdata)

    with ectree.write_html_root() as buf:
        with open("ectree.html", "w") as fout:
            write_page(fout, buf.getvalue())


def run(udt, checks, *, assume=None, custom_fn=None, debug_eggview=False):
    from egglog import eq

    sexpr = rvsdg.restructure_source(udt)
    if True:
        from sealir import rvsdg_conns
        import inspect
        sig = inspect.signature(udt)
        edges = rvsdg_conns.build_value_state_connection(
            sexpr, sig.parameters.keys()
        )

        dot = rvsdg_conns.render_dot(edges)
        dot.view()

    grm = rvsdg.Grammar(sexpr._tape)

    args = [0, 1, 2]
    sexpr = lam.app_func(
        grm, sexpr, *[grm.write(rvsdg.BindArg(x)) for x in args]
    )
    pprint(ase.as_tuple(sexpr, depth=-1))
    egraph, rootexpr, out = convert_tuple_to_egglog(sexpr, assume)

    if custom_fn is not None:
        custom_fn(egraph, rootexpr)

    if debug_eggview:
        serialize_to_viz(egraph)

    # Check
    facts = [eq(rootexpr).to(x)
             for x in checks]
    egraph.check(*facts)
    return egraph


def test_basic_return_1():
    def udt(a, b):
        return a

    checks = [
        VRet(Value.bound(STerm.param(0)), Value.bound(STerm.param(1)))
    ]
    run(udt, checks)

def test_basic_return_2():
    def udt(a, b):
        return b

    checks = [
        VRet(Value.bound(STerm.param(0)), Value.bound(STerm.param(2)))
    ]
    run(udt, checks)


def test_binop_add():
    def udt(a, b):
        return a + b + a

    def assume(egraph):
        @egraph.register
        def facts(x: Value, y: Value, z: Value):
            from egglog import rule

            yield rule(
                vcall(VFunc("PyBinOp::+"), x, y, z)
            ).then(
                is_pure(vcall(VFunc("PyBinOp::+"), x, y, z))
            )

    checks = [
        VRet(Value.bound(STerm.param(0)),
             VBinOp("+",
                    VBinOp("+",
                           Value.bound(STerm.param(1)),
                           Value.bound(STerm.param(2))),
                           Value.bound(STerm.param(1))))
    ]
    run(udt, checks, assume=assume)


def test_max_if_else():
    def udt(x, y):
        # scalar max
        if x > y:
            return x
        else:
            return y

    def assume(egraph):
        @egraph.register
        def facts(x: Value, y: Value, z: Value):
            from egglog import rewrite, rule

            yield rule(
                vcall(VFunc("PyCmpOp::>"), x, y, z)
            ).then(
                is_pure(vcall(VFunc("PyCmpOp::>"), x, y, z))
            )

            yield rewrite(
                VBinOp(">", x, y)
            ).to(
                Value.true()
            )

    checks = [
        VRet(Value.bound(STerm.param(0)),
             Value.bound(STerm.param(1)))
    ]
    run(udt, checks, assume=assume)

    def assume(egraph):
        @egraph.register
        def facts(x: Value, y: Value, z: Value):
            from egglog import rewrite, rule

            yield rule(
                vcall(VFunc("PyCmpOp::>"), x, y, z)
            ).then(
                is_pure(vcall(VFunc("PyCmpOp::>"), x, y, z))
            )

            yield rewrite(
                VBinOp(">", x, y)
            ).to(
                Value.false()
            )

    checks = [
        VRet(Value.bound(STerm.param(0)),
             Value.bound(STerm.param(2)))
    ]
    run(udt, checks, assume=assume)



def test_while_loop():
    def udt(x, y):
        c = 0
        i = x
        while i < y:
            c = c + i
            i = 1 + 1
        return c

    def assume(egraph):
        pass
        # @egraph.register
        # def facts(x: Value, y: Value, z: Value):
        #     from egglog import rewrite, rule

        #     yield rule(
        #         vcall(VFunc("PyCmpOp::>"), x, y, z)
        #     ).then(
        #         is_pure(vcall(VFunc("PyCmpOp::>"), x, y, z))
        #     )

        #     yield rewrite(
        #         VBinOp(">", x, y)
        #     ).to(
        #         Value.false()
        #     )

    checks = [
        # VRet(Value.bound(STerm.param(0)),
        #      Value.bound(STerm.param(2)))
    ]

    def manual_extract(egraph: EGraph, rootexpr):
        from sealir.egg_utils import extract_eclasses, ECTree
        import networkx as nx
        from networkx.drawing.nx_agraph import to_agraph

        print(egraph.extract(rootexpr))


        ecdata = extract_eclasses(egraph)
        [vloop] = ecdata.find("VLoop")
        print("VLOOP", vloop)

        ignore_types = {'Env', 'Scope', 'STerm'}
        ignore_ops = {"eval"}

        G = ecdata.to_networkx(vloop,
                               ignore_types=ignore_types)


        if False:
            # Rendering
            # Convert NetworkX graph to Graphviz graph
            A = to_agraph(G)

            # Generate DOT code
            # A.layout('dot')
            # A.attr(rankdir='TB')  # Top to Bottom layout
            A.draw('networkx_graph.svg', prog='dot')

        def compute_path_depths(G):
            # Check if G is a DAG
            if not nx.is_directed_acyclic_graph(G):
                raise ValueError("Graph must be a Directed Acyclic Graph (DAG)")

            # Compute topological generations
            generations = list(nx.topological_generations(G))

            # Compute depths
            depths = {}
            for generation_index, generation in enumerate(generations):
                for node in generation:
                    depths[node] = generation_index

            return depths

        depths = compute_path_depths(G.reverse())

        def extract_best_children(root):
            stack = [root]

            mapping = {}

            while stack:
                node = stack.pop()

                if node in mapping:
                    continue

                children = []
                for child in ecdata.children_of(node):
                    candidates = {}
                    for member in ecdata.eclasses[child.eclass]:
                        if member.op not in ignore_ops and member.type not in ignore_types:
                            d = depths.get(member.key, 0)
                            candidates[member] = d
                    if candidates:
                        best = min(candidates.items(), key=lambda x: x[1])[0]
                        children.append(best)
                        stack.append(best)

                mapping[node] = children

            return mapping

        out = extract_best_children(vloop)
        import graphviz as gz

        dotg = gz.Digraph()
        for node, children in out.items():
            dotg.node(node.key)
            for ch in children:
                dotg.edge(node.key, ch.key)

        # dotg.view()



    run(udt, checks, assume=assume, debug_eggview=True,
        custom_fn=manual_extract)


test_me = test_while_loop
if __name__ == "__main__":
    test_me()
