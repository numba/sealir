from __future__ import annotations

import os
from pprint import pprint
from contextlib import contextmanager
from dataclasses import dataclass, field

from sealir import rvsdg, ase, lam

from egglog import Expr, i64, Vec, String, EGraph, function


DEBUG = bool(os.environ.get("DEBUG", False))

# Borrowed deBruijn handling from normalization-by-evaluation


class Env(Expr):
    @classmethod
    def nil(cls) -> Env: ...

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


class Scope(Expr):
    def __init__(self, uid: i64): ...


class STerm(Expr):

    @classmethod
    def lam(cls, scope: Scope, body: STerm) -> STerm: ...

    @classmethod
    def app(cls, lam: STerm, arg: STerm) -> STerm: ...

    @classmethod
    def var(cls, scope: Scope, debruijn: i64) -> STerm: ...

    @classmethod
    def unpack(cls, idx: i64, tup: STerm) -> STerm: ...

    @classmethod
    def pack(cls, tup: Vec[STerm]) -> STerm: ...

    # Others
    @classmethod
    def param(cls, val: i64) -> STerm: ...

    @classmethod
    def ret(cls, iostate: STerm, retval: STerm) -> STerm: ...

    @classmethod
    def func(cls, name: String) -> STerm: ...


def make_call(fn, *args):
    for arg in args:
        fn = STerm.app(fn, arg)
    return fn


def PyBinop(opname: String, iostate: STerm, lhs: STerm, rhs: STerm) -> STerm:
    return make_call(STerm.func(f"PyBinOp::{opname}"), iostate, lhs, rhs)


def Pack(*args: STerm) -> STerm:
    return STerm.pack(Vec(*args))


@function
def eval(env: Env, expr: STerm) -> Value: ...


@function
def VClosure(env: Env, expr: STerm) -> Value: ...


@function
def Lookup(env: Env, debruijn: i64) -> Value: ...


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
def VUnpack(i: i64, x: Value) -> Value: ...
@function
def VFunc(fname: String) -> Value: ...

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
    def conversion(expr: ase.BasicSExpr, state: EggConvState):
        ctx = state.context
        match expr:
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
            case rvsdg.Py_BinOp(
                opname=str(op),
                iostate=iostate,
                lhs=lhs,
                rhs=rhs,
            ):
                return PyBinop(op, (yield iostate), (yield lhs), (yield rhs))
            case rvsdg.Return(iostate=iostate, retval=retval):
                return STerm.ret((yield iostate), (yield retval))
            case rvsdg.BindArg(val):
                return STerm.param(val)
            case _:
                raise ValueError(f"? {expr}")

    memo = ase.traverse(root, conversion, state=EggConvState(context=EnvCtx()))
    sterm = memo[root]

    egraph = EGraph()

    @egraph.register
    def _custom(
        lam: STerm,
        expr: STerm,
        expr2: STerm,
        expr3: STerm,
        term: STerm,
        val: STerm,
        val2: STerm,
        exprVec: Vec[STerm],
        env: Env,
        env2: Env,
        scope: Scope,
        x: Value,
        y: Value,
        z: Value,
        i: i64,
        j: i64,
        m: i64,
        n: i64,
        text: String,
    ):
        from egglog import birewrite, rewrite, set_, rule, eq, ne, union, set_

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

        # --- VFunc ---
        yield rewrite(
            eval(env, STerm.func(text))
        ).to(
            VFunc(text)
        )

        # --- PyBinOp ---
        call = vcall(VFunc("PyBinOp::+"), x, y, z)
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
            VBinOp("+", y, z),
            # given
            is_pure(call)
        )

    if assume:
        assume(egraph)

    env = Env.nil()
    rootexpr = egraph.let("root", eval(env, sterm))
    print(str(sterm).replace("STerm.", ""))

    saturate(egraph)
    print("output".center(80, "-"))

    out = egraph.simplify(rootexpr, 1)
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



def run(udt, checks, *, assume=None, debug_eggview=False):
    from egglog import eq

    sexpr = rvsdg.restructure_source(udt)

    grm = rvsdg.Grammar(sexpr._tape)

    args = [0, 1, 2]
    sexpr = lam.app_func(
        grm, sexpr, *[grm.write(rvsdg.BindArg(x)) for x in args]
    )
    pprint(ase.as_tuple(sexpr, depth=-1))
    egraph, rootexpr, out = convert_tuple_to_egglog(sexpr, assume)

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


if __name__ == "__main__":
    test_me()
