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


def convert_tuple_to_egglog(root):
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
        i: i64,
        j: i64,
        m: i64,
        n: i64,
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
    ecdata = extract_eclasses(egraph)

    ectree = ECTree(ecdata)
    pprint(ectree._parent_eclasses)
    print("Roots")
    roots = ectree.root_eclasses()
    pprint(roots)
    [root] = roots
    with ectree.write_html_root(root) as buf:
        with open("ectree.html", "w") as fout:
            write_style(fout)

            fout.write("<div id='hidden' >")
            fout.write("</div>")
            fout.write("<div id='main' >")
            fout.write(buf.getvalue())
            fout.write("</div>")

def write_style(fout):
    def println(*args):
        print(*args, file=fout)

    println("""
<style>
body {
    padding-bottom: 5em;
}

#hidden {
    display: none;
}

div.eclass {
    border: 1px solid #ccc;
    padding: 2px;
    margin: 1px;
    padding-right: 0;
    margin-right: 0;
    background-color: #eee;
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 5px;
}
div.term {
    border: 1px solid #ccc;
    padding: 1px;
    margin: 1px;
    background-color: #fff;
    flex: 0 1 auto;
}

.activated:not(:has(.activated))  {
    background-color: green;

}

.toolbutton {
    cursor: pointer;
}

.eclass_name  {
    font-size: 0.8em;
    color: #666;
}
.bottom-toolbox {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.5);
    padding: 10px;
    box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
    z-index: 9999;
}


.bottom-textbox {
    width: 90%;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 16px;
    margin: 0 auto;
    display: block;
}
</style>
<script>

document.addEventListener('click', (e) => {

    if (e.target.classList.contains("term")) {
        e.stopPropagation();

        document.querySelectorAll('.activated').forEach(term => {
            term.classList.remove('activated');
        });

        e.target.classList.add('activated');
    }
});

/* JS to copy HTML element of the referenced eclass */
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('eclass_name')) {
        const ec = e.target.dataset.eclass;
        const targetContent = document.querySelector(`div[data-eclass="${ec}"]`);
        if (targetContent) {
            e.target.parentElement.innerHTML = targetContent.innerHTML;
        }
        e.preventDefault();
    }
});
</script>

<div class="bottom-toolbox">
    <p>
    Click term to select. Once selected, key 'f' to focus; key 'v' to toggle
    visibility. Enter op name in box to filter out.
    </p>
    <input type="text" class="bottom-textbox" placeholder="op names to hide">
</div>
<script>
const filterBox = document.querySelector('.bottom-textbox');

filterBox.addEventListener('input', function() {
    const hideOps = this.value.split(',').map(s => s.trim());
    const allTerms = document.querySelectorAll('div.term[data-term-op]');

    allTerms.forEach(term => {
        const opname = term.getAttribute('data-term-op');
        if (hideOps.includes(opname)) {
            term.style.display = 'none';
        } else {
            term.style.display = 'block';
        }
    });
});



document.addEventListener('keydown', function(e) {
    const termDiv = document.querySelector("div.activated");
    if (!termDiv) return;

    if (e.key == 'f' ) {
        const mainDiv = document.getElementById('main');
        mainDiv.innerHTML = termDiv.innerHTML;
        e.preventDefault();
    } else if (e.key == 'v') {
        const op = termDiv.dataset.termOp;
        const terms = document.querySelectorAll(`div[data-term-op="${op}"] > div.content`);
        terms.forEach(term => {
            term.style.display = term.style.display === 'none' ? 'block' : 'none';
        });
        e.preventDefault();
    }
});


document.addEventListener('DOMContentLoaded', function(e) {
    /* copy original state */
    const mainDiv = document.getElementById('main');
    const hiddenDiv = document.getElementById('hidden');
    hiddenDiv.innerHTML = mainDiv.innerHTML;
});
</script>
""")

def run(udt, checks):
    from egglog import eq

    sexpr = rvsdg.restructure_source(udt)

    grm = rvsdg.Grammar(sexpr._tape)

    args = [0, 1, 2]
    sexpr = lam.app_func(
        grm, sexpr, *[grm.write(rvsdg.BindArg(x)) for x in args]
    )
    pprint(ase.as_tuple(sexpr, depth=-1))
    egraph, rootexpr, out = convert_tuple_to_egglog(sexpr)

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


def test_me():
    def udt(a, b):
        return a + b

    checks = [
        VRet(Value.bound(STerm.param(0)), Value.bound(STerm.param(1)))
    ]
    run(udt, checks)


if __name__ == "__main__":
    test_me()
