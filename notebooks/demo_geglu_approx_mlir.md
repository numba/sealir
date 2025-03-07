---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# GEGLU Approximation

```{code-cell} ipython3

from __future__ import annotations

import os
import time

import numpy as np
from egglog import EGraph, String, eq

from sealir import ase, rvsdg
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import (
    DoPartialEval,
    Env,
    Eval,
    GraphRoot,
    PartialEvaluated,
    Term,
    Value,
    make_rules,
    valuelist,
)
from sealir.eqsat.rvsdg_extract import egraph_extraction
from sealir.llvm_pyapi_backend import llvm_codegen
from sealir.mlir_backend import Compiler, call_ufunc
```

## Prelude


### Equality Saturation utilities



```{code-cell} ipython3

def equality_saturation(
    root,
    *extra_statements,
    checks=[],
    assume=None,
    debug_points=None,
    ruleset=None,
) -> EGraph:
    """
    Example assume
    --------------

    > def assume(egraph: EGraph):
    >     @egraph.register
    >     def facts(val: Value):
    >         from egglog import eq, rule, union
    >         yield rule(
    >             eq(val).to(VBinOp("Lt", Value.ConstI64(0), Value.ConstI64(1)))
    >         ).then(union(val).with_(Value.BoolTrue()))

    """

    egraph = EGraph()  # save_egglog_string=True)
    egraph.let("root", root)
    for i, stmt in enumerate(extra_statements):
        egraph.let(f"stmt{i}", stmt)
    # egraph.display()

    if ruleset is None:
        ruleset = make_rules()
    if debug_points:
        for k, v in debug_points.items():
            egraph.let(f"debug_point_{k}", v)

    if assume is not None:
        assume(egraph)

    ts = time.time()
    # saturate(egraph, ruleset)
    egraph.run(ruleset.saturate())
    te = time.time()
    print("saturation time", te - ts)

    ts = time.time()
    out = egraph.simplify(root, 1)
    te = time.time()
    print("extraction time", te - ts)
    # print(egraph.as_egglog_string)
    print("simplified output".center(80, "-"))
    print(out)
    print("=" * 80)
    # egraph.display()
    if checks:
        try:
            egraph.check(*checks)
        except Exception:
            if debug_points:
                for k, v in debug_points.items():
                    print(f"debug {k}".center(80, "-"))
                    for each in egraph.extract_multiple(v, 5):
                        print(each)
                        print("-=-")

            raise
    return egraph

```

### MLIR codegen 

```{code-cell} ipython3


def generate_mlir(root: ase.SExpr):
    import sealir.rvsdg.grammar as rg

    ctxargs = []
    ctxportmap = {}

    sourcebuf = []

    ctr = 0

    def codegen(expr: ase.SExpr, state: ase.TraverseState):

        def fresh_name():
            nonlocal ctr
            ctr += 1
            return f"%v{ctr}"

        match expr:
            case rg.Func(fname=str(fname), args=rg.Args(args), body=body):
                argbuf = []
                for i in range(len(args)):
                    argbuf.append(f"%arg{i}: f64")
                    ctxargs.append(f"%arg{i}")

                argfmt = ", ".join(argbuf)
                sourcebuf.append(
                    f"func.func @{fname}({argfmt}) -> f64  attributes {{llvm.emit_c_interface}}"
                )
                sourcebuf.append("{")
                out = yield body
                sourcebuf.append(f"return {out} : f64")
                sourcebuf.append("}")
                return sourcebuf

            case rg.RegionEnd(
                begin=rg.RegionBegin() as begin,
                outs=str(outs),
                ports=ports,
            ):
                (yield begin)
                outputs = []
                for p in ports:
                    ctxportmap[p] = pv = yield p
                    outputs.append(pv)
                return tuple(outputs)

            case rg.RegionBegin(
                ins=ins,
                ports=ports,
            ):
                for p in ports:
                    ctxportmap[p] = yield p

            case rg.Unpack(val=source, idx=int(idx)):
                pv = yield source
                return pv[idx]

            case rg.IO():
                return "<<<IO>>>"
            case rg.ArgRef(idx=int(idx), name=str(name)):
                return ctxargs[idx]

            case rg.PyBinOpPure(op=op, lhs=lhs, rhs=rhs):
                lhsval = yield lhs
                rhsval = yield rhs
                inst = _handle_binop(op, lhsval, rhsval)
                var = fresh_name()
                sourcebuf.append(f"{var} = {inst}")
                return var

            case rg.PyCallPure(func=func, args=args):
                callee = yield func
                argvals = []
                for arg in args:
                    argvals.append((yield arg))
                match callee:
                    case "arith.sitofp":
                        typ = "i64 to f64"
                    case _:
                        typ = "f64"
                inst = f"{callee} {', '.join(argvals)} : {typ}"
                var = fresh_name()
                sourcebuf.append(f"{var} = {inst}")
                return var
            case rg.PyLoadGlobal(io=io, name=str(varname)):
                match varname:
                    case "flt":
                        return "arith.sitofp"

                    case "sqrt":
                        return "math.sqrt"
                    case _:
                        raise NotImplementedError(f"unknown global: {varname}")
            case rg.PyFloat(float(val)):
                var = fresh_name()
                inst = f"arith.constant {val:e} : f64"
                sourcebuf.append(f"{var} = {inst}")
                return var

            case rg.PyInt(int(val)):
                var = fresh_name()
                inst = f"arith.constant {val} : i64"
                sourcebuf.append(f"{var} = {inst}")
                return var

            case _:
                print("\n".join(sourcebuf))
                raise NotImplementedError(f"failed: {expr}")

    def _handle_binop(op: str, lhsval, rhsval):

        match op:
            case "+":
                res = f"arith.addf {lhsval}, {rhsval} : f64"
            case "-":
                res = f"arith.subf {lhsval}, {rhsval} : f64"
            case "*":
                res = f"arith.mulf {lhsval}, {rhsval} : f64"
            case "/":
                res = f"arith.divf {lhsval}, {rhsval} : f64"
            case _:
                raise NotImplementedError(op)

        return res

    ase.traverse(root, codegen)
    body_source = "\n".join(sourcebuf)

    wrapper_1d_loop = r"""
func.func @do_work(%arg0: memref<?xf64>, %arg1: memref<?xf64>) attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xf64>
  affine.for %arg2 = %c0 to %dim {
    %v = affine.load %arg0[%arg2] : memref<?xf64>
    %res = func.call @transformed_geglu(%v) : (f64) -> f64
    affine.store %res, %arg1[%arg2] : memref<?xf64>
  }
  return
}
    """

    return body_source + wrapper_1d_loop

```


## Start of GEGLU Approximation demo


Firstly, define our target function `geglu()` and helpers to abstract away `numpy`
calls:

```{code-cell} ipython3


def float32(num):
    return np.float32(num)

flt = float32

def tanh(x):
    return np.tanh(x)

def sqrt(x):
    return np.sqrt(x)

pi = np.pi

def geglu(a):
    result = (
        0.5
        * a
        * (
            flt(1)
            + tanh(sqrt(flt(2) / flt(pi)) * (a + flt(0.044715) * a**3))
        )
    )
    return result
```

### Get RVSDG representation

```{code-cell} ipython3
rvsdg_expr, dbginfo = rvsdg.restructure_source(geglu)
print(rvsdg.format_rvsdg(rvsdg_expr))
```

### Convert RVSDG to EGraph


```{code-cell} ipython3

memo = egraph_conversion(rvsdg_expr)

egfunc = memo[rvsdg_expr]
```

### Prepare for Equality Saturation

```{code-cell} ipython3
# Eval with Env
env = Env.nil()
env = env.nest(
    # Argument list
    valuelist(Value.IOState(), Value.Param(0))
)
root = GraphRoot(egfunc)

extra_statements = [
    DoPartialEval(env, egfunc),
]

```
### Application specific rules

```{code-cell} ipython3
def extra_ruleset():
    from egglog import (
        Unit,
        Vec,
        f64,
        function,
        i64,
        rewrite,
        rule,
        ruleset,
        set_,
        subsume,
        union,
    )

    import sealir.eqsat.rvsdg_eqsat as eg

    @function(cost=400)
    def Tanh(val: eg.Term) -> eg.Term: ...

    @function
    def Sqrt(val: eg.Term) -> eg.Term: ...

    @function
    def Flt(val: eg.Term) -> eg.Term: ...

    @function
    def Pi() -> eg.Term: ...

    @function(unextractable=True)
    def IsFloat(val: eg.Term) -> Unit: ...

    @ruleset
    def rule_cheats(uid: String, ins: String, argvec: Vec[Term], i: i64):
        yield rewrite(
            eg.Region(uid, ins, eg.TermList(argvec)).begin().get(i)
        ).to(
            argvec[i],
            # given
            i < argvec.length(),
        )

    @ruleset
    def facts(x: eg.Term, y: eg.Term, z: eg.Term, fval: f64, io: eg.Term):
        yield rule(eq(x).to(eg.Term.Param(0))).then(
            set_(IsFloat(x)).to(Unit())
        )

        yield rule(eq(x).to(eg.Term.LiteralF64(fval))).then(
            set_(IsFloat(x)).to(Unit())
        )

        for fn in [Tanh, Sqrt, Flt]:
            yield rule(eq(x).to(fn(y))).then(set_(IsFloat(x)).to(Unit()))

        yield rewrite(eg.Term.LoadGlobal(io, "pi")).to(Pi())

        yield rule(eq(x).to(Pi())).then(set_(IsFloat(x)).to(Unit()))

    @ruleset
    def rules_simplify_python(
        io: eg.Term, argvec: Vec[eg.Term], lhs: eg.Term, rhs: eg.Term
    ):
        def shortcut_call(call_target: str, func_target):
            return rule(
                call := eg.Term.Call(
                    func=eg.Term.LoadGlobal(io=io, name=call_target),
                    io=io,
                    args=eg.TermList(argvec),
                ),
                call.getPort(0),
                call.getPort(1),
                eq(argvec.length()).to(i64(1)),
            ).then(
                union(call.getPort(1)).with_(func_target(argvec[0])),
                union(call.getPort(0)).with_(io),
                subsume(call),
            )

        yield shortcut_call("tanh", Tanh)
        yield shortcut_call("sqrt", Sqrt)
        yield shortcut_call("flt", Flt)

        # Remove IO
        def shortcut_io(fnio, fnpure):
            return rule(
                call := fnio(io, lhs, rhs),
                IsFloat(lhs),
                IsFloat(rhs),
            ).then(
                union(res := call.getPort(1)).with_(fnpure(lhs, rhs)),
                union(call.getPort(0)).with_(io),
                set_(IsFloat(res)).to(Unit()),
                subsume(call),
            )

        yield shortcut_io(eg.Term.AddIO, eg.Term.Add)
        yield shortcut_io(eg.Term.MulIO, eg.Term.Mul)
        yield shortcut_io(eg.Term.DivIO, eg.Term.Div)
        yield rule(
            call := eg.Term.PowIO(io, lhs, rhs),
            IsFloat(lhs),
        ).then(
            union(res := call.getPort(1)).with_(eg.Term.Pow(lhs, rhs)),
            union(call.getPort(0)).with_(io),
            set_(IsFloat(res)).to(Unit()),
        )

        yield rewrite(Flt(rhs)).to(rhs, IsFloat(rhs))

    @ruleset
    def pade44_tanh_expansion(x: Term, y: Term, z: Term):
        flt = lambda f: eg.Term.LiteralF64(f64(float(f)))
        liti64 = eg.Term.LiteralI64
        pow = eg.Term.Pow
        mul = eg.Term.Mul
        add = eg.Term.Add
        div = eg.Term.Div
        yield rewrite(Tanh(x)).to(
            div(
                add(mul(flt(10), pow(x, liti64(3))), mul(flt(105), x)),
                add(
                    add(
                        pow(x, liti64(4)), mul(flt(45), pow(x, liti64(2)))
                    ),
                    flt(105),
                ),
            )
        )

    @ruleset
    def expand_pow(
        x: eg.Term, y: eg.Term, z: eg.Term, term: eg.Term, i: i64
    ):
        yield rule(
            eq(term).to(eg.Term.Pow(x, eg.Term.LiteralI64(i))),
            IsFloat(x),
            i > i64(1),
        ).then(
            union(term).with_(
                eg.Term.Mul(
                    x, eg.Term.Pow(x, eg.Term.LiteralI64(i - i64(1)))
                )
            )
        )
        yield rewrite(eg.Term.Pow(x, eg.Term.LiteralI64(1))).to(
            x,
            # given
            IsFloat(x),
        )
        yield rewrite(eg.Term.Pow(x, eg.Term.LiteralI64(0))).to(
            eg.Term.LiteralF64(1.0),
            # given
            IsFloat(x),
        )

    return (
        rule_cheats
        | facts
        | rules_simplify_python
        | pade44_tanh_expansion
        | expand_pow
    )

```
### Equality Saturation

```{code-cell} ipython3
egraph = equality_saturation(root, *extra_statements, ruleset=extra_ruleset())

```

The result above is free of `Tanh()`.

### Run Custom Cost-based Extraction. (EGraph -> RVSDG)


```{code-cell} ipython3
# Extraction
from sealir.eqsat.rvsdg_extract import EGraphToRVSDG

class ExtendedConverter(EGraphToRVSDG):
    """
    Provide application specific EGraph -> RVSDG conversion
    """

    def handle_Term(self, op: str, children: dict | list, grm):
        import sealir.rvsdg.grammar as rg

        match op, children:
            case "Flt", {"val": val}:
                io = grm.write(rg.IO())  # dummy
                fn_flt = grm.write(rg.PyLoadGlobal(io=io, name="flt"))
                return grm.write(
                    rg.PyCallPure(func=fn_flt, args=tuple([val]))
                )

            case "Sqrt", {"val": val}:
                io = grm.write(rg.IO())  # dummy
                fn_flt = grm.write(rg.PyLoadGlobal(io=io, name="sqrt"))
                return grm.write(
                    rg.PyCallPure(func=fn_flt, args=tuple([val]))
                )

            ## Commented out because this should not be used
            # case "Tanh", {"val": val}:
            #     io = grm.write(rg.IO())  # dummy
            #     fn_flt = grm.write(rg.PyLoadGlobal(io=io, name="tanh"))
            #     return grm.write(
            #         rg.PyCallPure(func=fn_flt, args=tuple([val]))
            #     )

            case "Pi", {}:
                return grm.write(rg.PyFloat(np.pi))

            case _:
                return NotImplemented

cost, extracted = egraph_extraction(
    egraph,
    rvsdg_expr,
    converter_class=ExtendedConverter,
)
print(ase.as_tuple(extracted, depth=5))
print("cost =", cost)
print(rvsdg.format_rvsdg(extracted))

```

### Test extracted code with LLVM Python CAPI backend

```{code-cell} ipython3
ns = {
    "flt": flt,
    "pi": pi,
    "sqrt": sqrt,
    "tanh": tanh,
}
# LLVM output

cg = llvm_codegen(extracted, ns)

arg = 0.315
got = cg(arg)
expect = geglu(arg)
assert got == expect

```

### Use MLIR to get OMP ufunc version

```{code-cell} ipython3

# MLIR output

mlir_src = generate_mlir(extracted)
print(mlir_src)
```


```{code-cell} ipython3
comp = Compiler()
addr = comp.run_backend(mlir_src, symbol="_mlir_ciface_do_work")
print(comp.output_logs[-1]["mlir_optimized"])
```



```{code-cell} ipython3
# Run the code
arr = np.linspace(0, 1, 1000)
got = np.zeros_like(arr)
call_ufunc(addr, args=[arr], out=got)

expect = np.vectorize(geglu)(arr)
np.testing.assert_allclose(got, expect, rtol=1e-6)

diffnorm = np.linalg.norm(got - expect) / np.linalg.norm(expect)
print("L-norm", diffnorm)
```