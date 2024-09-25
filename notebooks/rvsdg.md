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

# RVSDG? Numba Intermediate Language (NIL)?

## Intro

SealIR provides a builtin Regionalized Value-State Dependence Graph (RVSDG) encoding via the Lambda-calculus grammar (``sealir.lam``) with extension to support structured control flow (SCF). By combining pure functional semantic and SCF, we get an encoding for RVSDG in lambda calculus S-expression.

``numba-rvsdg`` package provides the restructured Python AST. From that we get a AST that only uses a subset of the Python control flow constructs that are allowed in RVSDG; e.g. ``while`` (that behaves like ``do...while``) and ``if``.

Once we have the restructured Python AST (Python in SCF), we need to convert the program to value-state dependence. We do that by via multiple passes and intermediate encoding.

First the program is translated to a let-binding function form with explicit encoding of the IO state. This let-binding language has grammar of:

```
expr  : (Let name: str               # let name := value in body
             value: expr 
             body: expr)             
      | scf-expressions              # structured control flow
      | py-expressions               # any python expressions
      | lambda-calculus-expressions  # extend from lambda grammar
```

The translation to this grammar is straightforward. Every Python statement is translated to a let-binding expression. In the overly-simplified case, Python statements like:

```python
a = b()
c = a()
return a + c
```

is translated to:

```
Let a := b() in
Let c := a() in
(py_add a c)
```

The above is overly-simplified because it ignored IO state. Python is imperative and every operation can have side effects. So we need to keep track of the IO state. A translation with IO state is:

```
Let tmp_call_b := b(iostate) in
Let iostate    := (Unpack 0 tmp_call_b) in
Let a          := (Unpack 1 tmp_call_b) in
Let tmp_call_a := a(iostate) in
Let iostate    := (Unpack 0 tmp_call_a) in
Let c          := (Unpack 1 tmp_call_a) in
(py_add iostate a c)   # returns (iostate, value)
```

In short, every Python statement is translated into a let-binding expression that unpacks the outgoing value-states. By explicitly encoding IO state, all Python statement must at least return the IO state. If there are any modified variables, they are also returned.

The ``while`` construct is unique because of the loop-back variables. The loop body is equivalent to an expression that take and return the packed loop-back variables. The loop-back variables are `(loop-condition, other-modified-variables...)`.

Once everything is in the let-binding form, translation into let-free lambda calculus is done via De Bruijn index to make nameless reference to the original let-bound names. Every let-binding is translated into a lambda abstraction and an apply. For example, the let-binding:

```
function (b):
    Let a := b() in
    Let c := a() in
    (py_add a c)
```
is translated to:
```
(Lam                                            # function (b)
    (App (call (Arg 0))                         # let a := b() in
    (Lam
        (App (call (Arg 0)) )                   # let c := a() in
            (Lam (py_add (Arg 1) (Arg 0))       # (py_add a c)
        )
    )
  )
)
```

+++

## Examples of the RVSDG-ization

```{code-cell} ipython3
from pprint import pprint
from sealir import lam, ase
from sealir.rvsdg import restructure_source

def ifelse(a, b):
    if a < b:
        return a
    else:
        return c

pprint(ase.as_tuple(restructure_source(ifelse), -1))
```

```{code-cell} ipython3
def forloop(a, b, c):
    for i in range(a, b):
        c += i
    return c

node = restructure_source(forloop)
pprint(ase.as_tuple(node, -1))
```

```{code-cell} ipython3
node._tape.render_dot(only_reachable=True)
```

```{code-cell} ipython3

```

## Additional Notes on RVSDG and Lambda Calculus

### Regions and Lambda 

RVSDG defines a graph of values and states that are contained inside regions.
The concept of regions maps to lambda abstractions. Each `(Lam body)` has type of `a->b`, that takes some type `a` and returns some type `b`. Chained lambda abstractions are like nested regions. A `(Lam (Lam (Lam (Pack (Arg 0) (Arg 1) (Arg 2))))` has type `a->b->c->Pack(a, b, c)`. So it can be view as a region that takes three inputs and outputs a packed value.

### Converting Python Code Blocks

A Python code block is a sequence of statements. As mentioned before, statements are mapped to let-binding expressions. The let-binding expressions are translated to lambda abstractions. 

Code blocks can also be seen as regions. The inputs to the region are variables that are used but not defined locally. The outputs are all variables that are modified/defined locally.
