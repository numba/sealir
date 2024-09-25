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

# Lambda Calculus Encoding and Grammar

## Intro

SealIR provides a builtin lambda calculus encoding grammar that contains the bare bone of lambda calculus semantic. 

The lambda calculus grammar:

```
expr : (Lam body: expr)
     | (App arg: expr 
            lam: expr)
     | (Arg idx: int)
     | (Pack *values: expr)
     | (Unpack idx: int 
               tup: expr)
```

It uses De Bruijn index to achieve nameless reference to values. A `(Arg N)` refers to the Nth enclosing lambda abstraction. 


```{code-cell} ipython3
from sealir import lam, ase, grammar
```

Grammar can be extended to include user defined node types. To do so, user can define a new ``grammar.Rule`` class as the root of the grammar and subclass the root with additional rules for specific node types.

```{code-cell} ipython3
class _Val(grammar.Rule):
    pass

class Num(_Val):
    # (Num int)
    value: int

class _BinOp(_Val):
    # (_BinOp lhs rhs)
    lhs: ase.SExpr
    rhs: ase.SExpr

class Add(_BinOp): ...
class Sub(_BinOp): ...
class Mul(_BinOp): ...

# Build the grammar that combines the builtin lambda-calculus grammar
class Grammar(grammar.Grammar):
    start = lam.LamGrammar.start | _Val
```

``grammar.Grammar`` provides some checking to what is written into the tape. To write with a grammar object, wrap the tape with it:

```{code-cell} ipython3
grm = Grammar(ase.Tape())

@lam.lam_func(grm)
def func_body(x):
    a = grm.write(Mul(lhs=x, rhs=grm.write(Num(2))))
    b = grm.write(Add(lhs=a, rhs=x))
    c = grm.write(Sub(lhs=a, rhs=b))
    d = grm.write(Mul(lhs=b, rhs=c))
    return d
```

```{code-cell} ipython3
func_body
```

```{code-cell} ipython3
print(lam.format(func_body))
```

```{code-cell} ipython3
print(func_body._tape.dump())
```

```{code-cell} ipython3
func_body._tape.render_dot()
```

## Abstraction Pass

The ``lam`` module provides some common passes to work with lambda-calculus encoding. One such pass is the abstraction pass, which introduce lambda abstraction to avoid multiple references to the same ``SExpr`` node. 

```{code-cell} ipython3
func_body = lam.run_abstraction_pass(grm, func_body)
```

```{code-cell} ipython3
print(lam.format(func_body))
```

```{code-cell} ipython3
print(func_body._tape.dump())
```

```{code-cell} ipython3
func_body._tape.render_dot()
```

## Beta-reduction

Beta-reduction is a fundamental operation in lambda calculus. It is the process of substituting the argument of a lambda abstraction with an expression. SealIR provides a builtin beta-reduction pass that can be used to reduce a lambda abstraction:

```{code-cell} ipython3
with grm:
    app = grm.write(lam.App(arg=grm.write(Num(123)), 
                            lam=func_body))

print(lam.format(app))
```

```{code-cell} ipython3
print(lam.format(lam.beta_reduction(app)))
```
