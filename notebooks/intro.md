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

# Introduction

 SealIR (🦭 IR) is a compiler infrastructure for the development of domain-specific compilers using S-expression and lambda calculus as the building block.

```{code-cell} ipython3
from sealir import ase
```

## The Tape and S-expressions


SealIR's primary storage mechanism is the `Tape`, an append-only structure designed for efficient S-expression storage. S-expressions are always recorded in a tape, which must be "opened" using a context manager before it becomes writable. Once opened, `ase.expr` calls record new S-expressions into the active tape.

Here's an example of using the Tape:

```{code-cell} ipython3
with ase.Tape() as tape:
    a = ase.expr("num", 1)
    b = ase.expr("num", 2)
    c = ase.expr("add", a, b)
```

The tape's contents can be inspected using the dump() method:

```{code-cell} ipython3
print(tape.dump())
```

This dump provides a low-level view of the tape's contents in a tabular format. Each row represents a record corresponding to an S-expression. The first record is reserved as "NULL". Tokens are printed in-place, while references to other S-expressions are shown as `<N>`, where `N` is the row number (handle) of the referenced S-expression.

+++

For a more visual representation, the tape can be rendered using Graphviz:

```{code-cell} ipython3
tape.render_dot()
```

This visualization assumes the last node to be the root of the S-expression tree.

+++

To extend the S-expression structure, simply reopen the tape:

```{code-cell} ipython3
with tape:
    d = ase.expr("sub", c, b)
    e = ase.expr("mul", b, d)
```

```{code-cell} ipython3
print(tape.dump())
```

```{code-cell} ipython3
tape.render_dot()
```

Individual S-expressions can be printed in two ways:

- ``str(sexpr)``: Displays the structure without unpacking other S-expressions.
- ``sexpr.str()``: Shows the entire expression tree.

```{code-cell} ipython3
print(str(e))
print(e.str())
```

## A Rewrite System (``TreeRewriter``)

At the core of SealIR is a **recursion-free** S-expression rewrite system. This system allows for efficient manipulation and transformation of S-expressions.

Let's explore this concept by implementing a simple calculator that evaluates arithmetic expressions:

```{code-cell} ipython3
from sealir.rewriter import TreeRewriter


class RewriteCalcMachine(TreeRewriter[ase.Expr]):
    def rewrite_add(self, orig, lhs, rhs):
        [x] = lhs.args
        [y] = rhs.args
        return ase.expr("num", x + y)

    def rewrite_sub(self, orig, lhs, rhs):
        [x] = lhs.args
        [y] = rhs.args
        return ase.expr("num", x - y)

    def rewrite_mul(self, orig, lhs, rhs):
        [x] = lhs.args
        [y] = rhs.args
        return ase.expr("num", x * y)
```

### Applying Rewrite Rules: ``Expr.apply_bottomup`` and ``Expr.apply_topdown``

+++

To utilize the rewrite pass, we can use the ``.apply_bottomup()`` method. This method traverses the S-expression tree from leaves to root, ensuring that inner-most expressions are processed first.

Due to the append-only nature of the tape, older S-expressions always appear earlier than newer ones. The bottom-up traversal can simply runs from the first record in the tape to the last. By default, the method will compute the reachable set from the root of the expression tree and only apply the pass on to those in the set. It is possible to run the pass on all nodes irregardless of the reachability by setting keyword `reachable=None`. It's important to note that this will process all nodes in the tape, even those not part of the current expression tree. Therefore, it's crucial that rewrites are side-effect free.

For cases where you need to avoid processing unreachable nodes, use ``.apply_topdown()``. This method traverses from root to leaves, skipping unreachable nodes, but at the cost of slower performance.

Here's an example of applying the calculator rewrite rules:

```{code-cell} ipython3
calc = RewriteCalcMachine()
with e.tape:
    e.apply_bottomup(calc)
print(e.tape.dump())
```

### Accessing Rewrite Results: ``TreeRewriter.memo``

To retrieve the result of a rewrite operation, use the ``.memo`` attribute of the rewrite pass. This attribute maps original S-expressions to their replacements:

```{code-cell} ipython3
result = calc.memo[e]
print(result)
print(result.str())
```

The ``.memo`` attribute contains replacement mappings for all processed S-expressions:

```{code-cell} ipython3
calc.memo
```

### Tracking Rewrite History

+++

The ``TreeRewriter`` class maintains a history of each rewrite operation. Records with ``.md.rewrite`` are metadata records for rewrites, containing:

- The rewriter class name
- The original record
- The replacement record

You can visualize this history using the ``render_dot()`` method:

```{code-cell} ipython3
tape.render_dot()
```

(Note: Grey nodes in the graph represent tape records that are no longer reachable from the "start".)

+++

To include metadata in the visualization, set ``show_metadata=True``:

```{code-cell} ipython3
tape.render_dot(show_metadata=True)
```

## Manual Traversal Techniques

The `Expr` class in SealIR provides several methods for manually traversing the S-expression tree. These methods offer fine-grained control over tree exploration and analysis.

+++

### ``Expr.walk_descendants``

This method allows you to iterate over the expression tree in a top-down manner, yielding both the ancestry and the current `Expr`:

```{code-cell} ipython3
for parents, cur in e.walk_descendants():
    print('(', ' . '.join([p.head for p in parents]), ')', '--->', cur.str())
```

### ``Expr.walk_parents()``

Use this method to iterate over all parents (users) of a given expression:

```{code-cell} ipython3
list(a.walk_parents())
```

```{code-cell} ipython3
list(b.walk_parents())
```

### ``Expr.contains(Expr)``

This method tests whether an S-expression tree contains another S-expression:

```{code-cell} ipython3
for p in b.walk_parents():
    print(f"{p.str()} contains {b.str()}:", p.contains(b))
```

```{code-cell} ipython3
print(f'c = {c.str()}')
print(f'a = {a.str()}')
print("a in c?", c.contains(a))
```

It's important to note that .contains() tests for record identity, not structural equality:

```{code-cell} ipython3
with tape:
    x = ase.expr("num", 1)
c.contains(x)
```

### ``Expr.search_parents(predicate)``

This method allows you to iterate over parents that satisfy a certain predicate:

```{code-cell} ipython3
for p in b.search_parents(lambda x: x.head == "mul"):
    print(p.str())
```

### Using ``match`` on S-expression

``Expr`` supports the use of Python's ``match`` statement.

Matching simple expressions:

```{code-cell} ipython3
match e:
    case ase.Expr("mul", (x, y)):
        print(f"Multiplication: {x} * {y}")
```

Matching nested expressions:

```{code-cell} ipython3
match d:
    case ase.Expr("sub", (ase.Expr("add", (x, y)), ase.Expr("num", (2,)))):
        print(f"Subtraction: ({x} + {y}) - 2")
```

```{code-cell} ipython3

```
