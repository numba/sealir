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


```{code-cell} ipython3
from sealir import ase
```

```{code-cell} ipython3
with ase.Tree() as tree:
    a = ase.expr("num", 1)
    b = ase.expr("num", 2)
    c = ase.expr("add", a, b)
```

```{code-cell} ipython3
print(tree.dump())
```

```{code-cell} ipython3
tree.render_dot()
```

```{code-cell} ipython3
with tree:
    d = ase.expr("sub", c, b)
    e = ase.expr("mul", b, d)
```

```{code-cell} ipython3
print(tree.dump())
```

```{code-cell} ipython3
tree.render_dot()
```

```{code-cell} ipython3
from sealir.rewriter import TreeRewriter


class RewriteCalcMachine(TreeRewriter[ase.Expr]):
    def rewrite_add(self, lhs, rhs):
        [x] = lhs.args
        [y] = rhs.args
        return ase.expr("num", x + y)

    def rewrite_sub(self, lhs, rhs):
        [x] = lhs.args
        [y] = rhs.args
        return ase.expr("num", x - y)

    def rewrite_mul(self, lhs, rhs):
        [x] = lhs.args
        [y] = rhs.args
        return ase.expr("num", x * y)
```

```{code-cell} ipython3
calc = RewriteCalcMachine()
e.apply_bottomup(calc)
print(e.tree.dump())
```

```{code-cell} ipython3
tree.render_dot()
```

```{code-cell} ipython3
result = calc.memo[e]
print(result)
print(result.str())
```

```{code-cell} ipython3

```
