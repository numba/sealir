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

# Lambda

```{code-cell} ipython3
from sealir.lam import LamBuilder
```

```{code-cell} ipython3
lambar = LamBuilder()
```

```{code-cell} ipython3

@lambar.lam_func
def func_body(x):
    a = lambar.expr("mul", x, lambar.expr("num", 2))
    b = lambar.expr("add", a, x)
    c = lambar.expr("sub", a, b)
    d = lambar.expr("mul", b, c)
    return d
```

```{code-cell} ipython3
print(lambar.format(func_body))
```

```{code-cell} ipython3
print(lambar._tree.dump())
```

```{code-cell} ipython3
lambar._tree.render_dot()
```

```{code-cell} ipython3
func_body = lambar.run_abstraction_pass(func_body)
```

```{code-cell} ipython3
print(lambar.format(func_body))
```

```{code-cell} ipython3
print(lambar._tree.dump())
```

```{code-cell} ipython3
lambar._tree.render_dot()
```

```{code-cell} ipython3

```
