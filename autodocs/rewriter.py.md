## class `MdRewriteLayout`
## function `insert_metadata`

Inserts metadata into an `ase.SExpr` node, indicating that it has been rewritten using the given `rewriter`. The metadata includes the rewritten node and the original node.

**Args:**

* `rewriter`: The name of the rewriter that performed the transformation.
* `repl`: The rewritten `ase.SExpr` node.
* `orig`: The original `ase.SExpr` node.

**Note:**

* The `orig` and `repl` nodes must be from the same tape.
* The inserted metadata is represented by an expression with the head `".md.rewrite"` and the arguments `MdRewriteLayout(rewriter, repl, orig)`.
## function `insert_metadata_map`

Inserts metadata into the `repl` ASTs within a dictionary of mappings (`memo`). The metadata contains the rewriter name and the original AST (`orig`).

**Arguments:**

* `memo`: A dictionary where keys are original ASTs and values are the transformed ASTs.
* `rewriter_name`: The name of the rewriter that produced the transformed ASTs.

**Purpose:**

The function iterates over the `memo` dictionary and calls the `insert_metadata` function for each entry where the value is an AST (`isinstance(repl, ase.SExpr)`). The metadata inserted by `insert_metadata` includes the rewriter name, the transformed AST (`repl`), and the original AST (`orig`).

**Example Usage:**

```python
memo = {orig_ast: transformed_ast for ...}  # Create the memo dictionary
insert_metadata_map(memo, "my_rewriter")  # Insert metadata using the rewriter name "my_rewriter"
```
## function `metadata_find_original`

Finds the original metadata node for a given node.

**Arguments:**

* `node`: The node to find the original metadata for.
* `loc_test`: A function that tests whether a node is the location of the metadata.

**Returns:**

* The original metadata node if found, or `None` otherwise.

**Usage:**

```python
out = metadata_find_original(node, loc_test)
if out is not None:
    return out._args[-1]
else:
    return None
```
## class `TreeRewriter`
### function `__init__`

Initializes the `Expr` object.

The `__init__` method initializes an `Expr` object with the given `ase.SExpr` expression. It stores the expression, handles and slots in the object. It also sets up a memoization dictionary (`memo`) to avoid duplicate references.
### function `visit`

The `visit` function is a method of the `TreeRewriter` class that is used to visit each node in an abstract syntax tree (AST). It performs the following functionalities:

- Checks if the current node has already been visited. If so, it skips the node and returns.
- Calculates the result of the node using the `_dispatch` method.
- Stores the result in the `memo` dictionary for future reference.
- If `flag_save_history` is set to `True`, it creates metadata that maps the replaced node back to the original node. This is used to preserve the history of the AST transformation.
### function `_dispatch`

The `_dispatch` function takes an `ase.SExpr` as input and performs the following steps:

- Checks if any of the arguments in the `orig` SExpr are also SExprs.
- If so, it looks up these SExprs in the `memo` dictionary and replaces them with the corresponding values.
- It then calls the `_default_rewrite_dispatcher` function with the updated SExpr and other parameters.
- Finally, it returns the result of the `_default_rewrite_dispatcher` function.

The purpose of this function is to handle SExprs where some of the arguments may need to be replaced with values from the `memo` dictionary.
### function `passthru`

Returns the current state of the `_passthru_state` attribute. This is used to preserve the original AST structure when rewriting Python code. If the `updated` flag is set, the `_passthru_state` is set to a lambda function that replaces the original AST with the updated AST. Otherwise, it is set to a lambda function that returns the original AST.
### function `rewrite_generic`

The `rewrite_generic` function is a default implementation for rewriting expressions. It takes three arguments:

* `orig`: The original expression.
* `args`: A tuple of arguments to replace the children of `orig`.
* `updated`: A boolean indicating whether the children of `orig` have been updated.

The function returns either the original expression if its children are unmodified or a new node with updated children if `updated` is set to `True`.
### function `_default_rewrite_dispatcher`

The `_default_rewrite_dispatcher` function is responsible for dispatching rewrite operations based on the head of the input SExpr. It handles two cases:

* If a specific rewrite function is defined for the SExpr's head (e.g., `rewrite_call`), it is executed.
* If no specific rewrite function is found, the `rewrite_generic` function is used to handle the generic case.

The function returns the rewritten SExpr or the original SExpr if no rewrite is applied.
