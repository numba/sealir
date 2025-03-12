## class `RegionInfo`
### function `__getitem__`

Returns the term at the specified index.

- Takes an integer `idx` as input.
- Returns an `eg.Term` object.
- Raises an `IndexError` if the index is out of bounds.
## class `WrapIO`
### function `__getitem__`

Retrieves the `io` or `val` attribute of the `WrapTerm` object based on the provided index.

* **Index 0:** Returns the `io` attribute.
* **Index 1:** Returns the `val` attribute.
* **Raises `IndexError`:** If the provided index is not 0 or 1.
## class `WrapTerm`
### function `__getitem__`

Returns the port at the given index in the `term` attribute of the `WrapTerm` object.
```python
def __getitem__(self, idx: int) -> eg.Term:
    return self.term.getPort(idx)
```
## function `egraph_conversion`

Converts an abstract syntax tree (AST) into an egraph representation.

**Functionality:**

* Takes an `SExpr` object as input.
* Uses a coroutine-based traversal to generate an egraph term.
* The egraph term represents the AST structure in a graph-like format.
* Each node in the graph corresponds to a specific AST node type.
* Edges connect nodes based on the relationships between AST nodes.
* The `node_uid` function generates a unique identifier for each AST node.
* The `coro` function handles the traversal logic, converting each AST node into an egraph term.
* The `RegionInfo` class stores information about the region nodes in the AST.
* The `WrapTerm` class provides a convenient way to access the ports of an egraph term.

**Usage:**

```python
# Example usage:
root = SExpr(...)  # Create an AST tree
egraph = egraph_conversion(root)  # Convert the AST to an egraph term
```
