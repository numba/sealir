## class `ForLoopInfo`
## class `PythonAnalysis`
### function `find_for_loops`

The `find_for_loops` function uses the `uda.search_use_chain` method to search for for loops in the given code. It then yields `ForLoopInfo` objects for each loop it finds.

**Functionality:**

* Uses `uda.search_use_chain` to locate the relevant code structures for for loops.
* Creates `ForLoopInfo` objects to represent the found loops.
* Yields these objects as the result of the function.
### function `callee_is_global_load`

Checks if the callee of a call node is a global load with the given name.

**Parameters:**

* `node`: The call node to check.
* `name`: The name of the global variable to check for.

**Returns:**

* `True` if the callee is a global load with the given name, `False` otherwise.
### function `match_range_call`

Checks if the given `node` represents a call to the `range()` function.

**Functionality:**

- The function uses a `match` expression to extract the `callee` from the `node`.
- It then checks if the `callee` is a global load with the name `range`.
- If both conditions are met, the function returns `True`; otherwise, it returns `False`.
### function `match_iter_node`

Checks if the given node is an expression node with a global load call to the `iter` function.

**Parameters:**

* `prefixes`: A list of prefix nodes.
* `x`: The node to check.

**Returns:**

* `True` if the node is a valid iterator node, `False` otherwise.
### function `match_next_node`

Checks if the given node represents a call to the `next()` global function. It performs the following checks:

* The node is an expression node.
* The expression is a Python call node.
* The call is a global load of the `next` variable.
* The call is made by the sole user of the iteration node.
```
### function `match_endloop_compare`

Checks if a user's comparison expression matches a sentinel.

**Parameters:**

- `prefixes`: A list of prefixes.
- `x`: The user to check.

**Returns:**

- `True` if the user's comparison expression matches the sentinel, `False` otherwise.
```
