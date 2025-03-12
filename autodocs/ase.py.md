## class `MalformedContextError`
## class `HeapOverflow`
## class `NotFound`
## class `HandleSentry`
### function `__repr__`

The `__repr__` function converts an object to a string representation in a Python-like syntax. It takes no arguments and returns a string containing the object's name and field values.

**Functionality:**

* Obtains the object's name from the `_sexpr_head` field.
* Uses the `_get_field_values()` method to get a dictionary of field names and their values.
* Returns a string in the format `<object_name>(field1=value1, field2=value2, ...)` where each field is represented as `field=value`.
## class `Tape`
### function `__init__`

Initializes a new instance of the `HandleSentry` class.
- It sets up internal data structures to track tokens and handles.
- It creates a map from token types and values to handles.
- It initializes the heap with a single None token.
- It sets up a lambda function to perform downcasting on values.
### function `__len__`

Returns the number of records stored in the heap. This value is updated every time a new record is added using the `write_end()` method.
### function `heap_size`

Returns the size of the heap as an integer.

```python
def heap_size(self) -> int:
    return len(self._heap)
```
### function `__enter__`

Increments the `_open_counter` by 1 and returns `self`. This method is typically used in a context manager to indicate that the context is being entered.
### function `__exit__`

This function is called when an object goes out of scope. It decrements the `_open_counter` by 1. If the counter reaches 0, it raises a `MalformedContextError` with the message "malformed stack: top is not self". This ensures that nested contexts are properly closed and prevents misuse of the context manager.
### function `iter_expr`

Returns an iterator of `SExpr` objects for the current record.

The function uses a `TapeCrawler` to navigate the tape and converts each record to an `SExpr` object using `to_expr()`.
### function `expr`

Creates an `Expr` object by calling `BasicSExpr._write()` with the provided `head` and `args`. This function is the main API for creating `Expr` objects.

**Usage:**

```python
expr("head", *args)
```

**Parameters:**

* `head`: The head of the `Expr` object.
* `args`: The arguments of the `Expr` object.

**Returns:**

* An `SExpr` object.
### function `dump_raw`

Dump the raw representation of the tape, including the heap and tokens.

The `dump_raw()` function creates a string containing the following information:

* A header for the heap.
* A list of each element in the heap, with an index and a representation of the object.
* A header for the tokens.
* A formatted representation of the tokens using `pformat()`.

This information provides a comprehensive overview of the tape's internal state.
### function `dump`

Dump the contents of a tape, including the position and contents of each record.

The function performs the following steps:

- Creates a `TapeCrawler` object to navigate the tape.
- Appends the tape's ID to the output buffer.
- Appends the current position and contents of the tape head to the buffer.
- Iterates through the records on the tape and appends their handles, heads, and arguments to the buffer.

The output of the function is a string containing the formatted tape contents.
### function `render_dot`

Generates a Graphviz DOT graph representation of the state of the tape.

**Parameters:**

* `gv`: An instance of the `Graphviz` library's `Digraph` class.
* `show_metadata`: Whether to include metadata in the graph. Defaults to `False`.
* `only_reachable`: Whether to only include reachable nodes in the graph. Defaults to `False`.

**Functionality:**

The `render_dot()` function creates a Graphviz DOT graph that represents the current state of the tape. It iterates through the tape's records and creates nodes for each record. The nodes represent the record's index, head, and arguments.

* **Nodes:**
    * Each node has a label that includes the record's index, head, and arguments.
    * Reachable nodes are highlighted in black.
    * Non-reachable nodes are highlighted in light gray.
* **Edges:**
    * Edges connect each node to its child nodes.
    * The weight of each edge is set to 5 for reachable nodes and 1 for non-reachable nodes.

**Example Usage:**

```python
gv = Graphviz.Digraph()
tape = Tape()  # Assuming Tape is defined elsewhere
dot_graph = tape.render_dot(gv=gv)
```

**Note:**

* The `metadata_prefix` variable is assumed to be defined elsewhere in the codebase.
* The `TapeCrawler` class is assumed to be defined elsewhere in the codebase.
### function `get`

Returns the item at the specified position in the heap.

**Arguments:**

* `pos`: The position of the item to retrieve.

**Returns:**

* The item at the specified position.

**Assertions:**

* `pos >= 0`: The position must be non-negative.
### function `load`

Loads a slice of handles from the heap.

**Arguments:**

* `start`: The starting handle of the slice.
* `stop`: The stopping handle of the slice.

**Returns:**

A tuple of handles.
### function `last`

Returns the handle of the last element in the heap. The complexity is O(1).

**Usage:**

```python
heap.last()
```

**Returns:**

- `handle_type`: The handle of the last element in the heap.
### function `index`

Searches for the first occurrence of the target value in the heap starting from the given start position.

**Arguments:**

- `target`: The value to search for.
- `startpos`: The starting position in the heap.

**Returns:**

- The position of the first occurrence of the target value in the heap.

**Raises:**

- `NotFound`: If the target value is not found in the heap.
### function `rindex`

The `rindex` function searches for the last occurrence of a given `target` in the heap starting from the `startpos`.

- It iterates backward from `startpos` until it finds the first occurrence of `target`.
- If it doesn't find `target`, it raises a `NotFound` exception.
- The function returns the position of the last occurrence of `target`.
### function `read_head`

Returns the string value stored at the specified handle.

**Arguments:**

* `handle`: The handle of the string to read.

**Returns:**

* A string containing the value stored at the specified handle.

**Usage:**

```python
string_handle = ...  # Get the handle of the string
string_value = reader.read_head(string_handle)  # Read the string value
```
### function `read_args`

Reads arguments from a handle.

- Takes a `handle_type` as input.
- Reads the values associated with the handle.
- Returns a `tuple` of `value_type` values.
- Raises a `HeapOverflow` exception if the handle is invalid.
### function `read_value`

Reads the value at the given handle.

- If the handle is less than or equal to 0, it reads the token at the handle using `_read_token`.
- If the handle is between 0 and `HandleSentry.BEGIN`, it creates a `BasicSExpr` with the handle.
- If the handle is greater than `HandleSentry.BEGIN`, it raises a `MalformedContextError`.

**Usage:**

- `read_head(handle)`: Reads the head of the SExpr at the given handle.
- `expr(head, *args)`: Creates an `SExpr` with the given head and arguments.
### function `_read_token`

Retrieves the token at the given index from the tokens list.
- Asserts that the index is less than or equal to 0.
- Returns the token at the specified index.
### function `write`

The `write()` function allows users to write data to a tape by providing a header and an arbitrary number of arguments. It handles both string and SExpr arguments.

**Arguments:**

* `head`: The header of the data being written.
* `args`: A tuple of arguments to be written.

**Returns:**

* `handle_type`: A handle to the newly created data.

**Functionality:**

* Begins a new write operation.
* Writes the header token.
* Iterates through the arguments and writes each token or SExpr handle.
* Ends the write operation.
* Returns the handle of the newly created data.

**Usage:**

```python
# Example usage:
handle = tape.write("hello", "world", SExpr("expression"))
```
### function `write_ref`

Appends the given reference `ref` to the internal heap.

**Preconditions:**

* `0 < ref < HandleSentry.BEGIN`

**Postconditions:**

* The reference `ref` is appended to the `_heap` list.
### function `write_token`

The `write_token` function takes a token of type `token_type` as input and adds it to the internal data structure. It handles various types of tokens, including integers, strings, and floats. If an invalid token type is provided, the function raises a `TypeError` exception.

- Appends the token to the `_tokens` list if it's not already present.
- Updates the `_tokenmap` dictionary to record the mapping between the token type and value and its corresponding handle.
- Appends the handle of the token to the `_heap` list.

**Usage:**

```python
# Example usage:
handle = self.write_token(a)
```
### function `write_begin`

Creates a new record in the heap and returns its handle.

**Functionality:**

* Appends `HandleSentry.BEGIN` to the `_heap` list.
* Returns the length of the `_heap` list before appending the sentinel, which serves as the handle for the new record.

**Preconditions:**

* The `_guard()` method is called to ensure the heap is not full.

**Postconditions:**

* A new record is created in the heap with a unique handle.

**Usage:**

```python
handle = object.write_begin()
```

**Note:**

* This function is typically called within the `write()` function to mark the beginning of a new record.
* The `handle` returned by this function can be used to reference the newly created record in other functions.
### function `write_end`

Appends the special token `HandleSentry.END` to the heap and increments the number of records. 

**Functionality:**

* Calls the `_guard()` method to ensure heap integrity.
* Appends `HandleSentry.END` to the `_heap` list.
* Increments the `_num_records` attribute by 1.
### function `_guard`

The `_guard` function is responsible for ensuring that the heap is not overflowing. It checks if the current length of the heap is greater than or equal to `HandleSentry.BEGIN`. If this condition is met, it raises a `HeapOverflow` exception to indicate that the heap is full and cannot accept any more elements.
## class `TraverseState`
## class `SExpr`
### function `_head`

`_head` is a private method used by the `expr` method to extract the head of an `Expr`. It reads the token at the specified handle, asserts that it is a string, and returns the string.
### function `_args`

Returns a tuple of the expression's arguments.

- It extracts the `_args` attribute of the `self` object, which is a tuple of arguments.
- The `_args` attribute is cached using the `cached_property` decorator.
- This function returns the `_args` tuple, which contains the arguments of the expression.
### function `_get_downcast`

Returns a callable that converts an `SExpr` to a `BasicSExpr`.

```python
def downcast(expr):
    if isinstance(expr, BasicSExpr):
        return expr
    else:
        return BasicSExpr(expr._tape, expr._handle)
```
### function `_wrap`

Creates a new `SExpr` instance with the given `tape` and `handle`. It is used internally by the `SExpr` class to represent a specific record in the tape.
### function `_write`

Creates an `SExpr` from the specified `head` and `args`. It handles both built-in and custom objects by writing their tokens or references to the tape.

**Args:**

* `tape`: The tape object to write to.
* `head`: The head of the `SExpr`.
* `args`: A tuple of values to be included in the `SExpr`.

**Returns:**

* An `SExpr` object representing the newly created expression.
### function `_replace`

Replaces the current SExpr with a new one with the same head and arguments.

- If the new arguments are the same as the current arguments, the function returns the current SExpr.
- Otherwise, the function calls the `_write` method to create a new SExpr with the given head and arguments.
- The new SExpr is then returned.
### function `__lt__`

Checks if one `SExpr` object is less than another based on their `_handle` attribute. It returns `True` if the first object's `_handle` is smaller than the second object's `_handle`, and `False` otherwise. If the objects are not of type `SExpr`, it returns `NotImplemented`.
### function `__eq__`

Checks if two `SExpr` objects are equal. Two `SExpr` objects are equal if and only if they have the same `_tape` and `_handle`.
### function `__hash__`

The `__hash__` function for the `SExpr` class calculates a unique hash value based on two attributes: `id(self._tape)` and `self._handle`. This hash value is used for efficient comparison and dictionary operations.
### function `__str__`

The `__str__` method converts an `SExpr` object into a human-readable string representation. It takes the following form:

```
<class_name>(head, arg1, arg2, ...)
```

where:

* `<class_name>` is the name of the `SExpr` class.
* `head` is the first element of the `SExpr`.
* `arg1`, `arg2`, ... are the remaining elements of the `SExpr`.

The `__str__` method is used by the `pretty_str` function to generate a pretty-printed string representation of an `SExpr`.
## function `pretty_str`

The `pretty_str` function takes an `SExpr` object as input and returns a string representation of it using the `pretty_print` function from the `prettyprinter` module. It handles metadata expressions by converting them to their string representation using `str()`. Otherwise, it applies bottom-up rewriting using the `Occurrences` and `Counter` objects to format the expression in a human-readable way.
## function `is_metadata`

Checks if an expression is a metadata expression by verifying if its head starts with the `metadata_prefix`. Metadata expressions are used to store additional information within the code structure.
## function `is_simple`

Checks if an expression is a simple expression, where all arguments are not `Expr` objects.

**Arguments:**

* `sexpr`: An `SExpr` object.

**Return value:**

* `True` if the expression is simple, `False` otherwise.

**Usage:**

The `is_simple()` function can be used to check if an expression is simple. It is particularly useful in conjunction with the `arg_eq()` function to compare expressions based on their simplicity.

**Example:**

```python
sexpr = SExpr("f", 1, 2)  # Not simple
simple_sexpr = SExpr("g", 1)  # Simple

print(is_simple(sexpr))  # Output: False
print(is_simple(simple_sexpr))  # Output: True
```
## function `walk_parents`

Returns an iterator yielding all parent `SExpr` objects of the current object. The returned objects are in the order of their occurrence.
## function `search_parents`

Returns an iterator over parent expressions of `self` that satisfy the given predicate `pred`.

**Arguments:**

* `self`: The `SExpr` object for which to find parents.
* `pred`: A callable that takes an `SExpr` as input and returns `True` if the expression satisfies the condition.

**Returns:**

* An iterator of `SExpr` objects that are parents of `self` and satisfy the predicate `pred`.
## function `search_ancestors`

Performs a breadth-first search starting from the current expression, following the parent links up the expression tree. Yields all parent expressions that satisfy the given predicate function.
## function `walk_descendants`

Walks through the descendants of an `SExpr` node in a breadth-first order, left-to-right. For each descendant, it yields a tuple containing a tuple of its parent `SExpr` nodes and the descendant `SExpr`.
## function `walk_descendants_depth_first_no_repeat`

This function iterates over the descendants of an `SExpr` node in a depth-first manner, avoiding duplication.

**Functionality:**

* Uses a stack to keep track of nodes and their parent nodes.
* Tracks visited nodes using a set.
* Yields a tuple for each visited node, containing a tuple of its parent nodes and the node itself.
* Skips nodes that have already been visited.

**Usage:**

```python
for parents, node in walk_descendants_depth_first_no_repeat(root):
    # Process node and its parent nodes
```
## function `walk_descendants_depth_first`

Walks through the descendants of an `SExpr` node in a depth-first manner, left to right. For each descendant, it yields a tuple containing the parent expressions and the descendant itself.
## function `search_descendants`

This function iterates over all descendant expressions of a given `SExpr` node and returns those that satisfy a given predicate.

**Parameters:**

* `self`: The `SExpr` node to search.
* `pred`: A callable that takes an `SExpr` node as input and returns `True` if the node should be included in the result.

**Returns:**

An iterator of tuples, where each tuple contains a tuple of parent `SExpr` nodes and the descendant `SExpr` node.

**Usage:**

```python
# Find all descendant expressions that are numbers
numbers = search_descendants(tree, lambda x: isinstance(x, Number))

# Print the results
for parents, num in numbers:
    print(f"Parent: {parents}, Number: {num}")
```
## function `traverse`

Traverses the expression tree rooted at the current node, applying the provided coroutine function to each node in a depth-first order. The traversal is memoized, so that if a node is encountered more than once, the result from the first visit is reused. The function returns a dictionary mapping each visited node to the value returned by the coroutine function for that node.
## function `reachable_set`

Returns a set of all reachable SExpr nodes from the given SExpr node. Reachability is determined using a depth-first traversal of the expression tree, avoiding duplicates.

**Functionality:**

* Uses the `walk_descendants_depth_first_no_repeat` function to perform a depth-first traversal of the expression tree, avoiding duplicate nodes.
* Iterates over the returned tuples from `walk_descendants_depth_first_no_repeat`, which contain each visited node and its parent nodes.
* Returns a set containing all the reachable SExpr nodes.
## function `contains`

Checks if a given expression is part of the current expression tree.

**Functionality:**

- Iterates through all descendant nodes of the current expression using the `walk_descendants()` function.
- Compares each descendant node with the `test` expression.
- Returns `True` if the `test` expression is found as a descendant node, `False` otherwise.

**Usage:**

```python
# Create an expression tree
expr = SExpr(...)

# Check if a specific expression is part of the tree
contains_result = expr.contains(test_expr)

# Print the result
print(contains_result)
```
## function `matches`

Checks if two SExpr objects have the same structure and equal values for their arguments.

**Functionality:**

- Uses `walk_descendants_depth_first_no_repeat` to iterate through the SExpr objects recursively.
- Compares the head and arguments of each pair of corresponding nodes.
- Uses `arg_eq` to determine equality of arguments, including nested SExpr objects.
- Returns `True` if the SExpr objects have the same structure and equal arguments, and `False` otherwise.
## function `apply_bottomup`

Applies a `TreeVisitor` to every sexpr in the expression tree in bottom-up order. Before visiting a sexpr, its children must have been visited.

**Args:**

* `visitor`: The `TreeVisitor` to apply to each expression.
* `reachable` (optional): The set of reachable expressions. If set to `"compute"`, the reachable set will be computed automatically. Defaults to `"compute"`.

**Returns:**

None

**Example Usage:**

```python
# Apply the visitor to the entire expression tree
apply_bottomup(expr, visitor)

# Apply the visitor only to reachable expressions
apply_bottomup(expr, visitor, reachable=reachable_set(expr))
```
## function `apply_topdown`

The `apply_topdown` function iterates through every sexpr under the given subtree and applies the specified `TreeVisitor` to each one. It excludes metadata nodes and applies the visitor only to the non-metadata nodes.
## function `as_tuple`

Converts an `Expr` object to a tuple, with a specified depth limit.

**Args:**

- `depth (int):` The maximum depth to traverse the `Expr` object. Set to -1 for unbounded.
- `dedup (bool):` Whether to deduplicate elements in the resulting tuple.

**Returns:**

- `tuple[Any, ...]`: A tuple containing the elements of the `Expr` object, up to the specified depth.

**Note:**

- The `depth` argument is applied recursively to the sub-expressions of the `Expr` object.
- The `dedup` argument determines whether to include duplicate elements in the resulting tuple.
- The function uses a bottom-up traversal of the `Expr` object, visiting the child nodes before the parent nodes.
## function `as_dict`

Converts an expression tree into a dictionary representation. The `as_dict` method takes an `Expr` object and returns a dictionary representation of the object. It uses a memoization technique to avoid duplicate references.

The method handles simple expressions differently, by directly including the argument values in the dictionary.
## function `copy_tree_into`

Copies an entire expression tree starting with the given `SExpr` node into a new `Tape`.

**Parameters:**

* `self`: The `SExpr` node to copy.
* `tape`: The new `Tape` to copy the tree into.

**Returns:**

* A new `SExpr` node in the new `Tape`.

**Functionality:**

* Creates a copy of the entire expression tree starting with the given `SExpr` node.
* The new `SExpr` node is stored in the new `Tape`.
* The copy includes all of the sub-expressions, tokens, and references.
* The copy is guaranteed to be structurally identical to the original tree.
## class `BasicSExpr`
### function `_wrap`

Creates a new instance of the class with the given tape and handle.

**Parameters:**

* `cls`: The class to create an instance of.
* `tape`: The tape to store the new instance.
* `handle`: The handle to store the new instance.

**Returns:**

A new instance of the class with the given tape and handle.
### function `__init__`

Initializes a new `Expr` instance.

**Args:**

- `tape`: A `Tape` object.
- `handle`: A handle value.

**Returns:**

None.
### function `_head`

Returns the value of the tape head at the current handle.

```python
def _head(self) -> str:
    tape = self._tape
    return tape.read_head(self._handle)
```
### function `_args`

Returns a tuple containing the arguments of the expression.

**Usage:**

```python
tape = ...  # Initialize a Tape object
handle = ...  # Handle of the expression

args = tape._args(handle)  # Get the arguments of the expression
```
### function `__repr__`

Returns a string representation of the `Expr` object. It concatenates the expression head and handle, followed by the hexadecimal ID of the tape object.
### function `_get_downcast`

Returns a function that converts an `SExpr` to a `BasicSExpr`. If the `SExpr` is already a `BasicSExpr`, it is returned as is. Otherwise, a new `BasicSExpr` is created with the same `Tape` and `handle`.
## class `TreeVisitor`
### function `visit`

The `visit` function is an abstract method that needs to be implemented by concrete visitor classes. It takes an `expr` argument of type `SExpr` and performs some unspecified action on it.

**Purpose:**

The purpose of the `visit` function is to visit an `SExpr` object and perform some action based on its type or contents.

**Functionality:**

- The function is declared as an abstract method in the `Visitor` class.
- It is defined with a single parameter, `expr`, of type `SExpr`.
- The `visit` function is not implemented in the `Visitor` class, but it must be implemented in concrete visitor classes.
- The `visit` function is responsible for handling different types of `SExpr` objects and performing actions based on their specific characteristics.
## class `TapeCrawler`
### function `__init__`

Initializes a new instance of `TapeCrawler`.

- Takes two arguments:
    - `tape`: The `Tape` object to crawl.
    - `downcast`: A callable that converts an `SExpr` to a `BasicSExpr`.
- Initializes the `_tape` attribute to the given `tape`.
- Initializes the `_pos` attribute to 0.
- Initializes the `_downcast` attribute to the given `downcast` callable.
### function `move_to_first_record`

Sets the current position to the first record in the tape.
### function `pos`

Returns the current position within the tape.

```python
    def pos(self) -> handle_type:
        return self._pos
```
### function `get`

Retrieves the record at the current position (`self._pos`) from the tape (`self._tape`).

**Returns:**

* `handle_type`: The handle of the retrieved record.
### function `seek`

Seek sets the position pointer to the specified handle. The handle must be of type `HandleSentry.BEGIN`. After calling this function, the `pos` property of the object will be set to the specified handle.
### function `step`

Increments the current position by 1.

**Functionality:**

- Updates the internal `_pos` attribute by adding 1.
- This function is typically used within the `walk()` method to iterate through a sequence of records in the tape.
### function `skip_to_record_end`

Skips to the record end handle by setting the current position to the index of the first `HandleSentry.END` handle starting from the next position. This is used to navigate to the end of a record.
### function `walk`

The `walk` function is an iterator that yields `Record` objects. It iterates through the tape starting from the current position (`self._pos`) and continues until the end of the heap (`self._tape.heap_size`). For each iteration, it calls the `read_record()` method to read the record and yields it as an iterator.
### function `walk_descendants`

**Purpose:**

The `walk_descendants()` function iterates through all descendant records starting from the current record. It uses a breadth-first search (BFS) approach, yielding each record along with its parent records in each iteration.

**Functionality:**

- The function initializes a queue with a tuple containing an empty tuple of parent records and the current record.
- It iterates through the queue until it is empty.
- In each iteration, it removes the first element from the queue and yields the parent records and the current record.
- For each child record of the current record, it adds a new element to the queue with the updated parent records and the child record.

**Return Value:**

- An iterator that yields tuples of (parents, record), where parents is a tuple of Record objects and record is a Record object.
### function `read_record`

Read a record from the tape.
- Asserts that the current position is at the beginning of a record.
- Finds the end of the record and creates a `Record` object.
- Advances the position to the end of the record.
- Returns the `Record` object.
### function `read_surrounding_record`

Reads a record from the tape by finding the `HandleSentry.BEGIN` and `HandleSentry.END` handles and creating a new `Record` object.

**Parameters:**

* None

**Returns:**

* A `Record` object representing the surrounding record.
### function `move_to_pos_of`

Moves the internal position pointer to the handle specified by the `target` argument. Returns `True` if the handle is found, `False` otherwise.

**Note:**

* This function assumes that the handle is located in the tape starting from the current position.
* It uses the `index()` method of the `_tape` object to find the handle.
* If the handle is not found, the function returns `False`.
### function `move_to_previous_record`

Moves the cursor to the start of the previous record in the tape. If `startpos` is not provided, the current position is used. The function first moves to the start of the current record, then to the start of the previous record.
## class `Record`
### function `children`

Returns an iterator of `Record` objects representing the child records of the current record. Child records are determined by the `HandleSentry.END` sentinel in the tape. Tokens are excluded from the returned iterables.
### function `read_head`

Returns the head of the record at the given handle.

**Parameters:**

* `handle`: The handle of the record.

**Returns:**

* The head of the record as a string.
### function `read_args`

Returns the arguments associated with the record represented by the `self.handle` property.
This information is stored in the `self.tape` object.
### function `to_expr`

The `to_expr` function converts a `BasicSExpr` object to an `SExpr` object. It checks if the `BasicSExpr` object is metadata by calling the `is_metadata()` function. If it is metadata, the function returns the `BasicSExpr` object directly. Otherwise, the function calls the `downcast()` method on the `BasicSExpr` object and returns the result.
### function `__repr__`

Returns a string representation of the `Record` object in the format:

```
<Record handle:end_handle tape@hex(tape_id) >
```

where:

* `handle` and `end_handle` are the handles of the record in the tape.
* `tape_id` is the unique identifier of the tape object.
## function `_select`

Returns the value at the specified index from each iterable in an `iterable`.

**Arguments:**

* `iterable`: An iterable of iterables.
* `idx`: The index of the value to return.

**Returns:**

* An iterator over the values at the specified index from each iterable in the `iterable`.
