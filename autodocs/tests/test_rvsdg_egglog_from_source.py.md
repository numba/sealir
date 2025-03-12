## function `read_env`

The `read_env` function reads an environment variable and converts it to an integer.

- If the environment variable is set, it parses the value as an integer.
- If the environment variable is not set, it returns 0.

This function is used to set the `DEBUG` flag in the codebase.
## function `run`

The `run` function is responsible for running the program. It takes in a root value, extra statements, checks, an assume function, and debug points as arguments.

**Functionality:**

1. Creates an `EGraph` object.
2. Sets the root value in the graph.
3. Creates variables for each extra statement.
4. Makes a ruleset if it doesn't exist.
5. Assumes the input if an assume function is provided.
6. Saturates the graph with the ruleset.
7. Simplifies the output.
8. Checks the graph for errors.
9. Returns the `EGraph` object.

**Example Usage:**

```python
run(root, extra_statements, checks, assume, debug_points)
```
## function `saturate`

Saturates an `EGraph` instance with the given `ruleset`.

**Functionality:**

* Runs the `ruleset` on the `egraph` instance repeatedly until no more updates are made.
* Creates a visual representation of the `egraph` at each iteration.
* Returns a list of reports, each containing information about the number of matches for each rule in the `ruleset`.

**Parameters:**

* `egraph`: An `EGraph` instance.
* `ruleset`: A `ruleset` instance.

**Returns:**

* A list of `Report` instances.
## function `test_max_if_else_from_source`

This function performs various tests on a user-defined function (`udt`) that implements the maximum of two numbers using an `if-else` statement.

**Functionality:**

* Restructures the source code of `udt` into an intermediate representation using the `rvsdg` library.
* Converts the restructured representation into an egraph using the `egraph_conversion` function.
* Creates an environment with the necessary arguments for `udt`.
* Evaluates `udt` with the environment and partial evaluation using the `run` function.
* Extracts the result of `udt` using `egraph_extraction`.
* Generates LLVM code for `udt`.
* Compares the result of `udt` with the expected value.

**Steps:**

1. Restructures the source code of `udt`.
2. Converts the restructured code into an egraph.
3. Creates an environment with the arguments `a` and `b`.
4. Evaluates `udt` with partial evaluation.
5. Extracts the result of `udt`.
6. Generates LLVM code for `udt`.
7. Compares the result with the expected value.

**Purpose:**

This function tests the correctness of the maximum function implemented in `udt` by comparing the result with the expected value.
## function `skip_test_sum_loop_from_source`

This function performs the following steps:

1. Restructures a source function using the `rvsdg` module.
2. Converts the restructured function to an egraph using `egraph_conversion`.
3. Evaluates the egraph with an environment that includes the source function's arguments.
4. Extracts the cost and result of the evaluated egraph.

The function can be used to test the correctness of source functions without having to run them in the actual program. It can also be used to profile the performance of source functions.
