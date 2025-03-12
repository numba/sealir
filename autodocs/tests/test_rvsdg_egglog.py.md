## function `read_env`

Reads an environment variable and converts it to an integer. If the environment variable is not set, it returns 0.

**Functionality:**

* Reads an environment variable with the name specified in the `v` argument.
* Converts the environment variable value to an integer using `int()`.
* Returns the integer value if the environment variable is set, or 0 if it is not.
## function `run`

`run` is a function that takes a root object and performs the following steps:

1. Creates an `EGraph` object.
2. Sets the root object in the graph.
3. Makes a set of rules.
4. Assumes facts into the graph using the `assume` function if provided.
5. Saturates the graph with the rules.
6. Extracts the simplified output from the graph.
7. Checks the extracted output against the provided checks.
8. Prints debug information if debugging is enabled.

**Example Usage:**

```python
# Run the function with a root object and checks
egraph = run(root, checks=[check1, check2])
```

**Note:**

* The `assume` function can be used to provide additional facts to the graph.
* The `debug_points` argument can be used to specify debug points in the graph.
* The `EGraph` object contains the simplified output of the function.
## function `saturate`

Saturates an egraph with a given ruleset.

**Parameters:**

* `egraph`: The egraph to saturate.
* `ruleset`: The ruleset to use for saturation.

**Returns:**

A list of reports, one for each iteration of saturation.

**Functionality:**

* Runs the ruleset on the egraph until no more progress is made.
* Visualizes the egraph at each iteration.
* Returns a list of reports, each report containing the number of matches for each rule.

**Notes:**

* The `DEBUG` flag determines whether to visualize the egraph at each iteration.
* The saturation process is limited to 1000 iterations to prevent infinite loops.
## function `region_builder`

The `region_builder` function is a decorator that creates a new region with a given arity. It takes an optional `arity` argument, which specifies the number of input parameters to the region. If no `arity` is provided, the function automatically creates a region with the number of input parameters specified by the first argument passed to the decorated function.

The `region_builder` function works by wrapping the decorated function and creating a new `Region` object with the specified arity. The wrapped function is then executed within the new region, and the outputs of the function are returned as a `Term.RegionEnd` object.

This function is typically used to define regions with a fixed number of input parameters. It can be used to simplify region construction and reduce boilerplate code.
## function `test_straight_line_basic`

This function tests the basic functionality of a straight line program. It creates an environment with five parameters, evaluates the main function, and checks if the result matches the expected value.

**Functionality:**

- Creates an environment with five parameters.
- Defines a main function that returns a list of the parameter values.
- Evaluates the main function.
- Compares the result with the expected value.
- Runs the program and checks for errors.
## function `test_max_if_else`

This function tests the `Term.Branch` construct with two child regions, `if_then` and `or_else`. It takes two inputs `a` and `b`, compares them using `Term.Lt`, and then branches based on the comparison result. If `a` is less than `b`, it returns `b`, otherwise it returns `a`.
## function `test_loop_analysis`

This function tests the loop analysis functionality by running a loop and checking if the loop induction variable (LV) analysis correctly identifies the loop's invariant.

**Functionality:**

* Creates a `main` region that takes two parameters `a` and `b`.
* Defines a `loop` region nested within `main`.
* Inside the `loop` region, it performs the following steps:
    * Stores the value of `a` in the `debug_points` dictionary with the key "a".
    * Calculates `na` by adding 1 to `a`.
    * Checks if `na` is less than `b`.
    * Returns a list containing the conditional expression, `na`, and `b`.
* Compiles the `main` region with the `Env` object.
* Runs the compiled region using the `run` function with a check that verifies the LV analysis result for the "a" key.

**Purpose:**

* To test the loop analysis mechanism and ensure it correctly identifies the loop invariant.
* To demonstrate how the `debug_points` dictionary is used to store loop invariant information.
## function `test_sum_loop`

The `test_sum_loop` function implements the sum function using a loop. It takes two parameters, `init` and `n`, and returns the sum of all integers from 0 to `n-1`, with the initial value set to `init`. The function uses the `Term.Loop` construct to achieve this.

**Functionality:**

- Takes two integer parameters, `init` and `n`.
- Initializes a variable `c` with the value of `init`.
- Initializes a variable `i` with the value of 0.
- Starts a loop that continues until `i` is greater than or equal to `n`.
    - In each iteration, adds the value of `i` to the variable `c`.
    - Increments the variable `i` by 1.
- Returns the value of `c`.
