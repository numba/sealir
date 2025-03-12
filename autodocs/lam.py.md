## class `_Value`
## class `Lam`
## class `Arg`
## class `App`
## class `Unpack`
## class `Pack`
## class `LamGrammar`
## function `_intercept_cells`

The `_intercept_cells` function modifies the closure of a given function `fn` to adjust the De Bruijn indices of its arguments.

**Purpose:**

* Increment De Bruijn indices for arguments in the function's closure.
* Replace `Arg` SExpr cells with cells containing the updated De Bruijn indices.

**Functionality:**

1. Checks if the function's closure is empty. If so, it returns the original function without any modifications.
2. Creates a new closure with updated De Bruijn indices.
3. Iterates through each cell in the original closure:
    * If the cell contains an `Arg` SExpr, it increments the De Bruijn index and replaces the cell with a new cell containing the updated index.
    * If the cell contains a `Lam` SExpr (lambda expression), no change is made.
    * If the cell contains anything else, it keeps the original cell.
4. Returns a new function with the updated closure.

**Usage:**

This function is used in the `_debruijn_to_python` function to adjust De Bruijn indices of arguments in a lambda expression. It is also used in other parts of the codebase to manipulate functions with De Bruijn indices.
## function `lam_func`

The `lam_func` is a decorator that helps build lambda expressions from a Python function. It takes multiple arguments and uses De Bruijn index.

**Functionality:**

* Takes a Python function as an argument.
* Creates a lambda expression with the same arguments as the Python function.
* Uses De Bruijn index to assign unique names to each argument.
* Intercepts cell variables in the Python function and replaces them with lambda expressions.
* Returns a lambda expression that can be used in the Lisp codebase.

**Example:**

```python
@lam_func(grm)
def my_function(a, b):
    return a + b
```

**Note:**

* The `_intercept_cells()` function is used to intercept cell variables in the Python function and replace them with lambda expressions.
* The `App()` function is used to create an apply expression in Lisp.
* The `Unpack()` function is used to unpack a tuple with known size in Lisp.
## function `app_func`

`app_func` creates an apply expression by recursively applying a lambda function to a sequence of arguments.

**Functionality:**

* Takes four arguments:
    * `grm`: A grammar object.
    * `lam`: A lambda function.
    * `arg0`: The first argument.
    * `*more_args`: Additional arguments.
* Creates a list of arguments by concatenating `arg0` and `more_args`.
* Iterates through the arguments and applies the lambda function to each argument using the `grm.write()` method.
* Returns the final apply expression.
## function `unpack`

Unpacks a tuple with known size `tup` into a new tuple of expressions using the `Unpack` rule.

**Parameters:**

* `grm`: A `grammar.Grammar` object.
* `tup`: An `ase.SExpr` representing the tuple to unpack.
* `nelem`: An integer representing the number of elements in the tuple.

**Returns:**

A new tuple of `ase.SExpr` objects.
## function `format`

Formats an S-expression into a multi-line string representation.

**Functionality:**

* Takes an `ase.SExpr` object as input.
* Uses the `format_lambda()` function to format the S-expression recursively.
* Returns a string representation of the formatted S-expression.

**Usage:**

```python
expr = ase.SExpr(...)  # Create an S-expression object
formatted_str = format(expr)  # Format the S-expression
```
## function `simplify`

Simplifies a grammar by removing dead nodes. It creates a copy of the grammar with the last node being the root. The function assumes that the last node in the grammar is the root.
## function `beta_reduction`

Reduces a lambda expression within an application expression.

**Arguments:**

* `app_expr`: The application expression containing a lambda expression.

**Returns:**

* `ase.SExpr`: The reduced expression.

**Functionality:**

The `beta_reduction` function performs beta reduction on an application expression that contains a lambda expression. It iterates through the descendants of the application expression, identifying arguments that match the lambda expression's argument. When a match is found, the lambda expression is replaced with the corresponding argument.

**Example:**

```python
app_expr = ase.SExpr("App", [ase.SExpr("Arg", [0]), lambda_expr])
reduced_expr = beta_reduction(app_expr)
```

**Additional Notes:**

* The function relies on the `walk_descendants()` function to iterate through the application expression and its descendants.
* The `BetaReduction` object is used to store the replacements and perform the reduction.
* The function assumes that the application expression has a valid structure and that the lambda expression is well-formed.
## function `run_abstraction_pass`

Converts expressions that would otherwise require a let-binding to use lambda-abstraction with an application. It iteratively identifies expressions that occur multiple times within a lambda expression and replaces them with lambda-abstractions and applications. The process continues until no more such expressions remain.

**Functionality:**

* Uses a counter to identify expressions that occur multiple times within a lambda expression.
* Uses a `TreeVisitor` to recursively traverse the expression and update the counter.
* Selects the oldest (outermost) multi-occurring expression.
* Finds the parent lambda expression containing the selected expression.
* Uses the `replace_by_abstraction` function to replace the expression with a lambda-abstraction and an application.
* Replaces the remaining expressions in the lambda expression using a `TreeRewriter`.
* Repeats the process until no more multi-occurring expressions are found.
* Returns the modified expression with lambda-abstractions and applications.
## function `replace_by_abstraction`

This function takes three arguments:

* `grm`: A `grammar.Grammar` object.
* `lamexpr`: An `ase.SExpr` representing a lambda node.
* `anchor`: An `ase.SExpr` that will be replaced with an argument in the new lambda abstraction.

The function returns a tuple containing two `ase.SExpr` objects:

* The original node from `lamexpr`.
* A new `ase.SExpr` that represents a lambda abstraction with the original node as an argument and the `anchor` node as an argument to the lambda.

The function works by:

1. Finding the lambda depth of each child node in the lambda node.
2. Rewriting the children nodes into a new lambda abstraction, replacing the `anchor` node with an argument node.
3. Returning the original node and the new lambda abstraction as a tuple.
## function `rewrite_into_abstraction`

The `rewrite_into_abstraction` function rewrites an expression into an abstraction. It takes the following parameters:

- `grm`: A grammar object.
- `root`: The root of the expression to rewrite.
- `anchor`: The anchor expression.
- `lam_depth_map`: A dictionary mapping lambda expressions to their depth in the expression.

The function works as follows:

1. It calculates the argument index based on the depth of the anchor expression in the lambda depth map.
2. It creates a `RewriteAddArg` rewriter object that shifts de Bruijn indices in the expression's body by 1 for lambda expressions before the anchor expression.
3. It applies the rewriter to the root expression using `apply_bottomup`.
4. It returns the rewritten expression.

The function is used in the `apply_arg_rules` function to rewrite arguments in an expression.
## function `format_lambda`

The `format_lambda` function takes an `ase.SExpr` as input and returns a formatted string representation of the lambda expression. The function iterates through the expression and formats each lambda expression and its nested expressions, including nested let statements.

**Functionality:**

* Identifies lambda expressions within the input expression.
* Calculates the depth of each lambda expression.
* Creates a nested scope for each lambda expression.
* Formats the lambda expression and its nested expressions using indentation and let statements.
* Returns a formatted string representation of the lambda expression.

**Example Usage:**

```python
expr = ase.parse("(lambda x (lambda y (+ x y)))")
formatted_string = format_lambda(expr)
print(formatted_string)
```

**Output:**

```
(let $0 = λ x
  (let $1 = λ y
    (+ x y)))
```
## class `BetaReduction`
### function `__init__`

Initializes a new instance of the class.

**Arguments:**

* `drops`: A dictionary of drops.
* `repl`: A dictionary of replacements.

**Purpose:**

The `__init__` function initializes a new instance of the class with the given `drops` and `repl` dictionaries. These dictionaries are used to specify the drop and replacement operations for the class instance.
### function `rewrite_Arg`

The `rewrite_Arg` function takes two arguments: `orig` and `index`. It returns an `ase.SExpr` object.

This function is responsible for rewriting an argument in an expression. It takes the original argument `orig` and the index `index` as input. It then uses the `_repl` dictionary to replace `orig` with a new expression. If `orig` is not found in the `_repl` dictionary, it returns the original argument `orig`.
### function `rewrite_App`

The `rewrite_App` function checks if the input expression `orig` is present in the `_drops` set. If it is, it returns the lambda expression `lam`. Otherwise, it calls the `passthru()` method to return the original expression `orig`.
### function `rewrite_Lam`

The `rewrite_Lam` function takes two arguments, `orig` and `body`. It checks if `orig` is in a list of elements called `_drops`. If it is, it returns `body`. Otherwise, it creates a new type variable `tv` and uses it to create a new equivalence between `tv` and a new lambda expression with `body` as its body and `self.memo[arg_node]` as its argument. Finally, it returns `tv`.
