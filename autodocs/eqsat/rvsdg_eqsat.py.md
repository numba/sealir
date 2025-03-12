## class `Region`
### function `__init__`

Initializes a new `Region` object.

**Parameters:**

* `uid`: A unique identifier for the region.
* `ins`: Instructions for the region.
* `ports`: A list of terms representing the input ports of the region.

**Purpose:**

This function creates a new `Region` object with the given parameters. It is used to represent a logical region of code within a program.
### function `begin`

Returns the input ports for the current region.

**Usage:**

```python
region = Region(...)  # Create a Region object
input_ports = region.begin()  # Get the input ports
```
## class `InputPorts`
### function `get`

Retrieves a term at the given index.

**Parameters:**

* `idx`: The index of the term to retrieve.

**Returns:**

* A `Term` object representing the retrieved term.
## class `Env`
### function `nil`

Returns an empty environment.

This function is used to initialize a new environment for evaluation. It creates a new instance of the `Env` class with no associated values or ports.
### function `nest`

Creates a new environment by nesting the given `ports` within the current environment.

**Parameters:**

* `ports`: A `ValueList` containing `Value.Param` objects, representing the ports to nest.

**Returns:**

* `Env`: A new environment with the nested ports.

**Example:**

```python
# Create a new environment with two nested ports
env = Env.nil().nest(valuelist(Value.Param(0), Value.Param(1)))
```
## class `Value`
### function `Param`

Creates a new `Value` object representing a parameter with the given index `i`.
```
### function `IOState`

Returns a special value that represents the state of the I/O devices in the program. This value is used to track the state of input and output operations and is used by the `_dispatch` function to implement the semantics of I/O operations.
### function `BoolTrue`

Creates a new `Value` object representing a boolean value with a `True` value.
### function `BoolFalse`

Returns a Value representing a Python `False` value.
### function `ConstI64`

Creates a new constant integer value with the given 64-bit integer value.

**Parameters:**

* `val`: A 64-bit integer value.

**Returns:**

* A new `Value` object representing the constant integer value.
### function `toList`

Converts the `ValueList` object to a `ValueList` object.

**Returns:**
- A `ValueList` object containing the elements of the original `ValueList` object.
### function `__or__`

The `__or__` function allows for the union of two `Value` objects. It takes another `Value` object as an argument and returns a new `Value` object. The specific behavior of the function depends on the types of the two operands. If both operands are of type `ValueList`, the function appends the elements of the second `ValueList` to the first `ValueList`. If one operand is of type `ValueList` and the other is not, the function converts the non-`ValueList` operand to a `ValueList` and then appends the elements of the second operand to the first `ValueList`.
## class `ValueList`
### function `__init__`

Initializes a new `ValueList` object with the given list of `Value` objects.

**Args:**

- `vals`: A vector of `Value` objects.

**Returns:**

None
### function `append`

Appends a `ValueList` to the current `ValueList`.

**Parameters:**

* `vs`: The `ValueList` to append.

**Returns:**

* The appended `ValueList`.
### function `toValue`

Returns the `Value` object representing this `Expr`. This method is used internally by the codebase to represent expressions as values.
### function `toSet`

Converts the `ValueList` to a set of `Value`.

**Functionality:**

* Takes a `ValueList` as input.
* Returns a set of `Value`.

**Example:**

```python
value_list = ValueList(Vec[Value]())
set_value = value_list.toSet()
```
### function `map`

Applies a function to each element in a `ValueList` and returns a new `ValueList` containing the results.

```python
def map(self, fn: Callable[[Value], Value]) -> ValueList:
    return ValueList(Vec(*map(fn, self.vals)))
```
### function `Merge`

Merges two `ValueList` objects using a provided `merge_fn` function.

**Arguments:**

- `merge_fn`: A function that takes two `Value` objects as input and returns a new `Value` object.
- `vas`: The first `ValueList` object.
- `vbs`: The second `ValueList` object.

**Returns:**

- A new `ValueList` object containing the merged values.

**Usage:**

```python
ValueList.Merge(
    merge_fn,
    ValueList(Vec[Value].empty()),
    ValueList(Vec[Value].empty()),
)
```
## class `Term`
### function `Func`

Creates a new function term.

**Arguments:**

* `uid`: The unique identifier of the function.
* `fname`: The name of the function.
* `body`: The body of the function.

**Returns:**

A new function term.

**Usage:**

```python
# Create a new function with the given identifier, name, and body.
func = Func(uid, fname, body)
```
### function `Branch`

The `Branch` function takes three arguments:

- `cond`: The condition to check.
- `then`: The term to execute if the condition is true.
- `orelse`: The term to execute if the condition is false.

It returns a new term that represents a conditional branch.
### function `Loop`

The `Loop()` function creates a new `Term.Loop` object based on the input `body` term. It takes the `body` term as an argument and returns a new `Term.Loop` object.

The function is used in the context of a codebase where loops are represented as terms. It takes the `body` term and converts it into a `Term.Loop` object, which can then be used in other functions or expressions.
### function `NotIO`

`NotIO` is a class method that performs bitwise NOT operation on an input term `io` and another term `term`. It returns a new term with the result of the operation.

This function is used in the codebase to perform bitwise NOT operations on input terms before passing them to other functions. For example, in the context of the codebase, the `NotIO` method is used in the `case "*"` and `case "/"` sections of the code.
### function `Lt`

Determines if one term is less than another.

**Args:**

* `a`: The first term.
* `b`: The second term.

**Returns:**

* `Term`: A term representing whether `a` is less than `b`.
### function `LtIO`

Performs less than comparison on two terms within an input-output context. It takes three arguments:

* `io`: The input-output context.
* `a`: The first term.
* `b`: The second term.

It returns a term that represents the result of the comparison.
### function `Add`

Adds two terms together.

**Usage:**

```python
Add(a, b)
```

**Arguments:**

* `a`: The first term.
* `b`: The second term.

**Returns:**

* A new term that represents the sum of the two input terms.
### function `AddIO`

Adds two terms together.

**Args:**

* `io`: The IO term.
* `a`: The first term.
* `b`: The second term.

**Returns:**

* The sum of `a` and `b`.
### function `Mul`

Multiplies two terms and returns the result.

**Usage:**

```python
Mul(a, b)
```

**Parameters:**

* `a`: The first term.
* `b`: The second term.

**Returns:**

* The result of the multiplication.
### function `MulIO`

Multiplies two terms with the given input `io`, `a`, and `b`. Returns the result as a `Term`.
### function `Div`

Divides two terms `a` and `b`. This operation is atomic and has a cost of 10.
### function `DivIO`

Divides two terms and performs input/output operations.

**Input:**

- `io`: Input/output term
- `a`: First term
- `b`: Second term

**Output:**

- Term representing the division result
### function `Pow`

Computes the power of two terms `a` and `b`. 
- It raises `a` to the power of `b`.
- It accepts two terms as input: `a` and `b`.
- It returns a term representing the result of the power operation.
### function `PowIO`

Calculates the power of a term `a` to the power of a term `b`.

**Parameters:**

* `io`: The input/output term.
* `a`: The base term.
* `b`: The exponent term.

**Returns:**

A term representing the result of `a` raised to the power of `b`.
### function `AttrIO`

The `AttrIO` function takes three arguments: `io`, `obj`, and `attrname`. It returns a `Term` object.

This function is used to access the attribute of an object. The `io` argument specifies the input stream, the `obj` argument specifies the object, and the `attrname` argument specifies the name of the attribute.
### function `LiteralI64`

Creates a new integer literal term with the given value. The value must be an integer less than 64 bits in size.

**Usage:**

```python
Term literal_term = LiteralI64(val)
```

**Parameters:**

* `val`: The integer value of the literal term.

**Return Value:**

* A new `Term` object representing the integer literal.
### function `LiteralF64`

Creates a new term representing a 64-bit floating-point literal value.

**Parameters:**

* `val`: The floating-point value to represent.

**Returns:**

* A new `Term` object representing the literal value.
### function `LiteralBool`

Creates a new `Term` instance representing a boolean literal value.

**Arguments:**

* `val`: A Python boolean value.

**Returns:**

A `Term` object representing the boolean literal value.

**Example:**

```python
# Create a boolean literal term with the value True
bool_term = LiteralBool(True)
```
### function `IO`

The `IO` function is a class method that takes a single argument, `cls`, and returns a `Term` object. The function is used in conjunction with other class methods to provide an input/output interface for operations on terms.
### function `Undef`

Returns an undefined term. This is a placeholder for terms that have not been defined yet.
### function `Param`

Creates a new `Term` representing a parameter with the given index.
```
### function `LoadGlobal`

Loads a global variable from the specified `io` object using the given `name`.
```python
ioterm = yield io
return eg.Term.LoadGlobal(ioterm, name)
```
### function `Call`

Creates a new term that represents a function call. It takes three arguments:

* `func`: The term representing the function to call.
* `io`: The term representing the input/output environment.
* `args`: A `TermList` containing the arguments to pass to the function.

The function returns a new term that represents the result of the function call.
### function `RegionEnd`

Creates a region end term.

**Args:**

* `region`: The region to end.
* `outs`: The outputs of the region.
* `ports`: The ports of the region.

**Returns:**

A term representing the region end.
### function `getPort`

Retrieves the term at the specified index from the input ports of a region.

**Args:**

* `idx`: The index of the port to retrieve.

**Returns:**

* `Term`: The term at the specified index.
## class `TermList`
### function `__init__`

Initializes an instance of the `EClassData` class.

**Parameters:**

* `terms`: A dictionary of `Term` objects.

**Purpose:**

The `__init__` method initializes the `_terms` dictionary with the provided `terms` argument. It also iterates over the `terms` dictionary and populates the `_eclasses` dictionary with a set of `Term` objects for each `eclass`.
### function `mapValue`

Maps each element in the `TermList` to a `Value` using the provided function `fn`. The result is a new `ValueList`.

**Args:**

* `self`: The `TermList` object.
* `fn`: A function that takes a `Term` and returns a `Value`.

**Returns:**

* A new `ValueList` containing the mapped values.
## class `Debug`
### function `ValueOf`

Returns the `Value` of a given `Term`. This function is used for debugging purposes and should not be used in production code.

**Usage:**

```python
Debug.ValueOf(term)
```

**Arguments:**

* `term`: The `Term` to get the value of.

**Returns:**

* A `Value` object.
## function `termlist`

Creates a new `TermList` from the given `Term` arguments.

**Usage:**

```python
termlist(term1, term2, ...)
```

**Returns:**

* `TermList`: A new `TermList` containing the given `Term` arguments.
## function `has_pure_type`

Checks if a given value has a pure type. A value is considered pure if all operations performed on it have no side effects. The function performs the following checks:

* Constant I64 values are always pure.
* Binary operations (`+`, `-`, etc.) with two pure values are considered pure.
* Comparisons (`==`, `!=`, etc.) with pure values are considered pure.

```python
def has_pure_type(v: Value) -> Bool:
    """Type of value is pure such that all operations on it has no side effects."""
    ...
```
## function `Eval`

The `Eval` function evaluates a given term within an environment. It takes two arguments:

- `env`: The environment in which the term should be evaluated.
- `term`: The term to be evaluated.

The function returns a `Value` object representing the result of the evaluation.

**Functionality:**

- The function evaluates the given term within the provided environment.
- It uses the `Env` object to access and manipulate variables within the environment.
- The evaluation process follows the syntax of the codebase, including functions like `has_pure_type`, `EvalMap`, `VFix`, `VGetPort`, and `VFunc`.
- The evaluation result is returned as a `Value` object.
## function `EvalMap`

Transforms a list of terms into a list of evaluated values.

**Arguments:**

* `env`: The environment in which to evaluate the terms.
* `terms`: A list of terms to evaluate.

**Returns:**

* `ValueList`: A list of evaluated values.
## function `VFix`

The `VFix` function takes a `Value` as input and returns a `Value`. Its purpose is to perform some operation on the input value, but the specific details of the operation are not specified in the codebase.
## function `VGetPort`

Returns the value at the specified index from a `ValueList`.

**Arguments:**

* `vs`: The `ValueList` to retrieve the value from.
* `idx`: The index of the value to retrieve.

**Returns:**

* `Value`: The value at the specified index.

**Usage:**

```python
# Get the first value from a ValueList
first_value = VGetPort(valuelist(value1, value2), 0)
```
## function `VFunc`

`VFunc` is a function that takes a single argument `body` of type `Value` and returns a `Value`. The function definition itself does not contain any code, suggesting that the actual functionality is implemented elsewhere in the codebase.
## function `VBranch`

`VBranch` is a function that performs a conditional branch operation. It takes three arguments:

- `cond`: A boolean value. If the value is `true`, the function returns the `then` argument. Otherwise, it returns the `orelse` argument.
- `then`: The value to return if the condition is `true`.
- `orelse`: The value to return if the condition is `false`.

The function is used to control the flow of execution in a program. It is similar to the `if-else` statement in other languages.
## function `_LoopTemp`

The `_LoopTemp` function is a helper function used within the loop construct. It takes a `ValueList` as input and returns a `Value`. Its purpose is to temporarily store intermediate values during the loop iterations.
## function `_LoopBack`

The `_LoopBack` function takes two arguments, `phis` and `body`, and returns a `ValueList`. It is used internally by the `VLoop` function to implement loop back functionality.

The specific functionality of `_LoopBack` is not explicitly defined in the given codebase, but it is likely responsible for handling the phi functions and body term within a loop iteration.
## function `_LoopDropCond`

The `_LoopDropCond` function takes a `Value` as input and returns a `ValueList`. Its functionality is to drop the condition argument of a loop, effectively removing it from the loop's execution.
## function `_LoopOutputs`

Returns a `ValueList` containing the value `v`.

**Arguments:**

* `v`: The value to be included in the `ValueList`.

**Returns:**

* A `ValueList` containing the value `v`.
## function `_LoopPhiOf`

Transforms a phi value within a loop by applying the loopback value.

**Args:**

* `phi`: The phi value.
* `loopback`: The loopback value.

**Returns:**

The transformed phi value.
## function `_MapLoopPhiOf`

The `_MapLoopPhiOf` function takes two `ValueList` arguments, `phis` and `loopbacks`, and returns a `ValueList`. It iterates through the elements of `phis` and `loopbacks`, applying the `_LoopPhiOf` function to each pair of elements. The result is a new `ValueList` containing the results of the `_LoopPhiOf` function calls.
## function `VLoop`

The `VLoop` function performs a loop operation over a list of input values. It takes two arguments:

* `phis`: A `ValueList` containing the initial phi values for each loop iteration.
* `body`: The `Value` to be executed for each iteration.

The function iterates over the `phis` list, executing the `body` function for each value in `phis`. The phi values from the previous iteration are passed to the `body` function as arguments.

The `VLoop` function returns a `Value` containing the results of the last iteration of the loop.
## function `VBinOp`

Performs binary operations on two values.

**Arguments:**

* `opname`: The name of the binary operation to perform.
* `lhs`: The left-hand side value.
* `rhs`: The right-hand side value.

**Returns:**

* `Value`: The result of the binary operation.

**Functionality:**

The `VBinOp` function takes two values as input and performs the specified binary operation on them. The supported operations are addition (`+`), subtraction (`-`), multiplication (`*`), and division (`/`). The result of the operation is returned as a `Value`.

**Example:**

```python
# Perform addition operation on two integers
result = VBinOp("+", 5, 3)

# Perform subtraction operation on two floats
result = VBinOp("-", 7.5, 2.5)
```
## function `VBinOpIO`

Performs binary operations on input values `io`, `lhs`, and `rhs`. It handles the specified binary operation `op` and returns the result along with the input value `io`.

**Functionality:**

* Takes four arguments:
    * `opname`: Name of the binary operation.
    * `io`: Input value.
    * `lhs`: Left-hand side operand.
    * `rhs`: Right-hand side operand.
* Ensures the input value `io` is valid.
* Converts the values of `lhs` and `rhs` to integers.
* Performs the binary operation specified by `op`.
* Returns a `PackedValues` object containing the result and the input value `io`.
## function `VMergeList`

Merges two values into a single value.

**Arguments:**

* `lhs`: The first value to merge.
* `rhs`: The second value to merge.

**Returns:**

A single value that contains the merged values.
## function `EnvEnter`

The `EnvEnter` function takes an environment and a list of terms as input and returns an environment. It is used to enter a new scope in the program, where the new environment inherits the bindings from the previous environment. The new environment is created by applying the new terms to the previous environment using the `EvalMap` function.
## function `EnvLoop`

`EnvLoop` takes a `ValueList` as input and returns an `Env` object. It is used to create a new environment with the values specified in the `ValueList`.

**Functionality:**

* Creates a new `Env` object.
* Creates a new nested environment with the values specified in the `ValueList`.
* Returns the new environment.

**Example Usage:**

```python
# Create a ValueList with two parameters
param_list = valuelist(Value.Param(0), Value.Param(1))

# Create a new environment with the parameters
env = EnvLoop(param_list)
```
## function `PhiMap`

`PhiMap` takes two `ValueList` as input and returns a new `ValueList` where each element in the first list is paired with the corresponding element in the second list.

**Usage:**

```python
phi_map = PhiMap(left_list, right_list)
```

**Example:**

```python
left_list = ValueList([1, 2, 3])
right_list = ValueList(['a', 'b', 'c'])

phi_map = PhiMap(left_list, right_list)

print(phi_map)  # Output: ValueList([1, 'a'], [2, 'b'], [3, 'c'])
```
## function `VPhi`

The `VPhi` function takes two `Value` objects as input and returns a `Value` object. It is used in the `PhiMap` function to merge two phi nodes. The `VPhi` function is also used in the `Eval` function to evaluate a phi node.
## class `LVA`
## function `LVAnalysis`

The `LVAnalysis` function is used to perform loop variable analysis on a given phi node. It analyzes the phi node and its associated loop variables to determine their relationships and dependencies. The result of the analysis is an `LVA` object, which contains information about the loop variables and their relationships.
## function `LoopIncremented`

```
Increments a loop variable by a specified step.

**Args:**

* `op`: Name of the loop operator.
* `phi`: Loop variable.
* `init`: Initial value of the loop variable.
* `step`: Step value for incrementing the loop variable.
* `res`: Destination for the incremented loop variable.

**Returns:**

* `LVA`: Loop Variable Assignment.
```
## function `LoopIndVar`

`LoopIndVar()` creates an index variable for a loop. It takes four arguments:

* `op`: The loop operator.
* `start`: The starting value of the index variable.
* `stop`: The stopping value of the index variable.
* `step`: The step value of the index variable.

The function returns an `LVA` object, which represents the index variable.
## function `LoopAccumIndVar`

The `LoopAccumIndVar` function takes in five arguments: `op`, `init`, `start`, `stop`, and `step`. It returns an `LVA` object.

This function accumulates the result of an operation within a loop. It takes the following arguments:

* `op`: The operation to perform on each iteration of the loop.
* `init`: The initial value of the accumulator.
* `start`: The starting value of the loop.
* `stop`: The stopping value of the loop.
* `step`: The step size of the loop.

The function iterates through the loop from `start` to `stop` by `step`. For each iteration, it applies the `op` function to the accumulator and the current value of the loop variable. The result is then stored in the accumulator.

After the loop completes, the accumulator contains the accumulated result of the operation.
## function `IsLoopInVariant`

Checks if a given value is a loop invariant. A loop invariant is a property of the loop that remains true throughout the loop. The function returns `True` if the value is a loop invariant, and `False` otherwise.

**Functionality:**

* Takes a single argument, `v`, of type `Value`.
* Returns a boolean value (`Bool`).

**Usage:**

```python
IsLoopInVariant(v)
```

**Example:**

```python
# Check if the variable 'a' is a loop invariant
is_invariant = IsLoopInVariant(a)
```
## function `VSum`

The `VSum` function calculates the sum of a range of values defined by three input parameters:

- `vstart`: The starting value of the range.
- `vstop`: The stopping value of the range.
- `vstep`: The increment or decrement step for each iteration.

The function iterates through the range defined by these parameters and returns the sum of all values in the range.

**Example Usage:**

```python
VSum(vstart=0, vstop=10, vstep=2)
```

**Purpose:**

The `VSum` function is useful for calculating the sum of a sequence of values in a loop or other iterative construct. It simplifies the process of summing a range of values without having to manually calculate each value in the range.
## function `PartialEvaluated`

The `PartialEvaluated` function takes a `Value` as input and returns a `Term`. It is used to represent the result of partially evaluating a term. The function works by first checking if the input value is equal to the result of evaluating the input term in the given environment. If they are equal, the function returns a `PartialEvaluated` term containing the input value.
## function `DoPartialEval`

Evaluates a function term in an environment, partially applying its arguments and replacing them with their evaluated values.

**Functionality:**

* Takes an environment and a function term as input.
* Partially evaluates the function by replacing its arguments with their evaluated values.
* Returns the evaluated function term.

**Usage:**

```python
DoPartialEval(env, func)
```

**Example:**

```python
env = ...  # Initialize environment
func = Term.Func("f", "x + y", Term.Var("x"), Term.Var("y"))  # Define function term

evaluated_func = DoPartialEval(env, func)  # Evaluate function partially

print(evaluated_func)  # Print the evaluated function term
```
## function `GraphRoot`

Returns the root of the graph containing the given term.

This function is used in the rule set to determine the root node of a graph based on a given term. The root node is used to establish the starting point of graph traversal and analysis.
## function `_propagate_RegionDef_from_the_end`

The function `_propagate_RegionDef_from_the_end` is responsible for generating a `RegionEnd` term, which signifies the end of a region definition. It performs the following steps:

- Creates a `Region` object with the given region name, input signature, and arguments.
- Constructs a `RegionEnd` term with the region object, output signature, and a list of terms.
- Evaluates the `RegionEnd` term within the given environment.
- Converts the evaluated terms into a `ValueList` using the `EvalMap_to_ValueList` function.
- Returns the `ValueList` as the result.
## function `_VBranch`

The `_VBranch` function performs conditional value branching based on a given condition. It takes the following parameters:

- `env`: The environment in which the function is executed.
- `cond`: The condition to check.
- `input_terms`: A list of input terms.
- `then_term`: The term to execute if the condition is true.
- `else_term`: The term to execute if the condition is false.
- `va`: The first value to branch.
- `vb`: The second value to branch.

The function works as follows:

1. Evaluates the condition term.
2. Evaluates the then and else terms.
3. Creates a new `VBranch` node with the evaluated condition, then and else terms.
4. Simplifies the `VBranch` node if the condition is `True` or `False`, by replacing it with the corresponding value.
## function `_VLoop`

_VLoop is a ruleset that implements the VLoop operation. It handles the logic of iterating over a value list and applying a body term to each element.

**Functionalities:**

* **Initializes the loop:** Creates a temporary loop structure based on the input terms and body term.
* **Iterates over the value list:** Implements the loop body for each element in the value list.
* **Drops the loop condition:** Removes the condition for continuing the loop.
* **Outputs the results:** Converts the loop results into a value list.
* **Maps phi values:** Transforms phi values used in the loop body.
* **Evaluates expressions in the loop environment:** Creates an environment with phi values set and evaluates expressions within it.

**Usage:**

The `_VLoop` function is called within the `_EvalMap_to_ValueList` function. It takes various parameters related to the loop structure, including the value list, body term, phi values, and loop condition.
## function `_eval_ins_get`

`_eval_ins_get` evaluates the input at the specified index `i` from the `InputPorts` `ins` within the given `Env`. It then sets the corresponding element in the `Vec` of `Value`s `vec_vals` to the result of the evaluation. It ensures that the index is within the bounds of the vector.
## function `_VGetPort`

`_VGetPort` is a function that retrieves a value from a vector based on an index. It accepts four parameters:

* `i`: The index of the value to retrieve.
* `vec_vals`: A vector of values.
* `env`: An environment.
* `term`: A term.

The function performs two rewrite rules:

1. **`Eval(term.getPort(i))` to `VGetPort(Eval(term).toList(), i)`:** This rule evaluates the port at the given index in the term and converts it to a `VGetPort` rule.
2. **`VGetPort(ValueList(vec_vals), i)` to `vec_vals[i]`:** This rule retrieves the value at the given index from the vector. It ensures that the index is within the bounds of the vector.

**In summary, the function `_VGetPort` retrieves a value from a vector based on an index, checking for bounds before accessing the element.**
## function `_Value_rules`

The `_Value_rules` function implements a set of rules for manipulating `Value` objects. It performs the following functionalities:

- **Associativity of the bitwise OR operator:** The function rewrites expressions of the form `a | b` to `b | a`.
- **Merging phi nodes:** The function merges phi nodes, which represent multiple possible values for a variable.
- **Merging loop back phis:** The function handles loop back phis by setting the `IsLoopInVariant` flag to `True`.
## function `_ValueList_rules`

This function defines a set of rules for manipulating `ValueList` objects. It includes operations such as:

* **Appending two ValueLists:** `ValueList(vs1).append(ValueList(vs2))`
* **Simplifying ValueList:** `vl.toValue().toList()`
* **Mapping a function over a ValueList:** `ValueList(vs1).map(fn)`
* **Merging two ValueLists:** `ValueList.Merge(merge_fn, ValueList(vs1), ValueList(vs2))`
* **Converting a ValueList to a Set:** `ValueList(vs1).toSet()`

The function also handles edge cases such as merging an empty list or converting an empty list to a set.
## function `_EvalMap_to_ValueList`

Converts an `EvalMap` to a `Value.List` by iterating over a vector of terms and applying a mapping function to each term.

**Functionality:**

* Takes three arguments:
    * `vec_terms`: A vector of `Term` objects.
    * `env`: An `Env` object.
    * `map_fn`: A function that takes a `Term` as input and returns a `Value`.
* Converts the `EvalMap` to a `TermList` and evaluates it using `partial(Eval, env)`.
* Creates a `ValueList` from the result of applying `map_fn` to each term in `vec_terms`.
* Checks if the vector is empty and generates an empty `ValueList` if it is.
* Otherwise, iterates over the vector, applies `map_fn` to the first term, and appends the result to the `ValueList`.
* Repeats the process for the remaining terms in the vector.
## function `_Debug_Eval`

Checks if the given term evaluates to the given value in the given environment. It does this by applying the `Eval` rule to the term in the environment and then checking if the result is equal to the given value. If the check succeeds, it sets the value of the given term to the given value in the environment.
## function `_EnvEnter_EvalMap`

Transforms an `EnvEnter` term into an `EvalMap` term by entering the environment and mapping each term in the list to an evaluated value.

- Takes two arguments:
    - `terms`: A list of terms to evaluate.
    - `env`: The environment to enter.

- Yields an `Env.nil().nest(EvalMap(env, terms))` term.
## function `_VBinOp_communtativity`

The function `_VBinOp_communtativity` is a rule set that rewrites binary operations in a commutative manner. It iterates over the list of binary operations defined in the codebase and rewrites each operation with the operands in reverse order.

**Functionality:**

- Takes two `Value` objects as input.
- Loops over the list of binary operations defined in the codebase.
- For each operation, rewrites it using the `rewrite()` function with the operands in reverse order.
- Yields the rewritten operation as a `VBinOp` object.
## function `_VBinOp_Lt`

Checks if one integer value is less than another integer value.

**Functionality:**

- Takes two integer values as input.
- Returns `True` if the first value is less than the second value, and `False` otherwise.
- Performs comparisons between constant I64 values.
- Follows the `_VBinOp_communtativity` rule set.

**Usage:**

```python
# Check if i is less than j
result = _VBinOp_Lt(env, Value.ConstI64(i), Value.ConstI64(j))
```
## function `_VBinOp_Pure`

The function `_VBinOp_Pure` performs checks for pure operations in binary expressions. It handles two cases:

* **Constant I64:** If both operands are constant I64 values, it generates a new `VBinOp` expression with the same operator and operands.
* **Constant I64 Initialization:** If one of the operands is a constant I64 value with a specific value, it sets the `has_pure_type` flag for that operand to `True`.

This function is part of a ruleset that includes other rules related to binary operations, such as commutativity and type checking.
## function `_VBinOp_Add`

The `_VBinOp_Add` function performs operations related to addition in a virtual binary operation context. It handles both standard addition and addition with an input/output stream. 

**Functionality:**

- Takes in four arguments:
    - `env`: The virtual environment.
    - `ta`: The first operand.
    - `tb`: The second operand.
    - `i`: The starting index.
    - `j`: The ending index.
    - `io`: The input/output stream.

- Performs three transformations:
    - Transforms `Eval(Term.Add(ta, tb))` to `VBinOp("Add", Eval(env, ta), Eval(env, tb))`.
    - Transforms `Eval(Term.Add(ta, tb))` to `VBinOp("Add", Eval(env, ta), Eval(env, tb))`.
    - Transforms `Eval(Term.AddIO(io, ta, tb))` to `VBinOpIO("Add", Eval(env, io), Eval(env, ta), Eval(env, tb))`.
## function `_LoopAnalysis`

Analyzes loop constructs and identifies loop variables. The function handles three cases:

* **LoopIncremented:** When the loop variable `i` is incremented by a constant value `consti64`.
* **LoopIndVar:** When the loop variable is incremented iteratively less than a bound `n`.
* **LoopAccumIndVar:** When an accumulator variable is incremented within the loop.

The function also assists in finding the phi node associated with the loop. It extracts the loop variables and phi nodes from the loop construct and performs necessary transformations to identify the loop variables and their initialization.
## function `_Eval_Term_Literals`

Converts a `Term.LiteralI64` to a `Value.ConstI64`.

**Arguments:**

* `env`: The environment in which to evaluate the term.
* `i`: The integer value of the literal.

**Yields:**

* A rule that rewrites an `Eval` of a `Term.LiteralI64` to a `Value.ConstI64` with the same integer value.
## function `_PartialEval_rules`

This function defines the rules for partial evaluation. It takes the following parameters:

* `env`: The environment in which the evaluation takes place.
* `term`: The term to be partially evaluated.
* `body`: The body of the function to be partially evaluated.
* `value`: The value to be used for partial evaluation.
* `uid`: The unique identifier of the function.
* `fname`: The name of the function.

The function performs two main tasks:

1. **Creates a new function:** It creates a new function `VFunc` with the same body as the input `body` term.
2. **Updates the term:** It updates the input `term` to be partially evaluated with the value provided.

The function uses the `rewrite` function to perform these transformations and yields both the new function and the updated term.
## function `valuelist`

Creates a new `ValueList` object from the given arguments. If no arguments are provided, it returns an empty `ValueList`.

**Args:**

* `*args`: A variable number of `Value` objects.

**Returns:**

* A new `ValueList` object containing the given arguments.
## function `make_rules`

`make_rules` is a function that creates a set of rules for processing terms in a program. The function takes an optional `communtative` parameter, which determines whether commutative binary operations are included in the set of rules.

The function returns a set of rules, which are used to perform various transformations on terms, such as evaluation, debugging, and loop analysis.

**Functionality:**

* Creates a set of base rules based on predefined functions like `_Value_rules`, `_Eval_Term_Literals`, and so on.
* Includes additional rules for partial evaluation and commutative binary operations if the `communtative` parameter is set to `True`.
* Returns the set of rules, which can be used to process terms in a program.
