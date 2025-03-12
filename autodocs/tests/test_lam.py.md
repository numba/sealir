## class `_Val`
## class `Num`
## class `_BinOp`
## class `Add`
## class `Sub`
## class `Mul`
## class `Lt`
## class `Loop`
## class `If`
## class `Func`
## class `Tuple`
## class `UsecaseLamGrammar`
## function `test_curry`

This function tests the currying functionality of the `app_func`. It creates two lambda functions, `lam1` and `lam2`, that take two arguments but only return one. Then, it calls the `app_func` with these lambda functions and some arguments, and asserts that the output matches the expected string representation.

**Key functionalities:**

* Demonstrates currying by defining two lambda functions with different argument order.
* Uses `app_func` to apply the lambda functions to the arguments.
* Asserts that the output matches the expected string representation of the curried functions.
## function `test_lam`

This function performs the following operations within a lambda grammar context:

* Creates a lambda function `lam1` that performs addition and subtraction on two numbers and returns a tuple of the results.
* Applies the lambda function to the numbers 1 and 2, resulting in a lambda application expression.
* Reduces the lambda application expression using beta reduction, resulting in a tuple containing the addition and subtraction results.
* Asserts that the reduced expression matches the expected output.

**Functionality:**

* **Lambda Function Creation:** The `lam1` function takes two arguments, `a` and `b`, performs addition and subtraction, and returns a tuple.
* **Lambda Application:** The `app_func` applies the `lam1` function to the numbers 1 and 2, resulting in a lambda application expression.
* **Beta Reduction:** The `beta_reduction` function reduces the lambda application expression, resulting in a tuple.
* **Assertion:** The function asserts that the reduced expression matches the expected output.
## function `test_lam_loop`

The `test_lam_loop()` function tests a lambda function that implements a loop using a functional approach. It creates a lambda function called `func_body` that calculates the sum of numbers from 0 to `n` using a loop.

**Functionality:**

* The `func_body` function takes one argument, `n`, and initializes a counter variable `i` to 0.
* It checks if `i` is less than `n`. If it is, the function enters a loop.
* In each iteration, the loop increments `i`, adds `i` to the counter `c`, and checks if the loop should continue.
* Once the loop completes, the function returns the value of `c`.

**Output:**

The function prints the lambda function `func_body`, the output of applying the function to an argument of `1`, and the result of beta reduction on the output. The expected output is a tuple containing the sum of numbers from 0 to 1 and the difference between 1 and 1.
## function `test_lam_abstract`

This function performs an abstraction pass on a lambda function, ensuring that all variables are used once. It converts nested functions into single-use variables within the outer function.

**Functionality:**

* Takes a lambda function as input.
* Uses the `UsecaseLamGrammar` to create a tape for the function.
* Applies the abstraction pass using `run_abstraction_pass`.
* Compares the transformed function with the expected function with single-use variables.
* Asserts that the transformed function matches the expected function.

**Example:**

```python
@lam_func(grm)
def func_body(x):
    a = grm.write(Mul(lhs=x, rhs=grm.write(Num(2))))
    b = grm.write(Add(lhs=a, rhs=x))
    c = grm.write(Sub(lhs=a, rhs=grm.write(Num(1))))
    d = grm.write(Mul(lhs=b, rhs=c))
    return d
```

**Transformed function:**

```python
@lam_func(grm)
def func_body(x):
    a = grm.write(Mul(lhs=x, rhs=grm.write(Num(2))))
    b = grm.write(Add(lhs=a, rhs=x))
    c = grm.write(Sub(lhs=a, rhs=b))
    d = grm.write(Mul(lhs=b, rhs=c))
    return d
```
## function `test_lam_abstract_deeper`

The `test_lam_abstract_deeper()` function applies an abstraction pass to a lambda function to ensure all variables are used only once.

**Functionality:**

- The function takes a lambda function as input.
- It uses the `UsecaseLamGrammar` to create a tape for the lambda function.
- The function body is rewritten with an additional layer of abstraction using the `app` function.
- This ensures that each variable is used only once, preventing side effects.
- The function then checks if the rewritten function is equivalent to the original function.

**Purpose:**

- To ensure functional purity in lambda functions.
- To prevent unexpected side effects.
- To improve code readability and maintainability.
## function `test_lam_identity`

The `test_lam_identity` function tests the identity of a lambda function. It verifies that the function takes an argument and returns the same argument.

**Functionality:**

* Creates a lambda function using the `lam_func` decorator.
* Asserts that the function body is a lambda expression.
* Asserts that the formatted string representation of the function body matches the expected string.
* Asserts that the pretty-printed string representation of the function body matches the formatted string representation.
