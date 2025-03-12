## function `test_rewrite`

This function tests the `RewriteCalcMachine` class, which is used to rewrite arithmetic expressions. The machine performs the following operations:

* Addition: `x + y` is rewritten as `num(x + y)`.
* Subtraction: `x - y` is rewritten as `num(x - y)`.
* Multiplication: `x * y` is rewritten as `num(x * y)`.

The function creates an expression tree with four operations and then applies the `RewriteCalcMachine` to it. It asserts that the result is as expected and that the reduced expression is a copy of the original expression.
