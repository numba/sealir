## function `test_bottom`

Tests that calling `tape._read_token(0)` returns `None` on an empty tape. This function is used in the `test_read_record` function to ensure the tape is properly initialized before reading a record.
## function `test_basic`

This function performs various assertions and checks the structure of an expression tree created using the `ase` library. It demonstrates the following functionalities:

* **Expression creation:** The function creates four expressions using the `tp.expr()` method: two `num` expressions, an `add` expression, and a `sub` expression.
* **Assertion of expression structure:** The function asserts the head and arguments of each expression, ensuring that the expressions are created correctly.
* **Parent-child relationships:** The function uses `ase.walk_parents()` to find the parent of each expression. It asserts that the `a` expression is a child of the `c` expression and that the `d` expression is a child of the `a` expression.
* **Assertion of expression containment:** The function uses `ase.contains()` to verify that each expression is contained within its parent.
## function `test_copy_tree`

The `test_copy_tree` function performs the following tasks:

- Creates a new tape (`new_tree`) and copies an expression (`e`) from the original tape (`tp`) into it using the `copy_tree_into` function.
- Asserts that the new tape has a smaller heap and token size compared to the original tape.
- Verifies that the copied expression is a different object from the original expression but has the same pretty-printed string.
## function `test_apply_bottomup`

This function tests the `apply_bottomup` function from the `ase` module. It performs the following steps:

1. Creates an `ase.Tape` object and defines five expressions `a`, `b`, `c`, `d`, and `e`.
2. Initializes an empty buffer to store visited expressions.
3. Creates a `BufferVisitor` class that appends each visited expression to the buffer.
4. Calls `apply_bottomup` with `e` and `BufferVisitor` as arguments, indicating that reachability is not specified.
5. Asserts that the buffer contains the expressions `a`, `b`, `c`, `d`, and `e`.
6. Clears the buffer and calls `apply_bottomup` again without specifying reachability.
7. Asserts that the buffer contains the expressions `a`, `b`, `c`, and `e`.

This test verifies that `apply_bottomup` correctly visits all reachable expressions in the tape, regardless of whether they are directly reachable from the root expression.
## function `test_calculator`

The `test_calculator()` function performs the following functionalities:

* Creates an expression tree with four operations (addition, subtraction, multiplication) on numbers.
* Uses a `Calc` visitor to calculate the result of the expression tree.
* Compares the calculated result with the expected result.

The function performs the following steps:

1. Creates an expression tree with four operations: addition, subtraction, multiplication, and division.
2. Instantiates a `Calc` visitor and applies it to the expression tree using `ase.apply_bottomup`.
3. Calculates the result of the expression tree using the `memo` dictionary of the `Calc` visitor.
4. Defines an `expected()` function to calculate the expected result based on the input numbers.
5. Asserts that the calculated result matches the expected result.
## function `test_calculator_traverse`

This function tests the functionality of the `calc` function. It performs the following steps:

* Creates a tape with four expressions: `a`, `b`, `c`, and `e`.
* Calculates `c` by adding `a` to itself.
* Calculates `d` by subtracting `b` from `c`.
* Calculates `e` by multiplying `b` by `d`.
* Uses the `calc` function to traverse the expression `e`.
* Compares the result of the traversal with the expected result.

The expected result is calculated by performing the operations in order.
