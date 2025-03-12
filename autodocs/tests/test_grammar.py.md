## class `Val`
## class `Num`
## class `BinOp`
## class `Add`
## class `Sub`
## class `Mul`
## class `CalcGrammar`
## function `test_calculator`

This function tests the calculator grammar and its ability to perform calculations. It performs the following steps:

1. Asserts that the grammar rules for different operations match.
2. Creates an expression tree using the calculator grammar.
3. Uses a tree rewriter to calculate the result of the expression.
4. Compares the calculated result with the expected result based on the expression.
## function `test_calculator_traverse`

This function tests the calculator functionality by performing mathematical operations on numbers within an expression tree. It uses the `calc` function to traverse the expression and calculate the result. The function asserts that the expected result matches the calculated result.

**Functionality:**

* Creates an expression tree with four operations: addition, subtraction, and multiplication.
* Uses the `calc` function to traverse the expression tree and calculate the result.
* Compares the calculated result with the expected result, which is calculated manually.
* Asserts that the calculated result matches the expected result.

**Purpose:**

* To test the calculator functionality and ensure that it correctly performs mathematical operations.
## class `_VarargVal`
## class `Grouped`
## class `Tuple`
## function `test_vararg`

The `test_vararg()` function demonstrates the use of the `VarargGrammar` class to create and manipulate a sequence of grouped elements. It performs the following steps:

- Creates a `VarargGrammar` instance.
- Writes elements into the tape using the `write()` method.
- Asserts the properties of the written elements, including their head, arguments, and presence in the `vargs` list.
- Uses pattern matching to extract specific information from the elements.
- Demonstrates how the `start` attribute of the grammar can be used to represent different types of elements in a single sequence.
## function `test_three_grammar`

This function tests the functionality of the `ThreeGrammar` class. It creates an instance of the `ThreeGrammar` class and writes three elements to the tape:

* A `Num` object with the value 123
* A `Grouped` object with the head "a" and a single argument, the `Num` object
* An `Another` object with the value of the `Grouped` object

The function then asserts that the `start` attribute of the `ThreeGrammar` class is a tuple containing `_VarargVal`, `Val`, and `Another`.
