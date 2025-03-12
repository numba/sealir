## function `pretty_print`

The `pretty_print` function takes an `SExpr` object as input and returns a string representation of the expression. The function handles metadata, variables, and nested expressions, formatting them in a readable way.

**Functionality:**

* Checks if the input expression is metadata. If so, it simply returns the string representation of the expression.
* Creates an `Occurrences` object and applies it to the expression using `apply_bottomup`.
* Creates dictionaries to store formatted expressions and identifiers for repeated variables.
* iterates through the expression's nested expressions, formatting them recursively.
* Returns the formatted string representation of the input expression.
## class `Occurrences`
### function `rewrite_generic`

The `rewrite_generic` function takes three arguments:

- `old`: An SExpr object.
- `args`: A tuple of any objects.
- `updated`: A boolean value.

The function creates a Counter object with the `old` SExpr as its initial element. It then iterates through the `args` tuple and updates the Counter object with any Counter objects found in the tuple. Finally, the function returns the Counter object.

This function is used to update Counter objects based on the provided arguments. It is part of a codebase that performs some transformations on SExpr objects.
