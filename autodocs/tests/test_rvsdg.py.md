## function `test_return_arg0`

The function `test_return_arg0()` takes two integer arguments, `n` and `m`, and returns the first argument, `n`. It defines a nested function `udt()` that takes two integer arguments, `n` and `m`, and returns `n`. It then calls the `run()` function with the `udt()` function and the arguments `(12, 32)`.
## function `test_return_arg1`

The `test_return_arg1` function tests the `udt` function by calling it with the arguments `(12, 32)` and `(32, 12)`. The `udt` function simply returns the second argument (`m`) regardless of the input.
## function `test_simple_add`

This function performs a simple addition operation by taking two integers as input and returning their sum. It uses a nested function `udt` to perform the addition and calls it with the given arguments `(12, 32)`.
## function `test_chained_binop`

Performs a chained binary operation on two integers, `n` and `m`. It calculates `a = n + m * 10` and returns the value of `a`.
## function `test_inplace_add_1`

This function performs an inplace addition operation on the variable `a`. It first calculates the sum of `n` and `m`, then adds `n` to `a` again. Finally, it returns the updated value of `a`.

**Functionality:**

* Calculates the sum of `n` and `m`.
* Adds `n` to the variable `a`.
* Returns the updated value of `a`.

**Example Usage:**

```python
args = (12, 32)
run(udt, args)
```

**Note:**

* The function takes two integer arguments, `n` and `m`.
* It modifies the variable `a` in place.
* The updated value of `a` is returned by the function.
## function `test_inplace_add_2`

This function performs an inplace addition operation on an integer variable `a`. It takes two integer arguments, `n` and `m`, and performs the following steps:

- Sets `a` to the value of `n`.
- Adds `n` and `m` to `a`, effectively doubling its value.
- Returns the updated value of `a`.

The function essentially performs an inplace doubling operation on the input argument `n`.
## function `test_multi_assign`

The `test_multi_assign` function uses the `udt` function to perform multiple assignments in a single line. It takes two integer arguments, `n` and `m`, and returns a tuple containing the values of `a` and `b`.

**Functionality:**

* Calculates the sum of `n` and `m` and assigns the result to both `a` and `b`.
* Returns a tuple containing the values of `a` and `b`.

**Example Usage:**

```python
args = (12, 32)
run(udt, args)
```

**Output:**

The function will return a tuple with the values `(44, 44)`.
## function `test_if_else_1`

The `test_if_else_1` function defines a function called `udt` that takes two integers as input and returns the minimum of the two. It uses an `if-else` statement to determine which value is smaller and stores it in the `out` variable.

**Functionality:**

- Takes two integers as input.
- Returns the minimum of the two integers.
- Uses an `if-else` statement to determine which value is smaller.
## function `test_if_else_2`

The `test_if_else_2` function takes two integer arguments, `n` and `m`, and returns a tuple containing two integers. It determines the smaller of the two numbers and sets the first element of the tuple to it, while setting the second element to the larger number.
## function `test_if_else_3`

The `test_if_else_3` function takes two integer arguments, `n` and `m`, and returns a tuple containing the values of `a` and `b`. 

- If `m` is greater than `n`, then `a` and `b` are set to the value of `m`.
- If `n` is greater than or equal to `m`, then `a` and `b` are set to the value of `n`.
## function `test_if_else_4`

Calculates and returns an integer value based on the values of `n` and `m`. If `m` is greater than `n`, it sets `a` and `b` to `n + m`. Otherwise, it sets `a` and `b` to `n * m`. The function then performs the following calculations:

- Adds `a` and `m`.
- Multiplies `c` by `a`.
- Multiplies `c` by `b`.
- Returns the final value of `c`.
## function `test_while_1`

The `test_while_1()` function uses a `while` loop to calculate the sum of multiples of a given number `m` up to a given number `n`.

- It initializes two variables: `i` to track the current iteration and `c` to store the sum.
- The loop iterates until `i` reaches `n`.
- In each iteration, it calculates `c += i * m` and increments `i`.
- After the loop, it returns a tuple containing the number of iterations (`i`) and the sum (`c`).

**Example Usage:**

```python
args = (5, 3)
run(udt, args)  # Output: (5, 30)

args = (0, 3)
run(udt, args)  # Output: (0, 0)
```
## function `test_range_iterator_1`

The `test_range_iterator_1` function creates an iterator over the range of integers from 0 to `n-1`. It then iterates over the iterator and returns the first two elements.

```python
def test_range_iterator_1():
    def udt(n: int) -> tuple[int, int]:
        it = iter(range(n))
        a = next(it)
        b = next(it)
        return a, b

    args = (5,)
    run(udt, args)
```
## function `test_for_loop_reduce_add_1d`

The function `test_for_loop_reduce_add_1d` calculates the sum of the first `n` natural numbers using a `for` loop.

**Input:**

- `n`: An integer representing the number of natural numbers to add.

**Output:**

- An integer representing the sum of the first `n` natural numbers.

**Functionality:**

- The function defines a nested `for` loop that iterates from 0 to `n`.
- Inside the loop, it initializes a variable `c` to 0 and adds the current value of `i` to it.
- After the loop completes, the function returns the value of `c`, which represents the sum of the first `n` natural numbers.
## function `test_for_loop_reduce_add_2d`

Computes the sum of the elements in a 2D array where the elements are the sum of the row and column indices.

The function iterates over each element in the array using nested loops and accumulates the sum in a variable `c`. If the row index is less than the column index, the element is added to the sum.

**Example:**

```
test_for_loop_reduce_add_2d(5)
```

**Output:**

The function will return 60.

**Note:**

* The function handles edge cases where the input is a 2D array with zero elements.
* The function uses the `range()` function to iterate over the elements in the array.
* The function uses the `break` statement to exit the nested loops if the sum exceeds 20.
## function `test_for_loop_reduce_add_2d_w_break`

Calculates the sum of the products of all pairs of integers `(i, j)` where `i` and `j` are both less than `n`. The summation stops when the value of `c` exceeds 20.
## function `test_for_if_1`

The `test_for_if_1` function defines a nested function `udt` that iterates through a range of integers from 0 to n-1. Within the loop, there's an `if` statement that checks if the current integer `i` is greater than 2. If this condition is met, the variable `s` is initialized with a value of 123. The function then calculates the sum of all integers in the range and returns both the sum and the value of `s` (which is only defined if the loop condition is satisfied).

**Functionality:**

* Iterates through a range of integers from 0 to n-1.
* Checks if the current integer is greater than 2.
* Initializes a variable `s` with a value of 123 if the condition is met.
* Calculates the sum of all integers in the range.
* Returns both the sum and the value of `s`.
## function `test_f_o_r_t_r_a_n`

The `test_f_o_r_t_r_a_n()` function performs a series of calculations and operations based on the input arguments. It uses the `numpy` library for array operations and initializes variables `n` and `t` to 0.

The function follows these steps:

1. Calculates `f` by adding `a` and `b`.
2. Modifies `a` by adding the constant `_FREEVAR`.
3. Creates an array `g` of complex numbers with `c` elements using `np.zeros()`.
4. Calculates `h` by adding `f` and `g`.
5. Calculates `i` as the reciprocal of `d`.
6. Checks if `i` is not zero and performs the following calculations:
    - Calculates `k` by dividing `h` by `i`.
    - Creates an array `l` using `np.arange()` with values from 1 to `c`.
    - Calculates `m` by taking the square root of `l - g`, adding `e * k`, and checking if the first element of `m` is less than 1.
    - If the first element of `m` is less than 1, iterates over `a` and increments `n` until it reaches a value greater than or equal to 3.
    - Calculates `p` by dividing `g` by `l`.
    - Creates an empty list `q`.
    - Iterates over `p` and appends each element to `q`.
    - If the index `r` is greater than 4, sets `s` to 123, `t` to 5, and increments `t` by `s` if it exceeds 122 minus `c`.
7. Returns the sum of `f`, `o`, `r`, `t`, and `a` along with the updated value of `n`.

The function uses variables defined in the context of the codebase, including `_FREEVAR`, `_GLOBAL`, and the `np` library.
## function `test_if_else`

The `test_if_else` function tests the functionality of an anonymous function `udt` that uses an `if-else` statement.

**What it does:**

* The `udt` function takes a single argument `c`.
* Inside the function, it calculates `a = c - 1`.
* If `a` is less than `c`, it calculates `b = a + 2`.
* Otherwise, it doesn't do anything.
* Finally, it returns `b + 3`.

**Purpose:**

The goal of this function is to demonstrate how the `if-else` statement works within an anonymous function.

**Usage:**

```python
test_if_else()
```

**Expected output:**

The function prints the following to the console:

```
res = 12
```

**Note:**

This function relies on external variables and functions from the codebase.
## function `test_for_loop`

Tests the functionality of the `for` loop.

The function uses the `rvsdg` library to restructure the source code of the `udt` function and then evaluates it using the `evaluate` function. It then asserts that the result of the evaluation is equal to the result of directly calling the `udt` function.

The `udt` function calculates the sum of the first `n` natural numbers using a `for` loop.
## function `run`

The `run` function is responsible for testing the functionality of a given function. It performs the following steps:

1. **Restructures the function:** It uses the `rvsdg.restructure_source` function to convert the given function into an intermediate representation (IR).
2. **Evaluates the IR:** It uses the `evaluate` function to execute the IR with the provided arguments and local scope.
3. **Asserts the result:** It compares the result of the evaluation with the expected result obtained by directly calling the function.
4. **Prints the results:** It prints the expected and actual results for debugging purposes.

The function is primarily used for testing functions that involve loops, conditionals, and variable assignments. It ensures that the IR correctly models the function's behavior and correctly evaluates it.
