## function `llvm_codegen`

The `llvm_codegen` function takes an `ase.SExpr` object and a dictionary of namespace mappings as input. It emits LLVM using the Python C-API and returns a `JitCallable` object.

**Purpose:**

The function generates LLVM code for an expression tree defined by the `ase.SExpr` object. The generated code includes a function that takes a list of Python objects as arguments and returns a single Python object.

**Functionality:**

1. Initializes the LLVM library and sets up a module.
2. Creates a function with the appropriate signature based on the number of arguments in the expression tree.
3. Compiles the LLVM code into an executable bitcode.
4. Creates a just-in-time compiler and loads the bitcode.
5. Returns a `JitCallable` object that can be called with Python objects.

**Warning:**

- The function is intended for testing purposes only.
- It does not perform memory management or error handling properly.
## class `JitCallable`
### function `from_pointer`

Creates a new instance of `JitCallable` from a C function pointer.

**Arguments:**

* `rt`: The `llvm.ResourceTracker` instance.
* `ptr`: The address of the C function pointer.
* `arity`: The number of arguments the function takes.

**Returns:**

A new instance of `JitCallable`.

**Usage:**

```python
# Create a JitCallable object from a C function pointer
jit_callable = JitCallable.from_pointer(rt, ptr, arity)

# Call the function using the JitCallable object
result = jit_callable(*args)
```
### function `__call__`

The `__call__` method of the `JitCallable` class acts as a callable interface for the underlying Python function. It takes any number of arguments and calls the `pyfunc` attribute with these arguments.
## class `CodegenState`
## function `_codegen_loop`

Generates code for a loop expression.

**Parameters:**

* `expr`: The loop expression.
* `state`: The codegen state.

**Returns:**

* A `PackedValues` object containing the values of the loop outputs.

**Description:**

The `_codegen_loop` function handles loop expressions in the `ase` language. It receives the loop expression and the codegen state as input and generates the corresponding code.

**Usage:**

```python
# Example usage:
loop_expr = ...  # The loop expression
codegen_state = ...  # The codegen state
loop_output = _codegen_loop(loop_expr, codegen_state)
```

**Notes:**

* The function uses the `ensure_io` function to ensure that the input and output values are `IOState` objects.
* The function uses the `debug_packs` function to print debugging information about the loop outputs.
* The function supports the following loop types:
    * `rg.RegionEnd`: The end of a loop region.
    * `rg.IO`: An input or output value.
## class `CodegenCtx`
## function `determine_arity`

Determines the arity of a function node. The arity is the number of arguments the function takes.

**Arguments:**

* `root`: An `ase.SExpr` node representing a function node.

**Returns:**

* An integer representing the arity of the function node.

**Raises:**

* `TypeError`: If the `root` node is not a function node.
## class `LLVMValue`
## class `IOState`
## class `SSAValue`
### function `__post_init__`

Ensures that the `value` attribute of the `SSAValue` instance is an instance of the `ir.Value` class.
## class `PackedValues`
### function `make`

Creates a new instance of `PackedValues` with the given arguments.

Args:
    *args: The arguments to pass to the constructor.

Returns:
    A new instance of `PackedValues`.
```
### function `__post_init__`

Ensures that all values in the `values` tuple of a `PackedValues` object are either `SSAValue` or `IOState` objects. It raises an `AssertionError` if any value is not of these types.
### function `__getitem__`

Returns the value at the given index in the `values` tuple.

**Signature:**

```python
def __getitem__(self, idx: int) -> SSAValue:
```

**Arguments:**

* `idx`: The index of the value to retrieve.

**Returns:**

* An `SSAValue` object.

**Usage:**

```python
packed_values = PackedValues(ssa1, ssa2, ...)
value = packed_values[0]
```
### function `__len__`

Returns the length of the `values` list.
## function `handle_error`

The `handle_error` function is a decorator that handles errors in the code. It checks if the `_EMIT_ERROR_HANDLING` flag is set to `True`. If it is, then the function wraps the function it is applied to and checks if the result is equal to `None`. If it is, then the function returns the result. Otherwise, the function continues to execute the function and returns the result.

If the `_EMIT_ERROR_HANDLING` flag is not set to `True`, then the function simply returns the function it is applied to.
## function `handle_error_negone`

The `handle_error_negone` function is a decorator that handles errors by checking if the result of the decorated function is -1. If it is, the function returns `None`. Otherwise, it returns the original result.

The function is used to handle errors in situations where the function may return -1 to indicate an error. This is useful for functions that are called from Python code, as Python does not have native support for error handling.
## class `PythonAPI`
### function `__init__`

Initializes a new `PythonAPI` instance with the given `builder`.

**Functionality:**

* Initializes various types used in the code.
* Creates a unique mapping of serialized objects for the module.
* Sets up an `IRBuilder` for code generation.
* Initializes the return value to `None`.
### function `_get_function`

`_get_function` is a helper function that retrieves or creates a new function with the given type `fnty` and name `name`. It is used in the `_long_from_native_int`, `incref`, and `decref` functions to dynamically generate functions based on the provided type and name.

**Functionality:**

* Takes two arguments:
    * `fnty`: The function type.
    * `name`: The name of the function.
* Uses `_get_or_insert_function` to check if a function with the given name already exists in the module.
* If the function doesn't exist, it creates a new function with the given type and name.
* Returns the retrieved or newly created function.
### function `_long_from_native_int`

Converts a native integer value to a Python long object.

**Parameters:**

* `ival`: The native integer value.
* `func_name`: The name of the corresponding C function.
* `native_int_type`: The type of the native integer value.
* `signed`: Whether the native integer value is signed or unsigned.

**Returns:**

A Python long object containing the converted value.
### function `long_from_long`

Converts an integer value to a Python long object.

**Parameters:**

* `ival`: The integer value to convert.

**Returns:**

A Python long object corresponding to the input integer.

**Usage:**

```python
long_obj = long_from_long(1234)
```
### function `long_from_ulong`

Converts an unsigned long integer into a Python long integer.

**Args:**

* `ival`: The unsigned long integer to convert.

**Returns:**

A Python long integer representation of the input value.
### function `long_from_ssize_t`

Converts an `ssize_t` value to a Python integer. It calls the `_long_from_native_int()` function with the following arguments:

* `ival`: The `ssize_t` value to convert.
* `func_name`: "PyLong_FromSsize_t"
* `native_type`: `py_ssize_t`
* `signed`: `True`
### function `long_from_longlong`

Converts a 64-bit integer value to a Python integer. The function uses the `PyLong_FromLongLong` function from the CPython API to create a new Python integer object.

**Parameters:**

* `ival`: A 64-bit integer value.

**Returns:**

* A new Python integer object.
### function `long_from_ulonglong`

Converts an unsigned long long integer into a Python `long` object.

**Parameters:**

* `ival`: The unsigned long long integer to convert.

**Returns:**

A Python `long` object containing the value of `ival`.
### function `long_from_signed_int`

Converts a native integer value to a Python integer. It handles different integer sizes and returns an appropriate Python integer type.

- If the native integer size is within the range of `long`, it converts it using `long_from_long`.
- If the size is within the range of `longlong`, it converts it using `long_from_longlong`.
- If the size exceeds these ranges, it raises an `OverflowError`.
### function `long_from_unsigned_int`

Converts an unsigned integer value to a Python integer.

**Functionality:**

* Takes an unsigned integer value as input.
* Determines the bit width of the input value.
* If the bit width is less than or equal to the width of `ulong`, converts the value using `long_from_ulong`.
* If the bit width is less than or equal to the width of `ulonglong`, converts the value using `long_from_ulonglong`.
* Raises an `OverflowError` if the bit width exceeds the maximum supported range.

**Usage:**

```python
long_from_unsigned_int(ival)
```

**Note:**

This function is similar to `long_from_signed_int`, but it handles only unsigned values.
### function `float_from_double`

Converts a double value to a Python float object using the `PyFloat_FromDouble` function.

**Functionality:**

* Takes a double value as input.
* Creates a function type that accepts a double argument and returns a Python object.
* Finds the `PyFloat_FromDouble` function using the `_get_function` method.
* Calls the function with the double value and returns the resulting Python float object.
### function `_get_number_operator`

Retrieves a function named `PyNumber_%s` from the IR module. This function is used for performing arithmetic operations on Python numbers. The function takes two arguments, `self` and `name`, and returns a function object.

**Parameters:**

* `self`: The `self` parameter is a reference to the current object.
* `name`: The name of the function to retrieve.

**Returns:**

* A function object.
### function `_call_number_operator`

Calls the specified mathematical operator on two numbers.

**Parameters:**

* `name`: The name of the operator.
* `lhs`: The first number.
* `rhs`: The second number.
* `inplace`: Whether to perform the operation in place.

**Returns:**

* The result of the operator.

**Functionality:**

* Takes the operator name, left-hand side (lhs), and right-hand side (rhs) as input.
* Checks if the `inplace` flag is set. If so, it prepends "InPlace" to the operator name.
* Retrieves the corresponding operator function using `_get_number_operator`.
* Calls the operator function with the lhs and rhs as arguments.
* Returns the result of the operator call.

**Usage:**

This function is used by other functions in the codebase to perform mathematical operations on numbers. For example, `number_add` uses `_call_number_operator` to call the "Add" operator.
### function `number_add`

Performs addition between two numbers.

**Arguments:**

* `lhs`: The first number.
* `rhs`: The second number.
* `inplace`: Whether to modify the first number in place.

**Returns:**

The sum of the two numbers.
### function `number_subtract`

Subtracts two numbers and returns the result.

**Args:**

* `lhs`: The first number.
* `rhs`: The second number.
* `inplace`: Whether to modify the first argument in place.

**Returns:**

* The result of the subtraction.
### function `number_multiply`

Multiplies two numbers.

**Parameters:**

* `lhs`: The first number.
* `rhs`: The second number.
* `inplace`: Whether to perform the operation in place (optional, default False).

**Returns:**

The result of the multiplication.
### function `number_truedivide`

Divides two numbers with true division, returning a floating-point result.

**Arguments:**

* `lhs`: The left-hand side operand.
* `rhs`: The right-hand side operand.
* `inplace`: Whether to perform the operation in place (default: False).

**Returns:**

* A floating-point number representing the result of the division.
### function `number_floordivide`

Performs floor division on two numbers.

**Args:**

* `lhs`: The first number.
* `rhs`: The second number.
* `inplace`: Whether to modify the first number in place (default: False).

**Returns:**

The result of the floor division operation.
### function `number_power`

Calculates the power of two numbers using the `PyNumber_Power` function.

- Takes three arguments:
    - `lhs`: The base number.
    - `rhs`: The exponent.
    - `mod` (optional): The modulus (default is `None`).
- Asserts that `mod` is `None` as three-argument `pow()` is not implemented.
- Calls the `PyNumber_Power` function with the arguments `lhs`, `rhs`, and `mod`.
- Returns the result of the function call.
### function `number_negative`

This function takes a Python object as an argument and returns a new Python object of the same type as the input object. The new object will have the negative value of the input object.
```python
def number_negative(self, operand):
    fnty = ir.FunctionType(self.pyobj, [self.pyobj])
    fn = self._get_function(fnty, name="PyNumber_Negative")
    return self.builder.call(fn, [operand])
```
### function `object_richcompare`

The `object_richcompare` function performs rich comparisons between two Python objects based on the specified operator (`opstr`). It handles various operators, including equality, inequality, containment, and identity checks.

**Functionality:**

* **Operator Handling:**
    * Supports comparison operators (<, <=, ==, !=, >, >=)
    * Implements "is" and "is not" operators for object identity comparison
    * Handles "in" and "not in" operators for membership checks

* **Object Comparisons:**
    * Calls the `PyObject_RichCompare` function to perform rich comparisons
    * Uses the `PySequence_Contains` function for "in" and "not in" checks

* **Return Value:**
    * Returns a boolean object indicating the result of the comparison
    * Returns an error object if an unsupported operator is encountered
### function `object_istrue`

Checks if a given Python object is True.

It calls the `PyObject_IsTrue` function to determine if the object is True. The result is a boolean value.
### function `object_not`

The `object_not` function performs bitwise NOT operation on the given object. It uses the `PyObject_Not` C function to achieve this functionality.

**Input:**

* `obj`: The object to perform bitwise NOT operation on.

**Output:**

* The result of the bitwise NOT operation.
### function `tuple_pack`

Creates a new Python tuple from a list of items.

**Arguments:**

* `items`: A list of Python objects.

**Returns:**

* An `SSAValue` representing the newly created tuple.

**Usage:**

```python
elts = []
for el in elems:
    elts.append((yield el).value)
return SSAValue(pyapi.tuple_pack(elts))
```
### function `bool_from_bool`

Converts an LLVM boolean value to a Python boolean.

- Takes an LLVM boolean as input.
- Extends the boolean value to a long integer.
- Calls the `bool_from_long` function to convert the long integer to a Python boolean.
- Returns a Python boolean value.
### function `bool_from_long`

Converts an integer value to a Python boolean.
- Takes an integer value `ival` as input.
- Converts the integer to a long value using `self.builder.zext`.
- Calls the `PyBool_FromLong` function with the long value to create a Python boolean object.
- Returns the newly created Python boolean object.
### function `call_function_objargs`

Calls the `PyObject_CallFunctionObjArgs` function with the provided `callee` and `objargs`.

**Parameters:**

* `callee`: The function to call.
* `objargs`: The arguments to pass to the function.

**Returns:**

The result of the `PyObject_CallFunctionObjArgs` function call.
### function `string_from_string`

Converts a Python string into a C `PyUnicode` object. It takes a Python string as input and returns a C `PyUnicode` object. The function uses the `PyUnicode_FromString` function to perform the conversion.
### function `object_getattr_string`

This function retrieves the value of an attribute of an object using the `PyObject_GetAttrString` C function. It takes two arguments:

* `obj`: The object to get the attribute from.
* `attrname`: The name of the attribute to get.

The function first creates a function type with two arguments, `obj` and `attrname`. It then calls the `_get_function` method to get the `PyObject_GetAttrString` function. Finally, it calls the function with the `obj` and `attrname` arguments and returns the result.
### function `make_none`

Creates a new Python `None` object. It borrows the existing `_Py_NoneStruct` object, increments its reference count, and returns it.
### function `borrow_none`

Returns a borrowed reference to the `_Py_NoneStruct` C object. This object represents the Python `None` value. The returned object should not be modified or destroyed.
### function `get_c_object`

Retrieves a Python object through its C-accessible name. The underlying variable must be a `PyObject *`, and the value of that pointer is returned. This is useful for accessing global variables and objects defined in C.

**Parameters:**

* `name`: The C-accessible name of the object to retrieve.

**Returns:**

A Python object corresponding to the C-accessible name.
### function `incref`

Increments the reference count of a Python object. This function calls the `Py_IncRef` C function, which increments the reference count of the object passed as an argument. This is necessary to prevent the object from being garbage collected before it is no longer being used.
### function `decref`

Decrements the reference count of an object.

This function uses the `Py_DecRef` C function to decrease the reference count of the specified object. It is typically used to release memory associated with objects when they are no longer needed.
### function `dump`

Dumps the given Python object to a byte stream.

- Takes a single argument, `obj`, which is a Python object.
- Calls the `_PyObject_Dump` function with the given object as an argument.
- The `_PyObject_Dump` function is responsible for serializing the object to a byte stream.
### function `printf`

Prints a formatted string to the console.

**Parameters:**

* `fmt`: A string containing the format specification.
* `*args`: Variable arguments to be formatted and inserted into the string.

**Usage:**

```python
obj.printf("Hello, %s!", "World")
```

This will print the following to the console:

```
Hello, World!
```
### function `_get_cstring`

Converts a Python string into a C string and creates a global variable in the IR module with the encoded string.

**Functionality:**

- Takes a Python string as input.
- Encodes the string with UTF-8 encoding and adds a null terminator.
- Creates a constant array of bytes with the encoded string.
- Generates a unique name for the global variable.
- Creates a global variable with the constant array and appropriate linkage and constants.
- Bitcasts the global variable to a C string type.

**Returns:**

- An `SSAValue` representing the C string.
## function `_get_or_insert_function`

Inserts a new function into a module if it doesn't already exist.

- Takes three arguments:
    - `module`: The module in which to insert the function.
    - `fnty`: The type of the function.
    - `name`: The name of the function.
- Checks if a function with the given name already exists in the module.
- If the function does not exist, it creates a new function with the given type and name.
- Returns the function, either the existing or newly created function.
## function `_get_c_value`

Creates a global variable named `name` with type `typ`. If the variable doesn't exist, it is created with a unique name. If `dllimport` is set to `True`, the storage class of the variable is set to `dllimport` to allow access from external modules. The function returns the global variable.
## function `_alloca_once_value`

Creates a new alloca slot for the given value if it hasn't been allocated yet.

**Functionality:**

- Allocates a new alloca slot with the same type as the input value.
- Stores the input value in the newly allocated slot.
- Returns the alloca slot.

**Usage:**

```python
# Example usage:
value = ...  # Some value
slot = _alloca_once_value(builder, value)
```
## function `_handle_binop`

Handles binary operations between two values. It accepts an operator `op`, and two values `lhsval` and `rhsval`. It then performs the corresponding operation and returns the result.

**Supported Operators:**

- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- True division (`/`)
- Floor division (`//`)
- Exponentiation (`**`)
- Comparison operators (`<`, `>`, `==`, `!=`, `in`)

**Inplace Operations:**

For operations marked with `io=True` in the `rg.PyInplaceBinOp` case, the result is assigned to the first operand (`lhsval`).

**Raises:**

- `NotImplementedError`: If the operator is not supported.
