## class `Grammar`
### function `__init__`

Initializes a new instance of the `Expr` class.

**Args:**

* `tape`: A `Tape` object.

**Returns:**

None
### function `__init_subclass__`

The `__init_subclass__` function is called when a new subclass is created. It is responsible for handling the case where the `start` attribute of the new subclass is a `UnionType`.

**Functionality:**

* If the `start` attribute is a `UnionType`, it creates a new subclass (`_CombinedRule`) with a `_combined` attribute containing a tuple of the types in the `UnionType`.
* It also creates a `ChainMap` object (`_rules`) that combines the `_rules` attributes of all the types in the `UnionType`.
* Finally, it updates the `start` attribute of the new subclass to point to the newly created `_CombinedRule` subclass.
### function `write`

Creates a NamedSExpr instance from a Trule instance.
- Converts the Trule instance into an expression using the Grammar's tape.
- Checks if the Trule instance is a valid start rule.
- Returns a new NamedSExpr instance with the expression and rule type.
- Raises a ValueError if the Trule instance is not a valid start rule.
### function `downcast`

Converts an `ase.SExpr` to a `NamedSExpr` based on the grammar rules.
It checks if the head of the expression exists as a rule in the grammar and creates a `NamedSExpr` instance accordingly.
Raises a `ValueError` if the head is not valid in the grammar.
### function `__enter__`

The `__enter__` method is a special method used in context managers. It is automatically called when an instance of the class is entered into a `with` statement. In this case, it calls the `__enter__` method of the `_tape` object and then returns the current instance of the class.

**Purpose:**

- Enters the context manager.

**Functionality:**

- Calls the `__enter__` method of the `_tape` object.
- Returns the current instance of the class.

**Usage:**

```python
with MyContextManager() as cm:
    # Code within the context manager
```
### function `__exit__`

The `__exit__` function is called when an exception is raised or when the context manager is exited. It handles any exceptions that may have occurred within the context and ensures that the tape is properly closed.

- It checks if the `_open_counter` is less than or equal to 0. If so, it raises a `MalformedContextError` indicating an invalid context stack.
- It decrements the `_open_counter` to indicate that the context is being exited.
- It calls the `__exit__` method of the underlying tape object (`self._tape`) with the exception information.
## class `_Field`
### function `is_vararg`

Checks if the annotation of a field is a tuple of any type.
Specifically, it uses `isinstance` to determine if the annotation is an instance of the type created by calling `type(tuple[Any])`.
This function is used within the `_Field` class to determine if a field is a vararg field.
## class `_MetaRule`
### function `__instancecheck__`

Checks if an object is an instance of a specific class or subclass.

**Functionality:**

* Checks if the class is a subclass of `_CombinedRule`.
    * If so, it checks if the object is an instance of any of the classes in `_combined`.
* Checks if the object is an instance of `ase.SExpr`.
    * If so, it checks if the `_head` attribute of the object matches the `_sexpr_head` attribute of the class.
* If neither of these conditions is met, it uses the default `__instancecheck__` method.

**Purpose:**

This function is used to determine if an object can be considered an instance of a specific class or subclass. It is used by the `Rule` class to determine if an object can be used as a starting point for a rule.
## class `Rule`
### function `__init__`

Initializes a new instance of the `Rule` class.

**Args:**

* `*args`: Positional arguments. Only a single field is allowed.
* `**kwargs`: Keyword arguments.

**Raises:**

* `TypeError`:
    * If too many positional arguments are provided.
    * If a duplicated keyword is used.
    * If required fields are missing.
### function `__init_subclass__`

Initializes a new grammar rule subclass. It performs the following functionalities:

* Finds the root grammar rule for the subclass.
* Initializes the subclass fields based on type hints.
* Sets up fields, match arguments, sexpr head, and root status for the subclass.
* Inserts the subclass into the grammar rule dictionary (`_rules`).
* Verifies the grammar rule.
### function `__repr__`

The `__repr__` function is responsible for generating a string representation of the object. It takes the following steps:

1. **Get field values:** The `_get_field_values()` method is called to retrieve a dictionary of field names and their corresponding values.
2. **Format arguments:** Each key-value pair from the dictionary is formatted into a string in the format `{field_name}={value!r}`, where `!r` ensures the value is represented in a readable format.
3. **Return representation:** The function returns a string in the format `{object_name}({arguments})`, where `object_name` is the class name and `arguments` is a comma-separated list of formatted field values.
### function `_get_field_values`

Returns a dictionary containing the values of the fields in the `self` object.

The keys of the dictionary are the field objects, and the values are the corresponding field values.

```python
{fd: getattr(self, fd.name) for fd in self._fields}
```
### function `_get_sexpr_args`

Extracts the arguments from an `ase.SExpr` object.

It iterates through the field values of the object, checking if each value is a tuple or a single value. If it's a tuple, it extends the `out` list with the elements of the tuple. Otherwise, it appends the value to the `out` list.

The function returns a list of `ase.value_type` objects containing the extracted arguments.
### function `_verify`

Verifies the validity of a class. Specifically, it checks if any field is marked as `vararg`. Only the last field can be marked as `vararg`. If any field is marked as `vararg` except the last one, a `TypeError` is raised with an appropriate message.
### function `_field_position`

Returns the position of the field with the given `attrname` in the class `cls`.

**Arguments:**

* `cls`: The class containing the field.
* `attrname`: The name of the field.

**Returns:**

* The position of the field.

**Raises:**

* `NameError`: If the field is not found in the class.
## class `_CombinedRule`
## class `NamedSExpr`
### function `_subclass`

Creates a new subclass of `NamedSExpr` with the given grammar and rule.

**Args:**

* `cls`: The class calling the method.
* `grammar`: The type of grammar to use.
* `rule`: The type of rule to use.

**Returns:**

* A new subclass of `NamedSExpr`.
### function `_wrap`

Creates a new instance of the `BasicSExpr` class with the given `tape` and `handle`. It is used to wrap the underlying `ase.BasicSExpr` object.
### function `__init__`

The `__init__` function initializes a new instance of the `NamedSExpr` class. It takes a single argument, `expr`, which is an instance of the `ase.SExpr` class. The function performs the following steps:

- Asserts that the `_grammar` and `_rulety` attributes are set.
- Sets the `_tape` attribute to the `_tape` attribute of the `expr` argument.
- Sets the `_handle` attribute to the `_handle` attribute of the `expr` argument.
- Creates a dictionary `_slots` that maps the argument names of the `_rulety` attribute to their corresponding indices.
- Sets the `_expr` attribute to the `expr` argument.
- Sets the `__match_args__` attribute to the `__match_args__` attribute of the `_rulety` attribute.
### function `__getattr__`

The `__getattr__` function is a special method that is called when an attribute is accessed on an instance of a class. It is used to dynamically retrieve attributes that are not explicitly defined in the class's `__init__` method.

The function takes the name of the attribute as an argument and returns the corresponding value. If the attribute is not found in the instance's `_slots` dictionary, the function raises an `AttributeError`.

If the attribute is found in the `_slots` dictionary, the function gets the index of the attribute and uses it to retrieve the corresponding value from the `_args` list.

If the attribute is the last field in the rule type and is a variadic argument, the function returns a tuple containing the remaining arguments in the `_args` list.
### function `__repr__`

The `__repr__` function is responsible for representing the `SExpr` object as a string. It does this by calling the `repr()` function on the `_expr` attribute and returning the result. This allows for easy debugging and logging of `SExpr` objects.
### function `_head`

Returns the head of the expression.

```python
def _head(self) -> str:
    return self._expr._head
```
### function `_args`

The `_args` function converts a list of `ase.value_type` objects into a tuple of `ase.value_type` objects.

It does this by iterating over the input list and applying a `cast` function to each element. The `cast` function checks if an element is an `ase.SExpr` object and if so, downcasts it using the `_grammar.downcast` method. Otherwise, it simply returns the element.

The function then returns a tuple containing the downcast or unchanged elements.
### function `_get_downcast`

Returns a callable that performs downcasting of an `ase.SExpr` to a `NamedSExpr`. This function is used internally by the `NamedSExpr` class to convert `ase.SExpr` objects into `NamedSExpr` objects.
### function `_replace`

Replaces the arguments in the expression `self._expr` with the given arguments.

The function takes a variable number of arguments of type `ase.value_type`. It then uses the `_grammar.downcast()` method to downcast the arguments to the appropriate type. Finally, it calls the `_expr._replace()` method to replace the arguments in the expression.

The function returns a `NamedSExpr` of type `Tgrammar`.
### function `_bind`

The `_bind` function maps the arguments in the `*args` tuple to the corresponding fields in the `__match_args__` tuple. It returns a dictionary where the keys are the field names and the values are the corresponding arguments.

If the last field in `__match_args__` is a variable-length argument, the remaining arguments in `*args` are combined into a tuple and assigned to the last field.
## class `TreeRewriter`
### function `__init__`

Initializes a new instance of the `TreeRewriter` class.

**Parameters:**

* `grammar`: An optional `Grammar` object.

**Functionality:**

* Checks if the `grammar` argument is not `None`.
* If `grammar` is provided, sets the `grammar` attribute of the instance to the provided `Grammar` object.
* Calls the `super().__init__()` method to initialize the base class.
### function `_default_rewrite_dispatcher`

The `_default_rewrite_dispatcher` function is a helper method used by the `TreeRewriter` class. It handles rewriting of `ase.SExpr` objects based on their head and the corresponding `rewrite_<head>` method. If no such method is found, it uses the generic `rewrite_generic()` method.

**Functionality:**

* Takes three arguments:
    * `orig`: The original `ase.SExpr` object.
    * `updated`: A boolean indicating whether the `orig` object has been updated.
    * `args`: A tuple of additional arguments.
* Checks if the `grammar` attribute is set.
    * If not, asserts that `orig` is a `NamedSExpr`.
    * Otherwise, downcasts `orig` using `grammar`.
* Generates the function name `fname` based on the head of `orig`.
* Obtains the corresponding function `fn` using `getattr`.
    * If `fn` is not None, calls it with `orig` and the unpacked arguments from `args`.
    * Otherwise, calls `rewrite_generic()` with `orig`, `args`, and `updated`.
* Returns the result of the function call or `rewrite_generic()`.
## function `field_position`

Returns the position of the attribute in the grammar rule.

**Args:**

* `grm`: The grammar rule.
* `attr`: The attribute name.

**Returns:**

The position of the attribute in the grammar rule.

**Raises:**

* `NameError`: If the attribute is not found in the grammar rule.
