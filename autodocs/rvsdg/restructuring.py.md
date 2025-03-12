## class `BakeInLocAsStr`
### function `__init__`

Initializes a new instance of the `LocDirective` class.

**Arguments:**

* `srcfile`: The source file where the location is defined.
* `srclineoffset`: The line offset in the source file.
* `coloffset`: The column offset in the source file.

**Attributes:**

* `srcfile`: The source file where the location is defined.
* `srclineoffset`: The line offset in the source file.
* `coloffset`: The column offset in the source file.
### function `generic_visit`

The `generic_visit` function is responsible for adding location and file information to the AST nodes in a Python program. It performs the following functionalities:

- **Adds location information:** It iterates through the statements in the code and appends an `ast.Expr` node with a `Constant` value containing the start and end line and column offsets of each statement.
- **Adds file information:** For the first statement in a function definition, it appends an `ast.Expr` node with a `Constant` value containing the filename of the source file.
- **Returns the modified AST:** The function returns the modified AST with the added location and file information.
## function `restructure_source`

Restructures an AST (Abstract Syntax Tree) of a function by adding source location information as dangling strings. This information includes the source file path, line offset, and column offset. The restructured AST is then used to generate a Reverse Symbolic Debug Graph (RVSdg) object, which can be used for debugging and other purposes.

**Functionality:**

* Extracts source location information from the function's code.
* Bakes the source location into the AST as dangling strings.
* Transforms the AST to an ASTCFG object and then to an SCFG object.
* Restructures the SCFG object.
* Transforms the restructured SCFG object back to an AST object.
* Converts the AST object to an RVSdg object.

**Usage:**

```python
restructured_rvsdg, debugger = restructure_source(my_function)
```

**Parameters:**

* `function`: The function whose AST should be restructured.

**Returns:**

* `restructured_rvsdg`: The restructured RVSdg object.
* `debugger`: A SourceDebugInfo object that can be used to debug the function.
## class `SourceDebugInfo`
### function `__init__`

Initializes an object with information about source code lines and their corresponding locations.

**Parameters:**

* `source_offset`: The starting offset of the source code lines.
* `src_lines`: A sequence of strings representing the source code lines.
* `inter_lines`: A sequence of strings representing the intermediate source code lines.
* `stream`: The stream to print the output to. Defaults to `sys.stderr`.
* `suppress`: Whether to suppress the output. Defaults to `False`.

**Attributes:**

* `_source_info`: A dictionary mapping line numbers to stripped source code lines.
* `_inter_source_info`: A dictionary mapping line numbers to intermediate source code lines.
* `stream`: The stream to print the output to.
* `suppress`: Whether to suppress the output.

**Purpose:**

The `__init__` function initializes an object with information about source code lines and their corresponding locations. It creates dictionaries that map line numbers to source code lines and intermediate source code lines. It also sets the stream and suppress attributes based on the provided parameters.
### function `show_sources`

Generates a string containing the original and interpolated source information.

The function iterates through the `_source_info` and `_inter_source_info` dictionaries, which contain line numbers and text for each source location. It then formats the information into a string with each line number and text pair aligned in columns.
### function `set_src_loc`

Sets the source location for the object. It takes a `srcloc` argument, which is an instance of the `srcfile` class. The `srcloc` object contains the source file name, line offset, and column offset. The `set_src_loc` method sets the `_srcloc` attribute of the object to the value of the `srcloc` argument.
### function `set_inter_loc`

Sets the internal location of the current source code.

**Args:**

* `interloc`: The location of the source code being interleaved.

**Description:**

This function stores the `interloc` argument in the `_interloc` attribute of the object. The `_interloc` attribute is used by the `show_sources()` function to display the interleaved source code.
### function `show_source_lines`

Prints the source code lines of a specific file range.

**Parameters:**

* None

**Returns:**

* None

**Prints:**

* The filename, line range, and column range of the source code.
* The source code lines within the specified range.
* A marker indicating the column range of the source code.
### function `show_inter_source_lines`

Prints the lines of code within a specific source code range defined by `_interloc`. It iterates through the lines from `line_first` to `line_last` and prints each line along with a marker indicating the range of columns (`col_first` to `col_last`).
### function `setup`

```python
@contextmanager
def setup(self, srcloc=None, interloc=None):
    """
    Sets up the context for running a test.

    Args:
        srcloc: The source location.
        interloc: The intermediate source location.

    Yields:
        None.

    Finally:
        Prints the source lines, intermediate source lines, and a separator.
    """
```
### function `print`

Prints messages to the console with a prefix of `>` if the `suppress` flag is not set.

**Functionality:**

* Takes any number of arguments and keyword arguments.
* Checks if the `suppress` flag is set.
* If `suppress` is False, prints the arguments with a prefix of `>` to the `stream` object.

**Context:**

The `print` function is defined within a context manager called `setup`. The context manager sets the `stream` and `suppress` attributes of the instance.

**Usage:**

```python
# In the context of setup:
print("Hello, world!")  # Output: > Hello, world!

# Outside of setup:
print("This message will not be printed.")  # Output: This message will not be printed.
```
## class `RvsdgizeState`
## class `Scope`
## class `LocDirective`
## class `LocTracker`
### function `__init__`

Initializes a new instance of the `LocDirective` class.

**Parameters:**

* None

**Attributes:**

* `file`: A string representing the filename of the source code. Defaults to "unknown".
* `lineinfos`: A dictionary containing line information.

**Purpose:**

The `__init__` function initializes a new `LocDirective` object with default values for the `file` and `lineinfos` attributes.
### function `get_loc`

Returns a `rg.Loc` object representing the location of the code where this function is called. It extracts the location information from the `lineinfos` dictionary and uses it to construct a new `rg.Loc` object.
## class `RvsdgizeCtx`
### function `scope`

Returns the last element of the `scope_stack`. This is used to access the current scope within the context manager functions `new_block` and `new_function`.
### function `initialize_scope`

Initializes the scope for a given SExpr by iterating over its instructions and creating entries in the `varmap` dictionary. Each entry maps a variable name to a `rg.Unpack` expression that unpacks the value of the SExpr at the corresponding index.
### function `new_function`

Creates a new function scope and pushes it onto the scope stack. The function returns a context manager that will automatically pop the scope from the stack when it is exited.

- Initializes a new `Scope` object with the kind "function".
- Sets the `io` variable in the scope map to the output of `self.grm.write(rg.IO())`.
- Appends the new scope to the `scope_stack`.
- Yields the new scope.
- Finally, pops the new scope from the `scope_stack`.
### function `new_block`

Creates a new scope of type "block" and pushes it onto the `scope_stack`. The scope is associated with the given `node`. The context manager yields the scope and finally pops it from the `scope_stack`.
### function `add_argument`

Adds an argument to the current function scope.

- Takes two arguments:
    - `i`: Index of the argument in the function argument list.
    - `name`: Name of the argument.
- Asserts that the function scope is being used.
- Asserts that the argument name is not already in use.
- Adds the argument to the function scope's `varmap` with a reference to the argument in the grammar.
### function `load_var`

Loads a variable from the current scope or globally.

- Checks if the variable exists in the current scope.
- If it does, returns the variable.
- Otherwise, it creates a `PyLoadGlobal` expression to load the variable globally.
### function `store_var`

Stores a value in the current scope.

- Takes two arguments:
    - `name`: The name of the variable to store.
    - `value`: The value to store.
- If the name is a special internal prefix (`_`), it does nothing.
- Otherwise, it stores the value in the `varmap` dictionary of the current scope.
### function `store_io`

Stores the given `SExpr` value in the "io" variable within the current scope.

**Functionality:**

- Calls the `store_var` method with the name "io" and the provided `SExpr` value.
- This stores the value in the "io" variable, accessible through the `load_io` method.

**Usage:**

```python
# Assuming 'self' is an instance of a class with the store_io method
self.store_io(value)
```
### function `load_io`

Loads the `io` variable from the current scope. If the variable is not found, it is loaded as a global variable. This function is used to access the standard input and output streams.
### function `updated_vars`

Returns a sorted list of unique variable names updated within the given scope list.

**Functionality:**

- Takes a list of `Scope` objects as input.
- Iterates through each scope and updates a set with the variable names from the `varmap` attribute of each scope.
- Returns a sorted list of unique variable names.

**Usage:**

```python
updated_vars = ctx.updated_vars([ctx.scope_map[body]])
```
### function `load_vars`

Loads multiple variables from the current scope or globally if not found.

**Parameters:**

* `names`: A list of variable names to load.

**Returns:**

* A tuple of SExpr objects representing the loaded variables.

**Functionality:**

1. Iterates over the list of variable names in `names`.
2. Calls the `load_var()` function for each name to retrieve the corresponding SExpr object.
3. Returns a tuple containing the loaded SExpr objects.

**Note:**

* This function assumes that the variables being loaded are in the current scope.
* If a variable is not found in the current scope, it will be loaded globally.
### function `insert_io_node`

`insert_io_node` is a method that takes a `grammar.Rule` object as input and performs the following steps:

1. Uses the `grm` object to write the input node.
2. Creates two `rg.Unpack` objects by iterating over the range of 2.
3. Calls `store_io()` with the first `io` object.
4. Returns the second `res` object.

**Purpose:**

The purpose of this function is to prepare an input node for further processing by storing its intermediate results in the `io` object and returning a result object.
### function `read_directive`

Parses and processes directives in the code.

- Reads a `Directive` object.
- Based on the directive's kind:
    - Sets the `file` property of `loc_tracker` for "#file" directives.
    - Parses and updates `lineinfos` in `loc_tracker` for "#loc" directives.
- Raises a `ValueError` for unknown directives.
### function `write_src_loc`

```
Writes the location information of the current source code location using the `loc_tracker` and `grm` objects.
```
## function `unpack_pystr`

Extracts the text from a `rg.PyStr` SExpr. Returns the text if successful, or `None` otherwise.
## function `unpack_pyast_name`

Extracts the name from a `PyAst_Name` SExpr.

**Arguments:**

* `sexpr`: A `SExpr` object.

**Returns:**

* The name as a string.

**Raises:**

* AssertionError: If the `sexpr` is not a `PyAst_Name` SExpr.
## function `is_directive`

Checks if a given string is a valid compiler directive.

**Purpose:**

- Determines if a string starts with the "#file:" or "#loc:" prefixes, indicating compiler directives.

**Functionality:**

- Takes a single argument, `text`, which is a string.
- Returns `True` if `text` starts with either "#file:" or "#loc:", indicating a compiler directive.
- Returns `False` otherwise.

**Usage:**

```python
text = "#file: filename.txt"
is_directive(text)  # Output: True

text = "invalid string"
is_directive(text)  # Output: False
```
## function `parse_directive`

Parses a text string into a `Directive` object.

The function checks if the input text is a valid directive using the `is_directive()` function. If it is, it splits the text into two parts: the directive kind and the content. It then creates a new `Directive` object with these values and returns it. Otherwise, it returns `None`.

```python
def parse_directive(text: str) -> Directive | None:
    if not is_directive(text):
        return

    def cleanup(s: str) -> str:
        return s.strip()

    kind, content = map(cleanup, text.split(":", 1))
    return Directive(kind=kind, content=content)
```
## class `Directive`
## function `get_vars_defined`

`get_vars_defined` takes an `ase.BasicSExpr` object as input and returns a set of variable names defined within the expression. It uses a `VarDefined` visitor to traverse the expression and identifies assignments to variables. The visitor ignores assignments to unnamed variables (`_`) and adds the variable names to a set.
## function `rvsdgization`

This function handles the creation of a RegionEnd node in a generated RegionGraph (RG). It is called when the outputs of a RegionEnd node are modified but its ports are not. The function performs the following steps:

- It checks if the updated outputs match the outputs of the original RegionEnd node.
- If they match, it returns the original RegionEnd node.
- Otherwise, it creates a new RegionEnd node with the updated outputs and ports.
- The ports are determined by checking if the variables used in the RegionEnd node's outputs are present in the context. If a variable is present, its port is used. Otherwise, an Undef node is created.

The `prep_varnames` function is used to convert a list of variable names into a space-separated string. The `fixup` function is used to update the ports of a RegionEnd node based on the updated outputs.
## function `format_rvsdg`

Formats an `SExpr` representation of an RVSDG program into a string representation.

**Purpose:**

The `format_rvsdg` function converts an RVSDG program represented as an `SExpr` into a human-readable string representation. This string representation provides a visual representation of the program's structure and connections.

**Arguments:**

* `prgm`: An `SExpr` representing the RVSDG program.

**Returns:**

A string representation of the RVSDG program.

**Usage:**

```python
rvsdg_program = ...  # Create an RVSDG program as an SExpr
formatted_program = format_rvsdg(rvsdg_program)
print(formatted_program)
```

**Example:**

```
Func main(x y)
    x + y
```

**Note:**

This function is part of an RVSDG compiler and is used to generate the RVSDG program's string representation. It is not intended to be used independently.
## function `convert_to_rvsdg`

Converts a parsed program into an intermediate representation (RVSDG) using the `rvsdgization` formatter.

**Functionality:**

- Takes a parsed program (`prgm`) and a grammar (`grm`) as input.
- Uses the `rvsdgization` formatter to convert the program to an RVSDG representation.
- Inserts metadata into the RVSDG for debugging purposes.
- Returns the RVSDG representation of the program.

**Usage:**

```python
rvsdg_out = convert_to_rvsdg(grm, prgm)
```

**Note:**

- The `_DEBUG` flag controls whether debugging information is printed to the console.
- The `RvsdgizeState` and `RvsdgizeCtx` objects are used to track the state of the RVSDGization process.
