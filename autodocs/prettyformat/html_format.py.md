## function `find_source_md`

Identifies the source metadata for a given SExpr node.

**Parameters:**

* `node`: The SExpr node to search for source metadata.

**Returns:**

* The SExpr node containing the source metadata, or `None` if not found.

**Functionality:**

The `find_source_md()` function uses the `metadata_find_original()` function to search for an SExpr node with the head `PyAst_loc`. This node is typically associated with the location of a Python AST node. If the `PyAst_loc` node is found, the function returns its last argument, which is the SExpr node containing the source metadata. Otherwise, `None` is returned.

**Usage:**

```python
source_md = find_source_md(node)
```
## function `to_html`

The `to_html` function converts an `SExpr` object to an HTML string representation. It handles the following functionalities:

* Traverses the `SExpr` tree using the `ase.walk_descendants_depth_first_no_repeat` function to identify reachable nodes.
* Replaces specific `SExpr` nodes with HTML elements that provide references to their source locations.
* Ensures that referenced `SExpr` nodes are only included once to avoid duplication.
* Renders the `SExpr` tree as an HTML string within a `<div>` element with the class "sexpr-container".

**Note:** The `to_html` function requires the following context from the codebase:

* The `metadata_find_original` function is used to retrieve source location information.
* The `rewrite_generic` method is responsible for generating HTML elements based on the `SExpr` nodes.
## function `prepare_source`

Prepares the given source text by formatting it with HTML tags to create a pre-formatted display within a container.

**Purpose:**

* To format the source text for presentation within an HTML document.

**Functionality:**

* Takes a single argument, `source_text`, which represents the text to be formatted.
* Returns a formatted string within an HTML `<div>` element with the following structure:
    * A `<pre>` element with the id `source-text` to display the source text.
    * A `<div>` element with the class `pre-container` to wrap the `<pre>` element.

**Usage:**

```python
source_text = "This is a sample source text."
formatted_text = prepare_source(source_text)

# Print the formatted text
print(formatted_text)
```

**Output:**

```
<div class='pre-container'>
<pre id='source-text'>This is a sample source text.</pre>
</div>
```
## function `style_text`

The `style_text()` function generates a style sheet for the application. It creates HTML code that defines CSS styles for various HTML elements, including:

* `div.sexpr-container`: A container for sexpr elements.
* `div`: The base element with no background color.
* `div.sexpr`: Style for sexpr elements.
* `div.sexpr .collapsed`: Collapsed sexpr elements.
* `div.handle_origin`: Style for origin handles.
* `div.handle_ref`: Style for reference handles.
* `div.source_info`: Style for source information.

The function returns an HTML string containing the style sheet, which can be used to dynamically style the application based on these CSS rules.
## function `write_html`

Creates an HTML file with the given contents.

**Arguments:**

* `file`: The file to write the HTML to.
* `contents`: A list of strings to be printed within the `<body>` tag.

**Functionality:**

* Creates the HTML header and footer, including `<DOCTYPE html>`, `<html>`, `<head>`, and `<body>` tags.
* Prints the contents of the `contents` argument within the `<body>` tag.
* Appends a `<canvas>` element with the ID `canvas`.

**Usage:**

```python
# Write an HTML file with the contents "Hello, world!"
with open("output.html", "w") as f:
    write_html(f, "Hello, world!")
```
## function `script_text`

This function generates a JavaScript script that sets up event listeners for click and mouseover events on elements with the class `source_info`. When a user clicks on a `source_info` element, the script highlights the corresponding code snippet in the codebase. When a user hovers over a `source_info` element, the script displays a tooltip with the highlighted code snippet.

**Arguments:**

None

**Returns:**

A JavaScript script as a string.

**Usage:**

```
script_text()
```

**Notes:**

* The script assumes the presence of code snippets in elements with the class `source_info`.
* The script uses a third-party library to highlight the code snippets.
* The script adjusts the tooltip position based on the user's scroll position.
