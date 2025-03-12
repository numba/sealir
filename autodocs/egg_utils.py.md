## class `Term`
### function `literal`

The `literal` function is part of the codebase's API for creating an `Expr`. It returns the literal value of an expression.

```python
def literal(self) -> int | str:
    return self.op
```
### function `__repr__`

The `__repr__` function is responsible for the string representation of objects of the `TypeRef` class. It takes no arguments and returns a string in the format `TypeRef(eclass)`.

The function first extracts the `eclass` attribute of the `TypeRef` object. It then checks if there is only one element in the `eclass_ty_map` dictionary for that `eclass`. If there is, it returns the string representation of that element. Otherwise, it returns the `TypeRef` object in the specified format.
## class `_TermRef`
## class `EClassData`
### function `__init__`

Initializes an `EClassData` object from a dictionary of `Term` objects.

- Creates a dictionary of `Term` objects, keyed by their keys.
- Creates a dictionary of sets of `Term` objects, keyed by their eclasses.
- For each `Term` object in the dictionary, adds it to the set of `Term` objects for its eclass.
### function `terms`

Returns the dictionary of terms for this `EClassData` object.

```python
    def terms(self) -> dict[str, Term]:
        return self._terms
```
### function `eclasses`

Returns the set of terms belonging to each eclass. The key of the dictionary is the eclass name, and the value is a set of terms belonging to that eclass.
### function `children_of`

Returns a list of terms that are children of the given term. The children are retrieved from the `children` property of the term, which is a list of term keys. The function iterates over these keys and uses the `terms` dictionary to retrieve the corresponding terms.
### function `find`

Finds all terms in the expression tree with the specified operator name.

- Takes an operator name as input.
- Iterates through all terms in the expression tree.
- Returns an iterator of terms with the specified operator name.
### function `to_networkx`

Converts a graph of terms into a NetworkX DiGraph.

The function takes two arguments:

* `root_term`: The root term of the graph.
* `ignore_types`: A frozenset of term types to ignore when creating edges.

The function creates a NetworkX DiGraph object and adds nodes and edges to represent the graph of terms. The root term is added as a node, and edges are added between each term and its children. The function ignores edges between terms of the types specified in `ignore_types`.

The function returns the NetworkX DiGraph object.
## function `extract_eclasses`

Extracts the eclasses from an `EGraph` object.

**Arguments:**

* `egraph`: An `EGraph` object.

**Returns:**

An `EClassData` object containing the extracted eclasses.

**Functionality:**

1. Serializes the `egraph` using the `serialize()` method.
2. Maps the operations in the serialized graph using the `map_ops()` method.
3. Converts the serialized graph to a JSON object using `json.loads()`.
4. Reconstructs the terms from the JSON object using the `reconstruct()` function.
5. Returns an `EClassData` object containing the reconstructed terms.
## function `reconstruct`

Reconstructs a dictionary of `Term` objects from a dictionary of node data and a dictionary of class data. It performs two passes over the input data:

* **First Pass:** Maps each node key to a `_TermRef` object containing the type, operator, and eclass of the node.
* **Second Pass:** Maps each `_TermRef` to a `Term` object containing the key, type, operator, children, eclass, and cost of the node.

The function returns a dictionary of `Term` objects with keys corresponding to the original node keys.
## class `ECTree`
### function `__init__`

Initializes an object with the given `ecdata`.

This constructor performs the following steps:

* Stores the `ecdata` instance variable.
* Computes the parent relationships for each eclass in the ECTree.
* Computes the depth of each eclass in the ECTree.
### function `_compute_depth`

Computes the depth of each eclass in the ECTree. The depth is the minimum acyclic distance from the eclass to any of its leaf eclasses.

The function iterates over the terms in the `_ecdata` and sets the depth of each leaf eclass to 0. Then, it performs a breadth-first traversal of the ECTree, starting from the leaf eclasses. For each eclass in the frontier set, it calculates its depth based on the depths of its parent eclasses and updates the `depthmap` accordingly. The frontier set is updated to include the parent eclasses of the current eclasses.
### function `_compute_parents`

Computes the parent-child relationships between terms in the `_ecdata` object.

It creates two dictionaries:

- `self._parents`: Maps each term to a set of its parent terms.
- `self._parent_eclasses`: Maps each eclass to a set of its parent eclasses.

The function iterates over each term in `_ecdata` and adds it as a parent to its children in `_parents`. It also adds the eclass of each child to the set of parent eclasses for its parent term in `_parent_eclasses`.
### function `root_eclasses`

The `root_eclasses` function finds the root eclasses in a set of eclasses.
It iterates through the eclasses and checks if their parent eclasses are empty.
If the parent eclass is empty, it means the eclass is a root eclass.
The function returns a set of root eclasses.
### function `leave_eclasses`

The `leave_eclasses` function identifies the eclasses in an egraph that have no parents. It returns a set of eclasses that are not present as keys in the `_parent_eclasses` dictionary.

**Functionality:**

* Creates a set of all eclasses in the `_ecdata` object.
* Iterates through each eclass in the `_ecdata` object.
* Removes the eclasses that are keys in the `_parent_eclasses` dictionary from the set of all eclasses.
* Returns the set of remaining eclasses.

**Usage:**

```python
eclass_data = ...  # Initialize eclass data
eclass_data.leave_eclasses()  # Find leave eclasses
```
### function `write_html_root`

**Description:**

The `write_html_root` function generates an HTML representation of the root elements in a given set of entities. It sorts the entities by depth, and then calls the `write_html_eclass` function for each entity.

**Functionality:**

- Sorts the entities in the `_depthmap` dictionary by depth in descending order.
- Calls the `write_html_eclass` function for each entity.
- Returns a StringIO object containing the generated HTML.

**Usage:**

```python
root_html = write_html_root()
```
### function `write_html_eclass`

Generates HTML representation of an eclass and its associated terms. It skips and draws eclasses based on whether they have more than one member and have already been drawn. The function also writes HTML for each term within the eclass, including their operations and child eclasses.
