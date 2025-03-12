## class `CostModel`
### function `__init__`

Initializes a `CostModel` object with the given graph data.

**Args:**

* `graph_data`: A dictionary containing the graph data in JSON format.

**Attributes:**

* `graph_data`: The graph data dictionary.
* `root_eclass`: The root node's eclass.
* `class_data`: A dictionary mapping each eclass to a set of node IDs.
* `nodes`: A dictionary mapping node IDs to `Node` objects.
* `cost_model`: The cost model object.
### function `get_cost_function`

Calculates the total cost for a given node by adding the node's cost to the sum of costs of its child nodes.

**Parameters:**

* `nodename`: The name of the node.
* `op`: The operation of the node.
* `cost`: The cost of the node.
* `nodes`: A dictionary of nodes.
* `child_costs`: A list of child node costs.

**Returns:**

* `float`: The total cost of the node.
## function `egraph_extraction`

`egraph_extraction` extracts an RVSDG (Reduced Variable System Dynamic Graph) from an `EGraph`. It performs the following steps:

1. Serializes the `EGraph` into a JSON dictionary.
2. Finds the root node of the graph.
3. Creates a `CostModel` object based on the graph data.
4. Uses an `Extraction` object to choose the best RVSDG candidate based on cost.
5. Converts the chosen RVSDG candidate to an RVSDG expression using `convert_to_rvsdg`.

The function returns a tuple containing the cost of the extracted RVSDG and the RVSDG expression.
## function `convert_to_rvsdg`

Converts an extended graph into a Reverse Symbolic Data Graph (RVSDG). It iterates through the graph in postorder, extracts argument names for function nodes, and performs the conversion using the provided converter class.

**Args:**

* `exgraph`: A multi-directed graph representing the extended graph.
* `gdct`: A dictionary containing graph data.
* `rvsdg_sexpr`: A symbolic expression representing the RVSDG.
* `root`: The root node of the extended graph.
* `egraph`: An EGraph object.
* `converter_class`: The converter class to use for the conversion.

**Returns:**

The RVSDG represented as a symbolic expression.
## function `get_graph_root`

Identifies and returns the unique identifiers of the graph root nodes in a given `EGraphJsonDict`.

**Functionality:**

* Takes an `EGraphJsonDict` as input.
* Iterates through the `nodes` dictionary within the `graph_json` dictionary.
* Checks if the `op` attribute of each node is equal to `"GraphRoot"`.
* If the condition is met, the identifier of the node is added to the `roots` set.
* Returns the set of unique identifiers for the graph root nodes.

**Usage:**

The `get_graph_root()` function can be used to extract the root nodes of a graph from an `EGraphJsonDict` object.

**Example:**

```python
graph_json = ...  # An EGraphJsonDict object
roots = get_graph_root(graph_json)

print(roots)  # Print the unique identifiers of the graph root nodes
```
## class `Node`
## class `Extraction`
### function `__init__`

The `__init__` function initializes an `Extraction` object with the provided graph JSON, root eclass, and cost model.

- It extracts node data from the graph JSON and creates a dictionary of `Node` objects.
- It stores the root eclass and cost model in the object.
- It builds a `class_data` dictionary that maps each eclass to a set of node keys.
### function `_compute_cost`

Computes cost for the EGraph using dynamic programming with iterative cost propagation.

**Time complexity:** O(N * M) where N is the number of iterations and M is the number of nodes in the graph.

**Functionalities:**

- Iterates through all nodes in the graph for each iteration.
- Computes costs based on children of each node.
- Updates selections dictionary with the best cost and node name for each eclass.
- Terminates early if a finite root score is computed.

**Usage:**

- The function is called with a maximum number of iterations (default: 10000).
- It returns a dictionary mapping eclasses to Buckets containing the best cost and node name.
### function `choose`

The `choose()` function makes a selection of nodes in a graph based on the costs associated with each node. It returns a tuple containing the root cost and the selected graph.

**Functionality:**

- It calculates the costs for each node in the graph.
- It selects the node with the lowest cost as the root node.
- It iterates through the children of the root node, selecting the node with the lowest cost.
- It creates a new graph with the selected nodes.
- The function returns the root cost and the selected graph.
## function `sentry_cost`

Checks if the given cost is infinite. If so, it raises a `ValueError` with the message "invalid cost extracted". Otherwise, it does nothing.

This function is used within the `get_cost_function` and `compute_cost` functions to ensure that only valid costs are used in the cost calculations.
## class `Bucket`
### function `__init__`

Initializes a new instance of the `Bucket` class.

**Purpose:**

- Creates a new `Bucket` object.

**Functionality:**

- Initializes a dictionary `data` with default values set to `MAX_COST`.
- This dictionary stores costs associated with different keys.

**Usage:**

```python
bucket = Bucket()
```
### function `put`

Updates the cost associated with a given key in the `Bucket` object. If the key already exists, the cost is updated to be the minimum of the current cost and the new cost.

**Arguments:**

* `cost`: The new cost associated with the key.
* `key`: The key for which the cost should be updated.

**Returns:**

* None
### function `best`

Returns the key with the lowest cost in the `data` dictionary.

**Usage:**

```python
best_key, best_cost = bucket.best()
```

**Return Value:**

A tuple containing the key and the lowest cost.
### function `__len__`

Returns the length of the `data` attribute of the object. It uses the built-in `len()` function to determine the length.
### function `__repr__`

The `__repr__` function is a special method that defines how an object should be represented as a string. In this case, it is used to represent instances of the `Data` dataclass.

The function takes no arguments and returns a string in the following format:

```
<class_name>([...])
```

where:

* `<class_name>` is the name of the class, which is `Data` in this case.
* `[...]` contains a sorted list of key-value pairs from the `data` dictionary of the object.

The function uses the `pformat` function to format the key-value pairs in a human-readable way.
