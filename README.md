# S-Expression Array for Lightweight Immutable Representation (SealIR)


## Append-only Array-based Storage for S-Expression

This library provides a data structure for storing and manipulating 
S-expressions in an efficient and immutable manner. S-expressions are 
represented as a flat array (heap) of integers, where each integer either 
represents a token or a reference to another S-expression node in the heap. 
Tokens are stored in a separate dictionary, mapped to negative integer indices 
to differentiate them from S-expression node indices.

### Features

- **Immutable**: S-expressions are immutable, preventing accidental 
  modifications.
- **Flat Storage**: S-expressions are stored in a flat array, allowing for 
  efficient memory usage and fast access.
- **Fast Search**: The flat storage and integer-based representation enable fast
  searching and traversal of the S-expression tree.
- **Non-recursive Processing**: The design eliminates the need for recursive 
  functions to process the S-expression tree, simplifying the code and improving 
  performance.

### Design

The core data structure is the `Tape` class, which manages the heap and token 
dictionary. The tree has the following key features:

- S-expressions are stored in a flat list (heap) of integers, where each 
  integer represents a node or a token.
- Tokens (e.g., integers, strings, etc.) are stored in a separate dictionary, 
  with negative integer keys mapping to the tokens.
- Immutability is enforced by the append-only heap. New nodes is added to the 
  end of the heap. This ensures that for any expression node, its parent node 
  appears later in the heap, while its child nodes appear earlier.
- Traversing the expression tree bottom-up can be achieved by scanning the 
  heap from the start to the position of the root node. By the time a node is 
  visited, all its child nodes must have been visited.
- This design allows for efficient tree traversal without the need for 
  recursive functions.
