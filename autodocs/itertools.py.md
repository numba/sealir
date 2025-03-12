## function `first`

Returns the first element in an iterator.

**Parameters:**

* `iterator`: An iterator of type `T`.

**Returns:**

* The first element in the iterator.

**Example:**

```python
iterator = [1, 2, 3]
first_element = first(iterator)  # first_element will be 1
```
## function `maybe_first`

`maybe_first` is a function that takes an iterator as input and returns the first element of the iterator if it exists, or `None` if the iterator is empty. It is used in the codebase to handle cases where it is necessary to get the first element of an iterator, but it is not guaranteed to exist.
