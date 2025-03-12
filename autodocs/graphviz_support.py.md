## function `graphviz_function`

The `graphviz_function` is a decorator that checks if the `graphviz` library is available and wraps the decorated function with additional functionality. If `graphviz` is not available, the decorator raises a `TypeError`.

The decorated function receives a single argument, `fn`, and returns a wrapped function that calls the original function with an additional keyword argument `gv`. This argument provides access to the `graphviz` library within the decorated function.
