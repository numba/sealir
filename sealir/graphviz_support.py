from functools import wraps
try:
    import graphviz as gv
except ImportError:
    gv_available = False
else:
    gv_available = True


def graphviz_function(fn):
    if not gv_available:
        raise TypeError("graphviz not available")

    @wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs, gv=gv)
    return wrapped
