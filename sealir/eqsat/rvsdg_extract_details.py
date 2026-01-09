from __future__ import annotations


from dataclasses import dataclass
from typing import Iterable, Iterator, MutableMapping

import sealir.eqsat.rvsdg_eqsat as _rvsdg_ns
from sealir import ase
from sealir.dispatchtable import DispatchTable, dispatchtable
from sealir.rvsdg import Grammar
from sealir.rvsdg import grammar as rg

from .egraph_utils import EGraphJsonDict, parse_type, Qualname


@dataclass(frozen=True)
class Data: ...


_memo_elem_type = ase.value_type | tuple


class _TypeCheckedDict(MutableMapping):
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._check_setitem(key, value)
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def _check_setitem(self, key, value):
        if not isinstance(value, _memo_elem_type):
            raise TypeError(f"{type(value)} :: {value}")


def _Primitive_condition(ns, tyname):
    def f(*args, node_type: Qualname, **kwargs):
        return node_type.prefix == ns and node_type.name == tyname

    return f


class DispatchPrimitive:

    egglog_ns = "egglog.builtins"

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Make sure each subclass has unique dispatch tables.
        cls._dispatch_primitive = cls._dispatch_primitive.copy()

    @dispatchtable
    def _dispatch_primitive(
        self, node_type: Qualname, node: dict, children: tuple, grm: Grammar
    ):
        raise NotImplementedError

    @_dispatch_primitive.case(_Primitive_condition(egglog_ns, "String"))
    def _(
        self, node_type: Qualname, node: dict, children: tuple, grm: Grammar
    ):
        unquoted = node["op"][1:-1]
        return unquoted

    @_dispatch_primitive.case(_Primitive_condition(egglog_ns, "bool"))
    def _(
        self, node_type: Qualname, node: dict, children: tuple, grm: Grammar
    ):
        match node["op"]:
            case "true":
                return True
            case "false":
                return False
            case _:
                raise AssertionError()

    @_dispatch_primitive.case(_Primitive_condition(egglog_ns, "i64"))
    def _(
        self, node_type: Qualname, node: dict, children: tuple, grm: Grammar
    ):
        return int(node["op"])

    @_dispatch_primitive.case(_Primitive_condition(egglog_ns, "f64"))
    def _(
        self, node_type: Qualname, node: dict, children: tuple, grm: Grammar
    ):
        return float(node["op"])

    @_dispatch_primitive.case(_Primitive_condition(egglog_ns, "Vec"))
    def _(self, *args, **kwargs):
        return self._dispatch_Vec(self, *args, **kwargs)


def _Vec_condition(ns, tyname):
    def f(*args, node_type, **kwargs):
        prefix = node_type.param.prefix
        name = node_type.param.name
        return prefix == ns and name == tyname

    return f


class DispatchVec:
    egglog_ns = "egglog.builtins"
    sealir_ns = "sealir.eqsat.rvsdg_eqsat"

    @dispatchtable
    def _dispatch_Vec(
        self, node_type: Qualname, node: dict, children: tuple, grm: Grammar
    ):
        return grm.write(
            rg.GenericList(
                name=node_type.get_fullname(), children=tuple(children)
            )
        )

    @_dispatch_Vec.case(_Vec_condition(sealir_ns, "Term"))
    @_dispatch_Vec.case(_Vec_condition(sealir_ns, "Port"))
    @_dispatch_Vec.case(_Vec_condition(egglog_ns, "String"))
    def _(
        self, node_type: Qualname, node: dict, children: tuple, grm: Grammar
    ):
        return tuple(children)


def condition(op_pattern):
    def f(*args, op, **kwargs):
        return op == op_pattern

    return f


class DispatchValue:
    @dispatchtable
    def _dispatch_Value(self, op: str, children: dict, grm: Grammar):
        raise NotImplementedError

    @_dispatch_Value.case(condition("Value.ConstI64"))
    def _(op: str, children: dict, grm: Grammar):
        val = children["val"]
        return grm.write(rg.PyInt(val))

    @_dispatch_Value.case(condition("Value.Param"))
    def _(op: str, children: dict, grm: Grammar):
        key, idx = children["key"], children["idx"]
        return grm.write(rg.ArgRef(name=key, idx=str(idx)))


def op_matches(op_pattern):
    def condition(self, key, children, grm, op, **kwargs):
        return op == op_pattern

    return condition


def emit_node(fn):
    def wrapper(self, key: str, op: str, children: dict, grm: Grammar):
        return grm.write(fn(self, **children))

    return wrapper


class DispatchTerm:
    @dispatchtable
    def _dispatch_term(self, key, children, grm, op):
        # Fallback to existing handle_Term method for extensions
        extended_handle = self.handle_Term(op, children, grm)
        if extended_handle is not NotImplemented:
            return extended_handle
        return self.handle_unknown(key, op, children, grm)

    # Term operation handlers
    @_dispatch_term.case(op_matches("GraphRoot"))
    def handle_term_graphroot(self, key, children, grm, op):
        term = children["t"]
        return term

    @_dispatch_term.case(op_matches("Term.Func"))
    @emit_node
    def handle_term_func(self, uid, fname, body):
        orig_func: rg.Func = self.lookup_sexpr(int(uid))
        return rg.Func(
            fname=fname,
            args=orig_func.args,
            body=body,
        )

    @_dispatch_term.case(op_matches("Term.RegionEnd"))
    @emit_node
    def handle_term_regionend(self, region, ports):
        return rg.RegionEnd(
            begin=region,
            ports=tuple(ports),
        )

    @_dispatch_term.case(op_matches("Term.IfElse"))
    @emit_node
    def handle_term_ifelse(self, cond, then, orelse, operands):
        return rg.IfElse(
            cond=cond,
            body=then,
            orelse=orelse,
            operands=operands,
        )

    @_dispatch_term.case(op_matches("Term.IO"))
    @emit_node
    def handle_term_io(self):
        return rg.IO()

    @_dispatch_term.case(op_matches("Term.LiteralF64"))
    @emit_node
    def handle_term_literal_f64(self, val):
        return rg.PyFloat(float(val))

    @_dispatch_term.case(op_matches("Term.LiteralI64"))
    @emit_node
    def handle_term_literal_i64(self, val):
        return rg.PyInt(int(val))

    @_dispatch_term.case(op_matches("Term.Param"))
    @emit_node
    def handle_term_param(self, idx):
        return rg.ArgRef(idx=idx, name=str(idx))

    @_dispatch_term.case(op_matches("·.get"))
    @emit_node
    def handle_term_get(self_, **children):
        term, idx = children["self"], children["idx"]
        return rg.Unpack(val=term, idx=idx)

    @_dispatch_term.case(op_matches("·.getPort"))
    @emit_node
    def handle_term_getport(self_, **children):
        term, idx = children["self"], children["idx"]
        return rg.Unpack(val=term, idx=idx)

    @_dispatch_term.case(op_matches("PartialEvaluated"))
    def handle_term_partial_evaluated(self, key, children, grm, op):
        value = children["value"]
        return value

    @_dispatch_term.case(op_matches("Term.Undef"))
    @emit_node
    def handle_term_undef(self, name):
        return rg.Undef(name=name)

    @_dispatch_term.case(op_matches("Term.LiteralBool"))
    @emit_node
    def handle_term_literal_bool(self, val):
        return rg.PyBool(value=bool(val))

    @_dispatch_term.case(op_matches("Term.LiteralStr"))
    @emit_node
    def handle_term_literal_str(self, val):
        return rg.PyStr(value=str(val))

    @_dispatch_term.case(op_matches("Term.LiteralNone"))
    @emit_node
    def handle_term_literal_none(self):
        return rg.PyNone()

    @_dispatch_term.case(op_matches("Term.Loop"))
    @emit_node
    def handle_term_loop(self, body, operands):
        return rg.Loop(
            body=body,
            operands=operands,
        )

    @_dispatch_term.case(op_matches("Term.DbgValue"))
    def handle_term_dbgvalue(self, key, children, grm, op):
        varname = str(children["varname"])
        value = children["value"]
        return grm.write(
            rg.DbgValue(
                name=varname,
                value=value,
                srcloc=grm.write(rg.unknown_loc()),
                interloc=grm.write(rg.unknown_loc()),
            )
        )

    # Dynamic operations (require allow_dynamic_op)
    def dyn_get_condition(self, key, children, grm, op, **kwargs):
        return op == "·.dyn_get" and self.allow_dynamic_op

    @_dispatch_term.case(dyn_get_condition)
    def handle_term_dyn_get(self, key, children, grm, op):
        regionbegin = children["self"]
        idx = int(children["idx"])
        if isinstance(regionbegin, rg.RegionBegin):
            return grm.write(rg.Unpack(val=regionbegin, idx=idx))
        else:
            return NotImplemented


def op_matches(op_pattern):
    def condition(self, key, children, grm, op, **kwargs):
        return op == op_pattern

    return condition


# Dynamic index operation (requires allow_dynamic_op)
def dyn_index_condition(self, key, children, grm, op, **kwargs):
    return op == "·.dyn_index" and self.allow_dynamic_op


class DispatchDynIndex:
    """Initialize dispatch table for RVSDG DynInt operations."""

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Make sure each subclass has unique dispatch tables.
        cls._dispatch_function = cls._dispatch_function.copy()

    @dispatchtable
    def _dispatch_dynint(self, key, children, grm, op):
        raise NotImplementedError(key, children, grm, op)

    @_dispatch_dynint.case(dyn_index_condition)
    def handle_dynint_dyn_index(self, key, children, grm, op):
        termlist = children["self"]
        target = children["target"]
        for i, term in enumerate(termlist):
            if ase.matches(term, target):
                return i
        raise ValueError("cannot find target")

    @_dispatch_dynint.case(op_matches("DynInt"))
    def handle_dynint(self, key, children, grm, op):
        ival = int(children["num"])
        return ival


def node_type_matches(type_name):
    def condition(self, key, children, grm, node_type, **kwargs):
        return node_type.name == type_name

    return condition


class DispatchRVSDG:
    """Initialize dispatch table for RVSDG namespace node types."""

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Make sure each subclass has unique dispatch tables.
        cls._dispatch_rvsdg = cls._dispatch_rvsdg.copy()

    @dispatchtable
    def _dispatch_rvsdg(self, key, children, grm, node_type):
        # Fallback for other RVSDG node types
        nodes = self.gdct["nodes"]
        node = nodes[key]
        op = node["op"]

        return self.handle_unknown(key, op, children, grm)

    # Region handler
    @_dispatch_rvsdg.case(node_type_matches("Region"))
    def handle_region(self, key, children, grm, node_type):
        ins = children["inports"]
        attrs = self.handle_region_attributes(key, grm)
        return grm.write(rg.RegionBegin(inports=ins, attrs=attrs))

    # Term handler - delegate to Term dispatch table
    @_dispatch_rvsdg.case(node_type_matches("Term"))
    def handle_term_dispatch(self, key, children, grm, node_type):
        nodes = self.gdct["nodes"]
        node = nodes[key]
        op = node["op"]
        return self._dispatch_term(
            self, key=key, children=children, grm=grm, op=op
        )

    # TermList handler
    @_dispatch_rvsdg.case(node_type_matches("TermList"))
    def handle_termlist(self, key, children, grm, node_type):
        terms = children["terms"]
        return tuple(terms)

    # PortList handler
    @_dispatch_rvsdg.case(node_type_matches("PortList"))
    def handle_portlist(self, key, children, grm, node_type):
        ports = children["ports"]
        return tuple(ports)

    # Value handler - delegate to existing Value dispatch table
    @_dispatch_rvsdg.case(node_type_matches("Value"))
    def handle_value_dispatch(self, key, children, grm, node_type):
        nodes = self.gdct["nodes"]
        node = nodes[key]
        op = node["op"]
        return self.handle_Value(op, children, grm)

    # InPorts handler
    @_dispatch_rvsdg.case(node_type_matches("InPorts"))
    def handle_inports(self, key, children, grm, node_type):
        names = children["names"]
        return tuple(names)

    # Port handler
    @_dispatch_rvsdg.case(node_type_matches("Port"))
    def handle_port(self, key, children, grm, node_type):
        name = str(children["name"])
        value = children["term"]
        return grm.write(rg.Port(name=name, value=value))

    # DynInt handler - delegate to DynInt dispatch table
    @_dispatch_rvsdg.case(node_type_matches("DynInt"))
    def handle_dynint_dispatch(self, key, children, grm, node_type):
        nodes = self.gdct["nodes"]
        node = nodes[key]
        op = node["op"]
        return self._dispatch_dynint(
            self, key=key, children=children, grm=grm, op=op
        )


def namespace_matches(ns_name):
    def condition(self, key, children, grm, **kwargs):
        nodes = self.gdct["nodes"]
        node = nodes[key]
        eclass = node["eclass"]
        node_type = self._parse_type(self.gdct["class_data"][eclass]["type"])
        return node_type.prefix == ns_name

    return condition


class DispatchFunction:
    """Dispatch table for function-level routing by namespace."""

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Make sure each subclass has unique dispatch tables.
        cls._dispatch_function = cls._dispatch_function.copy()

    @dispatchtable
    def _dispatch_function(self, key, children, grm):
        # Non-RVSDG namespaces - handle with existing logic
        nodes = self.gdct["nodes"]
        node = nodes[key]
        op = node["op"]

        return self.handle_unknown(key, op, children, grm)

    # RVSDG namespace - delegate to RVSDG dispatch table
    @_dispatch_function.case(namespace_matches(_rvsdg_ns.__name__))
    def handle_rvsdg_namespace(self, key, children, grm):
        nodes = self.gdct["nodes"]
        node = nodes[key]
        eclass = node["eclass"]
        node_type = self._parse_type(self.gdct["class_data"][eclass]["type"])
        return self._dispatch_rvsdg(
            self, key=key, children=children, grm=grm, node_type=node_type
        )


def op_matches(op_pattern):
    def condition(self, op, children, grm, **kwargs):
        return op == op_pattern

    return condition


class _EGraphToRVSDG:
    allow_dynamic_op = False
    unknown_use_generic = False
    grammar = Grammar

    _dispatch_primitive: DispatchTable
    _dispatch_Vec: DispatchTable
    _dispatch_Value: DispatchTable
    _dispatch_term: DispatchTable
    _dispatch_py_term: DispatchTable
    _dispatch_handle: DispatchTable
    _dispatch_function: DispatchTable
    _dispatch_rvsdg: DispatchTable

    def __init__(
        self,
        gdct: EGraphJsonDict,
        rvsdg_sexpr: ase.SExpr,
        egg_fn_to_arg_names,
        memo: MutableMapping | None,
    ):
        self.rvsdg_sexpr = rvsdg_sexpr
        self.gdct = gdct
        self.memo = memo if memo is not None else _TypeCheckedDict()
        self.egg_fn_to_arg_names = egg_fn_to_arg_names

    def run(self, node_and_children):
        memo = self.memo
        with self.rvsdg_sexpr._tape as tape:
            grm = self.grammar(tape)
            for key, child_keys in node_and_children:
                if key in memo:
                    last = memo[key]
                else:
                    try:
                        last = memo[key] = self.handle(key, child_keys, grm)
                    except Exception as e:
                        e.add_note(f"Extracting: {key}, {child_keys}")
                        raise
        return last

    def lookup_sexpr(self, uid: int) -> ase.SExpr:
        tape: ase.Tape = self.rvsdg_sexpr._tape
        return Grammar.downcast(tape.read_value(uid))

    def search_calls(self, self_key: str, op: str) -> Iterator[str]:
        nodes = self.gdct["nodes"]
        for k, v in nodes.items():
            children = v["children"]
            if children and children[0] == self_key and v["op"] == op:
                yield k

    def search_method_calls(self, self_key: str, method: str) -> Iterator[str]:
        return self.search_calls(self_key, f"·.{method}")

    def search_eclass_siblings(self, self_key: str) -> Iterator[str]:
        child = self.gdct["nodes"][self_key]
        eclass = child["eclass"]
        for k, v in self.gdct["nodes"].items():
            if self_key != k and v["eclass"] == eclass:
                yield k

    def filter_by_type(
        self, target: str, iterable: Iterable[str]
    ) -> Iterator[str]:
        for k in iterable:
            if self.gdct["nodes"][k]["op"] == target:
                yield k

    def dispatch(self, key: str, grm: Grammar):
        if key in self.memo:
            return self.memo[key]
        assert False, "nothing should use this anymroe"
        node = self.gdct["nodes"][key]
        child_keys = node["children"]
        for k in child_keys:
            self.dispatch(k, grm)
        kind, _, egg_fn = key.split("-")
        if kind == "function":
            arg_names = self.egg_fn_to_arg_names(egg_fn)
            child_keys = dict(zip(arg_names, child_keys, strict=True))
        ret = self.memo[key] = self.handle(key, child_keys, grm)
        return ret

    def get_children(self, key):
        node = self.gdct["nodes"][key]
        return node["children"]

    def handle(
        self, key: str, child_keys: list[str] | dict[str, str], grm: Grammar
    ):
        """Handle node conversion using dispatch table pattern."""
        return self._dispatch_handle(
            self, key=key, child_keys=child_keys, grm=grm
        )

    def handle_primitive(
        self, node_type: Qualname, node, children: tuple, grm: Grammar
    ):
        return self._dispatch_primitive(
            self, node_type=node_type, node=node, children=children, grm=grm
        )

    def handle_Value(self, op: str, children: dict, grm: Grammar):
        return self._dispatch_Value(op=op, children=children, grm=grm)

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        return self.handle_Py_Term(op, children, grm)

    def handle_Py_Term(self, op: str, children: dict | list, grm: Grammar):
        """Handle Python Term operations using dispatch table pattern."""
        return self._dispatch_py_term(self, op=op, children=children, grm=grm)

    def handle_region_attributes(self, key: str, grm: Grammar):
        return grm.write(rg.Attrs(()))

    def handle_unknown(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        if self.unknown_use_generic:
            return self.handle_generic(key, op, children, grm)
        else:
            nodes = self.gdct["nodes"]
            node = nodes[key]
            eclass = node["eclass"]
            node_type = self._parse_type(
                self.gdct["class_data"][eclass]["type"]
            )
            raise NotImplementedError(f"{node_type}: {key} - {op}, {children}")

    def handle_generic(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        assert isinstance(children, dict)
        # flatten children.values() into SExpr
        values = []
        for k, v in children.items():
            if isinstance(v, tuple):
                values.append(grm.write(rg.GenericList(name=k, children=v)))
            else:
                values.append(v)
        return grm.write(rg.Generic(name=str(op), children=tuple(values)))

    def _parse_type(self, typename: str):
        return parse_type(typename)


def key_equals(target_key):
    def condition(self, key, **kwargs):
        return key == target_key

    return condition


def key_startswith(prefix):
    def condition(self, key, **kwargs):
        return key.startswith(prefix)

    return condition


class DispatchHandle:

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Make sure each subclass has unique dispatch tables.
        cls._dispatch_handle = cls._dispatch_handle.copy()

    @dispatchtable
    def _dispatch_handle(self, key, child_keys, grm):
        # Default fallback for unsupported keys
        raise NotImplementedError(key)

    # Special case: common_root
    @_dispatch_handle.case(key_equals("common_root"))
    def handle_common_root(self, key, child_keys, grm):
        values = []
        for k in child_keys:
            val = self.memo[k]
            if isinstance(val, ase.SExpr):
                values.append(val)
        return grm.write(rg.Rootset(tuple(values)))

    # Primitive cases - delegate to existing primitive dispatch table
    @_dispatch_handle.case(key_startswith("primitive-"))
    def handle_primitive_dispatch(self, key, child_keys, grm):
        nodes = self.gdct["nodes"]
        node = nodes[key]
        eclass = node["eclass"]
        node_type = self._parse_type(self.gdct["class_data"][eclass]["type"])

        def get_children():
            memo = self.memo
            if isinstance(child_keys, dict):
                return {k: memo[v] for k, v in child_keys.items()}
            else:
                return [memo[v] for v in child_keys]

        children = get_children()
        return self.handle_primitive(node_type, node, children, grm)

    # Function cases - delegate to function dispatch table
    @_dispatch_handle.case(key_startswith("function-"))
    def handle_function_dispatch(self, key, child_keys, grm):
        memo = self.memo
        if isinstance(child_keys, dict):
            children = {k: memo[v] for k, v in child_keys.items()}
        else:
            children = [memo[v] for v in child_keys]

        return self._dispatch_function(
            self, key=key, children=children, grm=grm
        )


class DispatchPyTerm:

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Make sure each subclass has unique dispatch tables.
        cls._dispatch_py_term = cls._dispatch_py_term.copy()

    @dispatchtable
    def _dispatch_py_term(self, op, children, grm):
        # default fallback
        return NotImplemented

    # Pure Binary Operations

    @_dispatch_py_term.case(op_matches("Py_Add"))
    def handle_py_add(self, op, children, grm):
        lhs, rhs = children["a"], children["b"]
        return grm.write(
            rg.PyBinOpPure(
                op="+",
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_Mul"))
    def handle_py_mul(self, op, children, grm):
        lhs, rhs = children["a"], children["b"]
        return grm.write(
            rg.PyBinOpPure(
                op="*",
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_Div"))
    def handle_py_div(self, op, children, grm):
        lhs, rhs = children["a"], children["b"]
        return grm.write(
            rg.PyBinOpPure(
                op="/",
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_Pow"))
    def handle_py_pow(self, op, children, grm):
        lhs, rhs = children["a"], children["b"]
        return grm.write(
            rg.PyBinOpPure(
                op="**",
                lhs=lhs,
                rhs=rhs,
            )
        )

    # Binary operations with IO
    @_dispatch_py_term.case(op_matches("Py_AddIO"))
    def handle_py_add_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyBinOp(
                op="+",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_SubIO"))
    def handle_py_sub_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyBinOp(
                op="-",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_MulIO"))
    def handle_py_mul_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyBinOp(
                op="*",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_DivIO"))
    def handle_py_div_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyBinOp(
                op="/",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_PowIO"))
    def handle_py_pow_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyBinOp(
                op="**",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    # Inplace operations
    @_dispatch_py_term.case(op_matches("Py_InplaceAddIO"))
    def handle_py_inplace_add_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyInplaceBinOp(
                op="+",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    # Comparison operations
    @_dispatch_py_term.case(op_matches("Py_LtIO"))
    def handle_py_lt_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyBinOp(
                op="<",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_GtIO"))
    def handle_py_gt_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyBinOp(
                op=">",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    @_dispatch_py_term.case(op_matches("Py_NeIO"))
    def handle_py_ne_io(self, op, children, grm):
        io, lhs, rhs = children["io"], children["a"], children["b"]
        return grm.write(
            rg.PyBinOp(
                op="!=",
                io=io,
                lhs=lhs,
                rhs=rhs,
            )
        )

    # Unary operations
    @_dispatch_py_term.case(op_matches("Py_NotIO"))
    def handle_py_not_io(self, op, children, grm):
        io, term = children["io"], children["term"]
        return grm.write(rg.PyUnaryOp(op="not", io=io, operand=term))

    # Attribute and global operations
    @_dispatch_py_term.case(op_matches("Py_AttrIO"))
    def handle_py_attr_io(self, op, children, grm):
        io, obj, attrname = (
            children["io"],
            children["obj"],
            str(children["attrname"]),
        )
        return grm.write(rg.PyAttr(io=io, value=obj, attrname=attrname))

    @_dispatch_py_term.case(op_matches("Py_LoadGlobal"))
    def handle_py_load_global(self, op, children, grm):
        io, name = children["io"], str(children["name"])
        return grm.write(rg.PyLoadGlobal(io=io, name=name))

    # Function call
    @_dispatch_py_term.case(op_matches("Py_Call"))
    def handle_py_call(self, op, children, grm):
        func, io, args = children["func"], children["io"], children["args"]
        return grm.write(rg.PyCall(func=func, io=io, args=tuple(args)))

    # Control flow
    @_dispatch_py_term.case(op_matches("Py_ForLoop"))
    def handle_py_for_loop(self, op, children, grm):
        iter_arg_idx = int(children["iter_arg_idx"])
        indvar_arg_idx = int(children["indvar_arg_idx"])
        iterlast_arg_idx = int(children["iterlast_arg_idx"])
        body, operands = children["body"], children["operands"]
        return grm.write(
            rg.PyForLoop(
                iter_arg_idx=iter_arg_idx,
                indvar_arg_idx=indvar_arg_idx,
                iterlast_arg_idx=iterlast_arg_idx,
                body=body,
                operands=operands,
            )
        )

    # Data structures
    @_dispatch_py_term.case(op_matches("Py_Tuple"))
    def handle_py_tuple(self, op, children, grm):
        elems = tuple(children["elems"])
        return grm.write(
            rg.PyTuple(
                elems=elems,
            )
        )

    # Indexing operations
    @_dispatch_py_term.case(op_matches("Py_SliceIO"))
    def handle_py_slice_io(self, op, children, grm):
        io, lower, upper, step = (
            children["io"],
            children["lower"],
            children["upper"],
            children["step"],
        )
        return grm.write(
            rg.PySlice(io=io, lower=lower, upper=upper, step=step)
        )

    @_dispatch_py_term.case(op_matches("Py_SubscriptIO"))
    def handle_py_subscript_io(self, op, children, grm):
        io, obj, index = children["io"], children["obj"], children["index"]
        return grm.write(rg.PySubscript(io=io, value=obj, index=index))


_dispatches = [
    DispatchPrimitive,
    DispatchVec,
    DispatchValue,
    DispatchTerm,
    DispatchRVSDG,
    DispatchFunction,
    DispatchHandle,
    DispatchPyTerm,
    DispatchDynIndex,
]


class EGraphToRVSDG(*_dispatches, _EGraphToRVSDG):
    pass
