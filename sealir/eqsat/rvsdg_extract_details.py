from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, MutableMapping

import sealir.eqsat.rvsdg_eqsat as _rvsdg_ns
from sealir import ase
from sealir.matcher import Matcher
from sealir.rvsdg import Grammar
from sealir.rvsdg import grammar as rg

from .egraph_utils import EGraphJsonDict, parse_type


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


class EGraphToRVSDG:
    allow_dynamic_op = False
    unknown_use_generic = False
    grammar = Grammar

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
        if key == "common_root":
            # legalize child
            values = []
            for k in child_keys:
                val = self.memo[k]
                if isinstance(val, ase.SExpr):
                    values.append(val)
            return grm.write(rg.Rootset(tuple(values)))

        allow_dynamic_op = self.allow_dynamic_op

        nodes = self.gdct["nodes"]
        memo = self.memo

        node = nodes[key]
        eclass = node["eclass"]
        node_type = self._parse_type(self.gdct["class_data"][eclass]["type"])

        def get_children() -> dict | list:
            if isinstance(child_keys, dict):
                return {k: memo[v] for k, v in child_keys.items()}
            else:
                return [memo[v] for v in child_keys]

        if key.startswith("primitive-"):
            return self.handle_primitive(node_type, node, get_children(), grm)
        elif key.startswith("function-"):
            op = node["op"]
            children = get_children()

            if node_type.prefix != _rvsdg_ns.__name__:
                raise AssertionError("rvsdg namespace mismatch")

            match node_type.name, children:
                case "Region", {"inports": ins}:
                    attrs = self.handle_region_attributes(key, grm)
                    return grm.write(rg.RegionBegin(inports=ins, attrs=attrs))
                case "Term", children:
                    match op, children:
                        case "GraphRoot", {"t": term}:
                            return term
                        case "Term.Func", {
                            "uid": uid,
                            "fname": fname,
                            "body": body,
                        }:
                            [uid, fname, body] = (
                                uid,
                                fname,
                                body,
                            )
                            orig_func: rg.Func = self.lookup_sexpr(int(uid))
                            return grm.write(
                                rg.Func(
                                    fname=fname,
                                    args=orig_func.args,
                                    body=body,
                                )
                            )
                        case "Term.RegionEnd", {
                            "region": region,
                            "ports": ports,
                        }:
                            return grm.write(
                                rg.RegionEnd(
                                    begin=region,
                                    ports=tuple(ports),
                                )
                            )
                        case "Term.IfElse", {
                            "cond": cond,
                            "then": then,
                            "orelse": orelse,
                            "operands": operands,
                        }:
                            [cond, then, orelse] = (
                                cond,
                                then,
                                orelse,
                            )
                            return grm.write(
                                rg.IfElse(
                                    cond=cond,
                                    body=then,
                                    orelse=orelse,
                                    operands=operands,
                                )
                            )
                        case "Term.IO", {}:
                            return grm.write(rg.IO())
                        case "Term.LiteralF64", {"val": float(val)}:
                            return grm.write(rg.PyFloat(val))
                        case "Term.LiteralI64", {"val": int(val)}:
                            return grm.write(rg.PyInt(val))
                        case "Term.Param", {"idx": idx}:
                            # TODO: get actual param name
                            return grm.write(rg.ArgRef(idx=idx, name=str(idx)))

                        case "·.get", {"self": term, "idx": idx}:
                            return grm.write(rg.Unpack(val=term, idx=idx))
                        case "·.getPort", {"self": term, "idx": idx}:
                            return grm.write(rg.Unpack(val=term, idx=idx))
                        case "PartialEvaluated", {"value": value}:
                            return value
                        case "Term.Undef", {"name": name}:
                            return grm.write(rg.Undef(name=name))
                        case "Term.LiteralBool", {"val": bool(val)}:
                            return grm.write(rg.PyBool(value=val))
                        case "Term.LiteralStr", {"val": str(val)}:
                            return grm.write(rg.PyStr(value=val))
                        case "Term.LiteralNone", {}:
                            return grm.write(rg.PyNone())
                        case "Term.Loop", {
                            "body": body_regionend,
                            "operands": operands,
                        }:
                            return grm.write(
                                rg.Loop(
                                    body=body_regionend,
                                    operands=operands,
                                )
                            )
                        case "Term.DbgValue", {
                            "varname": str(varname),
                            "value": value,
                        }:
                            # TODO: Loc
                            return grm.write(
                                rg.DbgValue(
                                    name=varname,
                                    value=value,
                                    srcloc=grm.write(rg.unknown_loc()),
                                    interloc=grm.write(rg.unknown_loc()),
                                )
                            )
                        case "·.dyn_get", {
                            "self": rg.RegionBegin() as regionbegin,
                            "idx": int(idx),
                        } if allow_dynamic_op:
                            return grm.write(
                                rg.Unpack(val=regionbegin, idx=idx)
                            )
                        case _:
                            extended_handle = self.handle_Term(
                                op, children, grm
                            )
                            if extended_handle is not NotImplemented:
                                return extended_handle
                            return self.handle_unknown(key, op, children, grm)

                case "TermList", {"terms": terms}:
                    return tuple(terms)
                case "PortList", {"ports": ports}:
                    return tuple(ports)

                case "Value", children:
                    return self.handle_Value(op, children, grm)

                case "InPorts", {"names": names}:
                    return tuple(names)

                case "Port", {"name": str(name), "term": value}:
                    return grm.write(rg.Port(name=name, value=value))

                case "DynInt", {
                    "self": termlist,
                    "target": target,
                } if op == "·.dyn_index" and allow_dynamic_op:
                    for i, term in enumerate(termlist):
                        if ase.matches(term, target):
                            return i
                    raise ValueError("cannot find target")

                case "DynInt", {"num": int(ival)} if op == "DynInt":
                    return ival
                case _:
                    handler = getattr(self, f"handle_{node_type}", None)
                    if handler is not None:
                        res = handler(key, op, children, grm)
                        if res is not NotImplemented:
                            return res

                    return self.handle_unknown(key, op, children, grm)
        else:
            raise NotImplementedError(key)

    def handle_primitive(
        self, node_type: Qualname, node, children: tuple, grm: Grammar
    ):
        matcher = Matcher()

        Eq = matcher.Eq
        egglog_ns = Eq("egglog.builtins")
        sealir_ns = Eq("sealir.eqsat.rvsdg_eqsat")

        @matcher.case(egglog_ns, Eq("String"))
        def _(modname: str, ty: str):
            unquoted = node["op"][1:-1]
            return unquoted

        @matcher.case(egglog_ns, Eq("bool"))
        def _(modname: str, ty: str):
            match node["op"]:
                case "true":
                    return True
                case "false":
                    return False
                case _:
                    raise AssertionError()

        @matcher.case(egglog_ns, Eq("i64"))
        def _(modname: str, ty: str):
            return int(node["op"])

        @matcher.case(egglog_ns, Eq("f64"))
        def _(modname: str, ty: str):
            return float(node["op"])

        @matcher.case(egglog_ns, Eq("Vec"))
        def _(modname: str, ty: str):
            param_match = Matcher()

            @param_match.case(sealir_ns, Eq("Term"))
            @param_match.case(sealir_ns, Eq("Port"))
            @param_match.case(egglog_ns, Eq("String"))
            def _(modname: str, ty: str):
                return tuple(children)

            @param_match.default
            def _(*args, **kwargs):
                return grm.write(
                    rg.GenericList(
                        name=node_type.get_fullname(), children=tuple(children)
                    )
                )

            return param_match(node_type.param.prefix, node_type.param.name)

        return matcher(node_type.prefix, node_type.name)

    def handle_Value(self, op: str, children: dict | list, grm: Grammar):
        matcher = Matcher()

        @matcher.case
        def _(val: int, op="Value.ConstI64"):
            return grm.write(rg.PyInt(val))

        @matcher.case
        def _(idx: int, op="Value.Param"):
            return grm.write(rg.ArgRef(idx=idx, name=str(idx)))

        return matcher(op=op, **children)

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        return self.handle_Py_Term(op, children, grm)

    def handle_Py_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Py_Add", {"a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOpPure(
                        op="+",
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_AddIO", {"io": io, "a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOp(
                        op="+",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_InplaceAddIO", {
                "io": io,
                "a": lhs,
                "b": rhs,
            }:
                return grm.write(
                    rg.PyInplaceBinOp(
                        op="+",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_SubIO", {"io": io, "a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOp(
                        op="-",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_Mul", {"a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOpPure(
                        op="*",
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_MulIO", {"io": io, "a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOp(
                        op="*",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_Div", {"a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOpPure(
                        op="/",
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_DivIO", {"io": io, "a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOp(
                        op="/",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_Pow", {"a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOpPure(
                        op="**",
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_PowIO", {"io": io, "a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOp(
                        op="**",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_LtIO", {"io": io, "a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOp(
                        op="<",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_GtIO", {"io": io, "a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOp(
                        op=">",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_NeIO", {"io": io, "a": lhs, "b": rhs}:
                return grm.write(
                    rg.PyBinOp(
                        op="!=",
                        io=io,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            case "Py_NotIO", {"io": io, "term": term}:
                return grm.write(rg.PyUnaryOp(op="not", io=io, operand=term))
            case "Py_AttrIO", {
                "io": io,
                "obj": obj,
                "attrname": str(attrname),
            }:
                return grm.write(
                    rg.PyAttr(io=io, value=obj, attrname=attrname)
                )
            case "Py_LoadGlobal", {"io": io, "name": str(name)}:
                return grm.write(rg.PyLoadGlobal(io=io, name=name))
            case "Py_Call", {
                "func": func,
                "io": io,
                "args": args,
            }:
                return grm.write(rg.PyCall(func=func, io=io, args=tuple(args)))
            case "Py_ForLoop", {
                "iter_arg_idx": int(iter_arg_idx),
                "indvar_arg_idx": int(indvar_arg_idx),
                "iterlast_arg_idx": int(iterlast_arg_idx),
                "body": body,
                "operands": operands,
            }:
                return grm.write(
                    rg.PyForLoop(
                        iter_arg_idx=iter_arg_idx,
                        indvar_arg_idx=indvar_arg_idx,
                        iterlast_arg_idx=iterlast_arg_idx,
                        body=body,
                        operands=operands,
                    )
                )
            case "Py_Tuple", {"elems": tuple(elems)}:
                return grm.write(
                    rg.PyTuple(
                        elems=elems,
                    )
                )
            case "Py_SliceIO", {
                "io": io,
                "lower": lower,
                "upper": upper,
                "step": step,
            }:
                return grm.write(
                    rg.PySlice(io=io, lower=lower, upper=upper, step=step)
                )
            case "Py_SubscriptIO", {"io": io, "obj": obj, "index": index}:
                return grm.write(rg.PySubscript(io=io, value=obj, index=index))
            case _:
                return NotImplemented

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
