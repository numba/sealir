from __future__ import annotations

from dataclasses import dataclass

from sealir import ase
from sealir.rvsdg import Grammar
from sealir.rvsdg import grammar as rg

from .egraph_utils import EGraphJsonDict


@dataclass(frozen=True)
class Data: ...


class EGraphToRVSDG:
    allow_dynamic_op = False

    def __init__(self, gdct: EGraphJsonDict, rvsdg_sexpr: ase.SExpr):
        self.rvsdg_sexpr = rvsdg_sexpr
        self.gdct = gdct
        self.memo = {}

    def run(self, node_and_children):
        memo = self.memo
        with self.rvsdg_sexpr._tape as tape:
            grm = Grammar(tape)
            for key, child_keys in node_and_children:
                try:
                    last = memo[key] = self.handle(key, child_keys, grm)
                except Exception as e:
                    e.add_note(f"Extracting: {key}, {child_keys}")
                    raise

        return last

    def lookup_sexpr(self, uid: int) -> ase.SExpr:
        tape: ase.Tape = self.rvsdg_sexpr._tape
        return Grammar.downcast(tape.read_value(uid))

    def handle(
        self, key: str, child_keys: list[str] | dict[str, str], grm: Grammar
    ):
        allow_dynamic_op = self.allow_dynamic_op

        nodes = self.gdct["nodes"]
        memo = self.memo

        node = nodes[key]
        eclass = node["eclass"]
        node_type = self.gdct["class_data"][eclass]["type"]

        def get_children() -> dict | list:
            if isinstance(child_keys, dict):
                return {k: memo[v] for k, v in child_keys.items()}
            else:
                return [memo[v] for v in child_keys]

        if key.startswith("primitive-"):
            match node_type:
                case "String":
                    unquoted = node["op"][1:-1]
                    return unquoted
                case "bool":
                    match node["op"]:
                        case "true":
                            return True
                        case "false":
                            return False
                        case _:
                            raise AssertionError()
                case "i64":
                    return int(node["op"])
                case "f64":
                    return float(node["op"])
                case "Vec_Term":
                    return get_children()
                case "Vec_Port":
                    return get_children()
                case "Vec_String":
                    return get_children()
                case _:
                    raise NotImplementedError(f"primitive of: {node_type}")
        elif key.startswith("function-"):
            op = node["op"]
            children = get_children()

            match node_type, children:
                case "Region", {"inports": ins}:
                    return grm.write(rg.RegionBegin(inports=ins))
                case "Term", children:
                    extended_handle = self.handle_Term(op, children, grm)
                    if extended_handle is not NotImplemented:
                        return extended_handle
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
                            raise NotImplementedError(
                                f"invalid Term: {node_type}, {children}"
                            )
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
                    raise NotImplementedError(
                        f"function of: {op!r} :: {node_type}"
                    )
        else:
            raise NotImplementedError(key)

    def handle_Value(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Value.ConstI64", {"val": val}:
                return grm.write(rg.PyInt(val))
            case "Value.Param", {"i": int(idx)}:
                return grm.write(rg.ArgRef(idx=idx, name=str(idx)))
            case _:
                raise NotImplementedError(f"Value of {op!r}")

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
            case _:
                return NotImplemented
