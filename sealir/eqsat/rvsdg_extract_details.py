from __future__ import annotations

from dataclasses import dataclass

from sealir import ase
from sealir.rvsdg import Grammar
from sealir.rvsdg import grammar as rg

from .egraph_utils import EGraphJsonDict


@dataclass(frozen=True)
class Data: ...


@dataclass(frozen=True)
class RegionBeginData(Data):
    begin: rg.RegionBegin
    ins: str
    ports: tuple


class EGraphToRVSDG:
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
                case "i64":
                    return int(node["op"])
                case "f64":
                    return float(node["op"])
                case "Vec_Term":
                    return get_children()
                case _:
                    raise NotImplementedError(f"primitive of: {node_type}")
        elif key.startswith("function-"):
            op = node["op"]
            children = get_children()

            rbd: RegionBeginData
            match node_type, children:
                case "Region", {"ins": ins, "ports": ports}:
                    return RegionBeginData(
                        begin=grm.write(
                            rg.RegionBegin(ins=ins, ports=tuple(ports))
                        ),
                        ins=ins,
                        ports=tuple(ports),
                    )
                case "InputPorts", {"self": RegionBeginData() as rbd}:
                    return rbd.begin
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
                            "outs": outs,
                            "ports": ports,
                        }:
                            assert len(outs.split()) == len(ports)
                            return grm.write(
                                rg.RegionEnd(
                                    begin=region.begin,
                                    outs=outs,
                                    ports=tuple(ports),
                                )
                            )
                        case "Term.Branch", {
                            "cond": cond,
                            "then": then,
                            "orelse": orelse,
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
                                    outs=then.outs,
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
                        case "Term.AddIO", {"io": io, "a": lhs, "b": rhs}:
                            return grm.write(
                                rg.PyBinOp(
                                    op="+",
                                    io=io,
                                    lhs=lhs,
                                    rhs=rhs,
                                )
                            )
                        case "Term.MulIO", {"io": io, "a": lhs, "b": rhs}:
                            return grm.write(
                                rg.PyBinOp(
                                    op="*",
                                    io=io,
                                    lhs=lhs,
                                    rhs=rhs,
                                )
                            )
                        case "Term.DivIO", {"io": io, "a": lhs, "b": rhs}:
                            return grm.write(
                                rg.PyBinOp(
                                    op="/",
                                    io=io,
                                    lhs=lhs,
                                    rhs=rhs,
                                )
                            )
                        case "Term.PowIO", {"io": io, "a": lhs, "b": rhs}:
                            return grm.write(
                                rg.PyBinOp(
                                    op="**",
                                    io=io,
                                    lhs=lhs,
                                    rhs=rhs,
                                )
                            )
                        case "Term.LtIO", {"io": io, "a": lhs, "b": rhs}:
                            return grm.write(
                                rg.PyBinOp(
                                    op="<",
                                    io=io,
                                    lhs=lhs,
                                    rhs=rhs,
                                )
                            )
                        case "Term.AttrIO", {
                            "io": io,
                            "obj": obj,
                            "attrname": str(attrname),
                        }:
                            return grm.write(
                                rg.PyAttr(io=io, value=obj, attrname=attrname)
                            )
                        case "Term.LoadGlobal", {"io": io, "name": str(name)}:
                            return grm.write(rg.PyLoadGlobal(io=io, name=name))
                        case "Term.Call", {
                            "func": func,
                            "io": io,
                            "args": args,
                        }:
                            return grm.write(
                                rg.PyCall(func=func, io=io, args=tuple(args))
                            )

                        case "·.get", {"self": term, "idx": idx}:
                            return grm.write(rg.Unpack(val=term, idx=idx))
                        case "·.getPort", {"self": term, "idx": idx}:
                            return grm.write(rg.Unpack(val=term, idx=idx))
                        case "PartialEvaluated", {"value": value}:
                            return value
                        case _:
                            raise NotImplementedError(
                                f"invalid Term op: {op!r}"
                            )
                case "TermList", {"terms": terms}:
                    return terms
                case "Value", children:
                    return self.handle_Value(op, children, grm)
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
