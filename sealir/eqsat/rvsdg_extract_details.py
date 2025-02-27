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

    def run(self, node_iterator):
        memo = self.memo
        with self.rvsdg_sexpr._tape as tape:
            grm = Grammar(tape)
            for key in node_iterator:
                last = memo[key] = self.handle(key, grm)
        return last

    def lookup_sexpr(self, uid: int) -> ase.SExpr:
        tape: ase.Tape = self.rvsdg_sexpr._tape
        return Grammar.downcast(tape.read_value(uid))

    def handle(self, key: str, grm: Grammar):
        nodes = self.gdct["nodes"]
        memo = self.memo

        node = nodes[key]
        eclass = node["eclass"]
        node_type = self.gdct["class_data"][eclass]["type"]

        def get_children():
            return list(map(memo.__getitem__, node["children"]))

        if key.startswith("primitive-"):
            match node_type:
                case "String":
                    unquoted = node["op"][1:-1]
                    return grm.write(rg.PyStr(unquoted))
                case "i64":
                    return grm.write(rg.PyInt(int(node["op"])))
                case "Vec_Term":
                    return get_children()
                case _:
                    raise NotImplementedError(f"primitive of: {node_type}")
        elif key.startswith("function-"):
            op = node["op"]
            children = get_children()

            rbd: RegionBeginData
            match node_type:
                case "Region":
                    uid, ins, ports = children
                    return RegionBeginData(
                        begin=grm.write(
                            rg.RegionBegin(ins=ins.value, ports=tuple(ports))
                        ),
                        ins=ins,
                        ports=tuple(ports),
                    )
                case "InputPorts":
                    [rbd] = children
                    return rbd.begin
                case "Term":
                    match op:
                        case "GraphRoot":
                            [term] = children
                            return term
                        case "Term.Func":
                            [uid, fname, body] = children
                            orig_func: rg.Func = self.lookup_sexpr(
                                int(uid.value)
                            )
                            return grm.write(
                                rg.Func(
                                    fname=fname.value,
                                    args=orig_func.args,
                                    body=body,
                                )
                            )
                        case "Term.RegionEnd":
                            [rbg, outs, ports] = children
                            return grm.write(
                                rg.RegionEnd(
                                    begin=rbg.begin,
                                    outs=outs.value,
                                    ports=tuple(ports),
                                )
                            )
                        case "Term.Branch":
                            [cond, then, orelse] = children
                            return grm.write(
                                rg.IfElse(
                                    cond=cond,
                                    body=then,
                                    orelse=orelse,
                                    outs=then.outs,
                                )
                            )
                        case "Term.IO":
                            return grm.write(rg.IO())
                        case "Term.Param":
                            [idx] = children
                            return grm.write(
                                rg.ArgRef(idx=idx.value, name=str(idx.value))
                            )
                        case "Term.LtIO":
                            [io, lhs, rhs] = children
                            return grm.write(
                                rg.PyBinOp(op="<", io=io, lhs=lhs, rhs=rhs)
                            )
                        case "·.get":
                            [term, idx] = children
                            return grm.write(
                                rg.Unpack(val=term, idx=idx.value)
                            )
                        case "·.getPort":
                            [term, idx] = children
                            return grm.write(
                                rg.Unpack(val=term, idx=idx.value)
                            )
                        case _:
                            raise NotImplementedError(
                                f"invalid Term op: {op!r}"
                            )
                case "TermList":
                    [terms] = children
                    return terms
                case _:
                    raise NotImplementedError(
                        f"function of: {op!r} :: {node_type}"
                    )
        else:
            raise NotImplementedError(key)
