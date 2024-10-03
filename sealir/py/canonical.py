from dataclasses import dataclass
from itertools import starmap
from types import SimpleNamespace
from typing import NamedTuple, cast

from sealir import ase, grammar, rvsdg, rvsdg_conns


class ForLoopInfo(NamedTuple):
    range_node: rvsdg_conns.ExprNode
    iter_node: rvsdg_conns.ExprNode
    next_node: rvsdg_conns.ExprNode
    endloop_compare: rvsdg_conns.ExprNode


@dataclass(frozen=True)
class PythonAnalysis:
    uda: rvsdg_conns.UseDefAnalysis

    def find_for_loops(self):
        yield from starmap(
            ForLoopInfo,
            self.uda.search_use_chain(
                self.match_range_call,
                self.match_iter_node,
                self.match_next_node,
                self.match_endloop_compare,
            ),
        )

    def callee_is_global_load(
        self, node: rvsdg_conns.GraphNode, name: str
    ) -> bool:
        callee_pos = grammar.field_position(rvsdg.Py_Call, "callee")
        callee_node = self.uda.node_inputs[node]
        callee_expr = cast(
            rvsdg_conns.ExprNode, callee_node.get_input_port(callee_pos)
        )
        match callee_expr.expr:
            case rvsdg.Py_GlobalLoad(name=k) if k == name:
                return True
        return False

    def match_range_call(self, prefixes, node: rvsdg_conns.GraphNode) -> bool:
        # get call to range()
        match node:
            case rvsdg_conns.ExprNode(expr=rvsdg.Py_Call(callee=callee)):
                pass
            case _:
                return False

        return self.callee_is_global_load(node, name="range")

    def match_iter_node(self, prefixes, x):
        [range_node] = prefixes
        if self.uda.is_sole_user(x, of=range_node, port=1):
            match x:
                case rvsdg_conns.ExprNode(expr=rvsdg.Py_Call()):
                    if self.callee_is_global_load(x, name="iter"):
                        return True

    def match_next_node(self, prefixes, x):
        [*_, iter_node] = prefixes
        if self.uda.is_sole_user(x, of=iter_node, port=1):
            match x:
                case rvsdg_conns.ExprNode(expr=rvsdg.Py_Call()):
                    if self.callee_is_global_load(x, name="next"):
                        return True

    def match_endloop_compare(self, prefixes, x):
        [*_, next_node] = prefixes
        for user in self.uda.get_users_of(of=next_node, port=1):
            if user != x:
                continue
            match user:
                case rvsdg_conns.ExprNode(expr=rvsdg.Py_Compare()):
                    pass
                case _:
                    continue
            lhs_idx = grammar.field_position(rvsdg.Py_Compare, "lhs")
            rhs_idx = grammar.field_position(rvsdg.Py_Compare, "rhs")
            lhs_node = self.uda.node_inputs[user].get_input_port(lhs_idx)
            rhs_node = self.uda.node_inputs[user].get_input_port(rhs_idx)

            # The following is needed because plain variable is for
            # capturing but `x.y` is a matched value
            tmp = SimpleNamespace()
            tmp.next_node = next_node
            pos = grammar.field_position(rvsdg.Py_Call, "args")
            sentinel = (
                self.uda.node_inputs[next_node].get_input_port(pos + 1).expr
            )

            def capture_info(node):
                match node:
                    case tmp.next_node:
                        return (0, node)
                    case rvsdg_conns.ExprNode(expr=rvsdg.Py_Str() as string):
                        return (1, string)
                return (-1, None)

            captured = sorted(map(capture_info, (lhs_node, rhs_node)))
            match captured:
                case ((0, _), (1, string)) if ase.matches(string, sentinel):
                    return True
                case _:
                    continue
