from __future__ import annotations

import ast
import functools
import inspect
import itertools
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Callable, MutableMapping, MutableSequence, Type, cast, TypeAlias

from numba_rvsdg.core.datastructures.ast_transforms import (
    AST2SCFG, SCFG2AST, AST2SCFGTransformer, unparse_code)
from numba_rvsdg.core.datastructures.basic_block import (PythonASTBlock,
                                                         RegionBlock,
                                                         SyntheticAssignment,
                                                         SyntheticExitBranch,
                                                         SyntheticExitingLatch,
                                                         SyntheticFill,
                                                         SyntheticHead,
                                                         SyntheticReturn,
                                                         SyntheticTail)
from numba_rvsdg.core.datastructures.scfg import SCFG


from sealir import ase
from sealir.lam import LamBuilder


_io_res:  TypeAlias = tuple[ase.Expr, ase.Expr]


class Scope:
    _defns: dict[Var, ase.Expr]
    _name_map: dict[str, Var]

    def __init__(self):
        self._defns = {}
        self._name_map = {}

    def define(self, var: Var, value: ase.Expr):
        self._defns[var] = value
        self._name_map[var.name] = var

    def get(self, name: str) -> ase.Expr:
        var = self._name_map[name]
        return self._defns[var]


@dataclass(frozen=True)
class Loc:
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None

    @classmethod
    def from_ast(cls, node: ast.AST):
        return Loc(
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset
        )


@dataclass(frozen=True, eq=False, order=False)
class Var:
    name: str
    loc: Loc | None = None

    @classmethod
    def from_ast_name(cls, name: ast.Name) -> Var:
        return Var(name.id, loc=Loc.from_ast(name))

class FindVar(ast.NodeVisitor):
    def __init__(self):
        self.loads = defaultdict(list)
        self.stores = defaultdict(list)

    def visit_Name(self, node: ast.Name):
        match node.ctx:
            case ast.Load():
                self.loads[node.id].append(Var.from_ast_name(node))
            case ast.Store():
                self.stores[node.id].append(Var.from_ast_name(node))
            case _:
                raise NotImplementedError(node)


def scan_variable_used(body: list[ast.stmt]):
    fv = FindVar()
    for node in body:
        fv.visit(node)
    allnames = set(fv.loads) | set(fv.stores)
    output = []
    for k in allnames:
        if k in fv.stores:
            output.append(fv.stores[k][0])
        else:
            output.append(fv.loads[k][0])
    return output

@dataclass()
class BuildState:
    lambar: LamBuilder
    defns: dict[Var, ase.Expr] = field(default_factory=dict)
    namemap: dict[str, Var] = field(default_factory=dict)

    _io_var = Var(".io")

    def define_io(self, expr: ase.Expr):
        self.define(self._io_var, expr)

    def define(self, var: Var, expr: ase.Expr):
        self.namemap[var.name] = var
        self.defns[var] = self.lambar.expr("store", var.name, expr)

    def get(self, name: str) -> ase.Expr:
        if name not in self.namemap:
            self.define(Var(name), self.lambar.expr("load_var", name))
        return self.defns[self.namemap[name]]

    @property
    def io(self) -> ase.Expr:
        return self.get(self._io_var.name)


class SCFG2SEXPR:


    def transform(
        self, scfg: SCFG, variables: list[Var],
    ):
        self.lambar = LamBuilder()
        self._variables = variables
        self.region_stack = [scfg.region]
        self.scfg = scfg
        for name, block in scfg.concealed_region_view.items():
            if type(block) is RegionBlock and block.kind == "branch":
                continue
            root = self.codegen(block)
        return self.lambar, root

    def lookup(self, item: Any) -> Any:
        subregion_scfg = self.region_stack[-1].subregion
        parent_region_block = self.region_stack[-1].parent_region
        if item in subregion_scfg:  # type: ignore
            return subregion_scfg[item]  # type: ignore
        else:
            return self.rlookup(parent_region_block, item)  # type: ignore

    def rlookup(self, region_block: RegionBlock, item: Any) -> Any:
        if item in region_block.subregion:  # type: ignore
            return region_block.subregion[item]  # type: ignore
        elif region_block.parent_region is not None:
            return self.rlookup(region_block.parent_region, item)
        else:
            raise KeyError(f"Item {item} not found in subregion or parent")

    def process_ast(self, body: list[ast.stmt]) -> ase.Expr:
        print("process", body)
        lb = self.lambar

        bs = self._get_build_state()

        expr = lb.expr("py_none")
        for stmt in body:
            expr = self._parse_stmt(stmt, bs)

        return lb.lam(expr)

    def _get_build_state(self) -> BuildState:
        lb = self.lambar

        bs = BuildState(lambar=lb)
        bs.define_io(lb.arg(0))
        for var in self._variables:
            bs.define(var, lb.expr("var", var.name))

        return bs

    def _parse_stmt(self, stmt: ast.stmt, bs: BuildState) -> ase.Expr:
        print('----', ast.dump(stmt))
        lb = self.lambar
        match stmt:
            case ast.Assign(targets=tars, value=val, type_comment=tc):
                _, rhs = self._parse_expr(val, bs, bs.io)
                variables = tuple(map(self._parse_Name, tars))
                for var in variables:
                    bs.define(var, rhs)
                return rhs

            case ast.AugAssign(target=target, op=op, value=val):
                io, rhs = self._parse_expr(val, bs, bs.io)
                var = self._parse_Name(target)
                lhs = bs.get(var.name)
                match op:
                    case ast.Add():
                        res = lb.expr("py_inplace_add", io, lhs, rhs)
                    case _:
                        raise NotImplementedError(op)
                io, res = lb.unpack(res, 2)
                bs.define_io(io)
                bs.define(var, res)
                return res
            case ast.Return(value=val):
                if val is None:
                    ret = lb.expr("py_none")
                else:
                    io, ret = self._parse_expr(val, bs, bs.io)
                    return lb.expr("py_return", io, ret)
                return ret
            case ast.Expr(value=value):
                io, ex_value = self._parse_expr(value, bs, bs.io)
                bs.define_io(io)
                return ex_value

        raise TypeError(stmt)

    def _parse_expr(self, expr: ast.expr, bs: BuildState, io: ase.Expr) -> _io_res:
        # Developer note: each case must not FALLTHROUGH
        lb = self.lambar
        match expr:
            case ast.Constant(value=value, kind=kind):
                match (value, kind):
                    case (int(), None):
                        const = lb.expr("const_int", value)
                        return io, const
                    case (None, None):
                        const = lb.expr("const_none")
                        return io, const
                    case (str(), None):
                        const = lb.expr("py_str", value)
                        return io, const
            case ast.Name(id, ctx=ast.Load()):
                return io, bs.get(id)
            case ast.BinOp(left=left, op=op, right=right):
                io, ex_lhs = self._parse_expr(left, bs, io)
                io, ex_rhs = self._parse_expr(right, bs, io)
                match op:
                    case ast.Add():
                        res = lb.expr("py_add", io, ex_lhs, ex_rhs)
                        io, res = lb.unpack(res, 2)
                        return io, res
                    case _:
                        raise NotImplementedError(op)
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                io, ex_left = self._parse_expr(left, bs, io)
                [op] = ops
                [comp] = comparators
                match op:
                    case ast.NotEq():
                        opname = "py_not_eq"
                    case _:
                        raise NotImplementedError(op)
                io, ex_comp = self._parse_expr(comp, bs, io)
                io, res = lb.unpack(lb.expr(opname, io, ex_left, ex_comp), 2)
                return io, res
            case ast.Call(func=func, args=args, keywords=keywords):
                io, ex_callee = self._parse_expr(func, bs, io)
                ex_args = []
                for x in args:
                    io, ex_arg = self._parse_expr(x, bs, io)
                    ex_args.append(ex_arg)
                assert not keywords   # TODO
                io, res = lb.unpack(lb.expr("py_call", io, ex_callee, *ex_args), 2)
                return io, res

        raise NotImplementedError(ast.dump(expr))

    def _parse_Name(self, expr: ast.expr) -> Var:
        match expr:
            case ast.Name(id=id, ctx=ctx):
                match ctx:
                    case ast.Store():
                        assert isinstance(id, str)
                        return Var.from_ast_name(expr)
                    # case ast.Load():
                    #     assert isinstance(id, str)
                    #     return Var.from_ast_name(expr)
                    case _:
                        raise NotImplementedError(ctx)
            case _:
                raise NotImplementedError(expr)

    def codegen(self, block: Any) -> ase.Expr:
        print('codegen', block)
        if type(block) is PythonASTBlock:
            if len(block.jump_targets) == 2:
                for node in block.tree:
                    print("node", '::::', ast.dump(node))
                ex_test = self.process_ast(block.tree)

                lb = self.lambar
                bs = BuildState(lb)
                bs.define_io(lb.expr("io")) # FIXME
                io = bs.io

                body = self.codegen(self.lookup(block.jump_targets[0]))
                assert not isinstance(body, list)
                orelse = self.codegen(self.lookup(block.jump_targets[1]))
                assert not isinstance(orelse, list)
                if_node = lb.expr("scfg.switch", io, ex_test, body, orelse)
                return if_node
            elif block.fallthrough and type(block.tree[-1]) is ast.Return:
                # The value of the ast.Return could be either None or an
                # ast.AST type. In the case of None, this refers to a plain
                # 'return', which is implicitly 'return None'. So, if it is
                # None, we assign the __scfg_return_value__ an
                # ast.Constant(None) and whatever the ast.AST node is
                # otherwise.
                return self.process_ast(block.tree)
            elif block.fallthrough or block.is_exiting:
                return self.process_ast(block.tree)
            else:
                raise NotImplementedError
        elif type(block) is RegionBlock:
            # We maintain a stack of the current region, in order to allow for
            # random node lookup by name.
            self.region_stack.append(block)

            # This is a custom view that uses the concealed_region_view and
            # additionally filters all branch regions. Essentially, branch
            # regions will be visited by calling codegen recursively from
            # blocks with multiple jump targets and all other regions must be
            # visited linearly.
            def codegen_view() -> list[ase.Expr]:
                return list(
                        self.codegen(b)
                        for b in block.subregion.concealed_region_view.values()  # type: ignore  # noqa
                        if not (type(b) is RegionBlock and b.kind == "branch")
                    )

            if block.kind in ("head", "tail", "branch"):
                rval = codegen_view()
            elif block.kind == "loop":
                # A loop region gives rise to a Python while __scfg_loop_cont__
                # loop. We recursively visit the body. The exiting latch will
                # update __scfg_loop_continue__.

                lb = self.lambar
                # bs = BuildState(lb)
                loopbody = codegen_view()
                rval = lb.expr("scfg.dowhile", loopbody, ) # FIXME

            else:
                raise NotImplementedError
            self.region_stack.pop()
            return rval
        elif type(block) is SyntheticAssignment:
            # Synthetic assignments just create Python assignments, one for
            # each variable..

            bs = self._get_build_state()
            for t, v in block.variable_assignment.items():
                bs.define()
            raise
            return [
                ast.Assign([ast.Name(t)], ast.Constant(v), lineno=0)
                for t, v in block.variable_assignment.items()
            ]
        elif type(block) is SyntheticTail:
            # Synthetic tails do nothing.
            raise
            return []
        elif type(block) is SyntheticFill:
            # Synthetic fills must have a pass statement to main syntactical
            # correctness of the final program.
            raise
            return [ast.Pass()]
        elif type(block) is SyntheticReturn:
            # Synthetic return blocks must re-assigne the return value to a
            # special reserved variable.
            raise
            return [ast.Return(ast.Name("__scfg_return_value__"))]
        elif type(block) is SyntheticExitingLatch:
            # The synthetic exiting latch simply assigns the negated value of
            # the exit variable to '__scfg_loop_cont__'.
            assert len(block.jump_targets) == 1
            assert len(block.backedges) == 1
            raise
            return [
                ast.Assign(
                    [ast.Name("__scfg_loop_cont__")],
                    ast.UnaryOp(ast.Not(), ast.Name(block.variable)),
                    lineno=0,
                )
            ]
        elif type(block) in (SyntheticExitBranch, SyntheticHead):
            # Both the Synthetic exit branch and the synthetic head contain a
            # branching statement with potentially multiple outgoing branches.
            # This means we must recursively generate an if-cascade in Python,
            # such that all jump targets may be visisted. Looking at the
            # resulting AST, it does appear as though the compilation of the
            # AST to source code will use `elif` statements.

            # Create a reverse lookup from the branch_value_table
            # branch_name --> list of variables that lead there
            reverse = defaultdict(list)
            for (
                variable_value,
                jump_target,
            ) in block.branch_value_table.items():
                reverse[jump_target].append(variable_value)
            # recursive generation of if-cascade

            raise
            def if_cascade(
                jump_targets: list[str],
            ) -> MutableSequence[ast.AST]:
                if len(jump_targets) == 1:
                    # base case, final else
                    return self.codegen(self.lookup(jump_targets.pop()))
                else:
                    # otherwise generate if statement for current jump_target
                    current = jump_targets.pop()
                    # compare to all variable values that point to this
                    # jump_target
                    if_test = ast.Compare(
                        left=ast.Name(block.variable),
                        ops=[ast.In()],
                        comparators=[
                            ast.Tuple(
                                elts=[
                                    ast.Constant(i) for i in reverse[current]
                                ],
                                ctx=ast.Load(),
                            )
                        ],
                    )
                    # Create the the if-statement itself, using the test. Do
                    # code-gen for the block that the is being pointed to and
                    # recurse for the rest of the jump_targets.
                    if_node = ast.If(
                        test=if_test,
                        body=cast(
                            list[ast.stmt], self.codegen(self.lookup(current))
                        ),
                        orelse=cast(list[ast.stmt], if_cascade(jump_targets)),
                    )
                    return [if_node]

            # Send in a copy of the jump_targets as this list will be mutated.
            return if_cascade(list(block.jump_targets[::-1]))
        else:
            raise NotImplementedError

        raise NotImplementedError("unreachable")


def restructure_source(function):
    ast2scfg_transformer = AST2SCFGTransformer(function)
    astcfg = ast2scfg_transformer.transform_to_ASTCFG()
    scfg = astcfg.to_SCFG()
    scfg.restructure()
    scfg2ast = SCFG2SEXPR()

    # Find all variables
    original_ast = unparse_code(function)[0]
    variables = scan_variable_used([original_ast])
    print(variables)

    lambar, root = scfg2ast.transform(scfg=scfg, variables=variables)
    print(lambar.get_tree().dump())

    print(lambar.format(root))
    print('---')
    root = lambar.run_abstraction_pass(root)
    print(lambar.format(root))
    print(root.str())

###############################################################################
def sum1d(n: int) -> int:
    c = 0
    for i in range(n):
        c += i
    return c

# def sum1d(n: int) -> int:
#     c = 0
#     for i in range(n):
#         for j in range(i):
#             c += i * j
#             if c > 100:
#                 break
#     return c

# def sum1d(n: int) -> int:
#     c = 0
#     for i in range(n):
#         for j in range(i):
#             c += i + j
#     return c

# def sum1d(n: int) -> int:
#     c = 0
#     c += c
#     c = c + 1
#     return c


def main(out_filename: str):
    source = restructure_source(sum1d)

    print(source)


def test():
    main(None)

if __name__ == "__main__":
    # main(out_filename=sys.argv[1])
    main(None)

