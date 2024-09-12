from __future__ import annotations
import sys
import ast
import inspect
from contextlib import contextmanager
from typing import Type, TypeAlias, Any, Iterator, Sequence, Iterable, NamedTuple
from pprint import pprint, pformat
from dataclasses import dataclass, field
from collections import defaultdict, deque

from numba_rvsdg.core.datastructures.ast_transforms import (
    unparse_code,
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
)

from sealir import ase
from sealir.lam import LamBuilder
from sealir.rewriter import TreeRewriter
from sealir.itertools import first, maybe_first

SExpr: TypeAlias = ase.Expr


def pp(expr: SExpr):
    print(pformat(expr.as_tuple(-1, dedup=True)).replace(',', '').replace('\'', ''))
    # print(pformat(expr.as_dict()))


class ConvertToSExpr(ast.NodeTransformer):
    _tape = ase.Tape

    def __init__(self, tape):
        super().__init__()
        self._tape = tape

    def generic_visit(self, node: ast.AST):
        raise NotImplementedError(ast.dump(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> SExpr:
        # attrs = ("name", "args", "body", "decorator_list", "returns", "type_comment", "type_params")
        fname = node.name
        args = self.visit(node.args)
        body = [self.visit(stmt) for stmt in node.body]
        # TODO: "decorator_list" "returns", "type_comment", "type_params",
        # decorator_list = [self.visit(x) for x in node.decorator_list]
        assert not node.decorator_list
        return ase.expr(
            "PyAst_FunctionDef",
            fname,
            args,
            ase.expr("PyAst_block", *body),
            self.get_loc(node),
        )

    def visit_Return(self, node: ast.Return) -> SExpr:
        return ase.expr(
            "PyAst_Return",
            self.visit(node.value),
            self.get_loc(node),
        )

    def visit_arguments(self, node: ast.arguments) -> SExpr:
        # TODO
        assert not node.posonlyargs
        assert not node.kwonlyargs
        assert not node.kw_defaults
        assert not node.defaults
        return ase.expr(
            "PyAst_arguments",
            *map(self.visit, node.args),
        )

    def visit_arg(self, node: ast.arg) -> SExpr:
        return ase.expr(
            "PyAst_arg",
            node.arg,
            self.visit(node.annotation),
            self.get_loc(node),
        )

    def visit_Name(self, node: ast.Name) -> SExpr:
        print(ast.dump(node))
        match node.ctx:
            case ast.Load():
                ctx = 'load'
            case ast.Store():
                ctx = 'store'
            case _:
                raise NotImplementedError(node.ctx)
        return ase.expr("PyAst_Name", node.id, ctx, self.get_loc(node))

    def visit_Assign(self, node: ast.Assign) -> SExpr:
        return ase.expr(
            "PyAst_Assign",
            self.visit(node.value),
            *map(self.visit, node.targets),
            self.get_loc(node),
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> SExpr:
        return ase.expr(
            "PyAst_AugAssign",
            self.map_op(node.op),
            self.visit(node.target),
            self.visit(node.value),
            self.get_loc(node),
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> SExpr:
        return ase.expr(
            "PyAst_UnaryOp",
            self.map_op(node.op),
            self.visit(node.operand),
            self.get_loc(node),
        )

    def visit_BinOp(self, node: ast.BinOp) -> SExpr:
        return ase.expr(
            "PyAst_BinOp",
            self.map_op(node.op),
            self.visit(node.left),
            self.visit(node.right),
            self.get_loc(node),
        )

    def visit_Compare(self, node: ast.Compare) -> SExpr:
        [(op, comp)] = zip(node.ops, node.comparators, strict=True)  # TODO
        match op:
            case ast.NotEq():
                opname = "!="
            case ast.Gt():
                opname = ">"
            case ast.In():
                opname = "in"
            case _:
                raise NotImplementedError(op)
        return ase.expr(
            "PyAst_Compare",
            opname,
            self.visit(node.left),
            self.visit(comp),
            self.get_loc(node),
        )

    def visit_Call(self, node: ast.Call) -> SExpr:
        # TODO
        assert not node.keywords
        posargs = ase.expr("PyAst_callargs_pos", *map(self.visit, node.args))
        return ase.expr(
            "PyAst_Call",
            self.visit(node.func),
            posargs,
            self.get_loc(node),
        )

    def visit_While(self, node: ast.While) -> SExpr:
        # ("test", "body", "orelse")
        test = self.visit(node.test)
        body = ase.expr(
            "PyAst_block",
            *map(self.visit, node.body),
        )
        assert not node.orelse
        return ase.expr(
            "PyAst_While",
            test,
            body,
            self.get_loc(node),
        )

    def visit_If(self, node: ast.If) -> SExpr:
        test = self.visit(node.test)
        body = ase.expr(
            "PyAst_block",
            *map(self.visit, node.body),
        )
        orelse = ase.expr(
            "PyAst_block",
            *map(self.visit, node.orelse),
        )
        return ase.expr(
            "PyAst_If",
            test,
            body,
            orelse,
            self.get_loc(node),
        )

    def visit_Constant(self, node: ast.Constant) -> SExpr:
        if node.kind is None:
            match node.value:
                case int():
                    return ase.expr(
                        "PyAst_Constant_int",
                        node.value,
                        self.get_loc(node),
                    )
                case None:
                    return ase.expr(
                        "PyAst_None",
                        self.get_loc(node),
                    )
                case str():
                    return ase.expr(
                        "PyAst_Constant_str",
                        node.value,
                        self.get_loc(node),
                    )
        raise NotImplementedError(ast.dump(node))

    def visit_Tuple(self, node: ast.Tuple) -> SExpr:
        match node:
            case ast.Tuple(elts=[*constants], ctx=ast.Load()):
                # e.g. Tuple(elts=[Constant(value=0)], ctx=Load())
                return ase.expr(
                    "PyAst_Tuple",
                    *map(self.visit, constants),
                    self.get_loc(node),
                )

        raise NotImplementedError(ast.dump(node))

    def map_op(self, node: ast.operator) -> str:
        match node:
            case ast.Add():
                return "+"
            case ast.Sub():
                return "-"
            case ast.Mult():
                return "*"
            case ast.Not():
                return "not"
            case _:
                raise NotImplementedError(node)

    def get_loc(self, node: ast.AST) -> SExpr:
        return ase.expr(
            "PyAst_loc",
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
            None
        )


def convert_to_sexpr(node: ast.AST):
    with ase.Tape() as stree:
        out = ConvertToSExpr(stree).visit(node)
    pprint(out.as_tuple(depth=2**30))
    return out

@dataclass(frozen=True)
class VariableInfo:
    nonlocals: set[str]
    args: tuple[str]
    loaded: set[str]
    stored: set[str]
    # expr_effect: dict
    # block_effect: dict
    # assign_map: dict

def find_variable_info(expr: SExpr):
    loaded: dict[str, SExpr] = {}
    stored: dict[str, SExpr] = {}

    def get_args(expr: SExpr) -> Iterator[str]:
        _, first_def = next(expr.search_descendants(lambda x: x.head=="PyAst_FunctionDef"))
        assert first_def == expr
        (fname, args, block, loc) = expr.args
        for arg in args.args:
            match arg:
                case ase.Expr("PyAst_arg", (name, *_)):
                    yield name
                case _:
                    raise AssertionError(arg)

    arguments = tuple(get_args(expr))

    class FindVariableUse(ase.TreeVisitor):
        def visit(self, expr: SExpr):
            match expr:
                case ase.Expr("PyAst_Name", (name, "load", loc)):
                    loaded[name] = expr
                case ase.Expr("PyAst_Name", (name, "store", loc)):
                    stored[name] = expr
                case ase.Expr("PyAst_arg", (name, anno, loc)):
                    stored[name] = expr
                case _ if expr.head in {"PyAst_Name", "PyAst_arg"}:
                    raise AssertionError(expr)

    expr.apply_bottomup(FindVariableUse())


    nonlocals = set()
    for k in loaded:
        if k not in stored:
            nonlocals.add(k)


    @dataclass(frozen=True)
    class VarEffect:
        defs: set[str] = field(default_factory=set)
        uses: set[str] = field(default_factory=set)

        def define(self, name) -> None:
            self.defs.add(name)

        def use(self, name) -> None:
            self.uses.add(name)

        def merge(self, other: VarEffect) -> VarEffect:
            self.defs.update(other.defs)
            self.uses.update(other.uses)
            return self

    # expr_effect: dict[SExpr, VarEffect] = defaultdict(VarEffect)

    # # block_effect tracks the variables used by each statement in a block.
    # # It shows which variable-uses are required for the remaining statements
    # # in the block to execute. Variable-defs are local to that statement.
    # block_effect: dict[SExpr, deque[VarEffect]] = defaultdict(deque)

    # class FindDataDependence(ase.TreeVisitor):
    #     def visit(self, expr: SExpr):
    #         match expr:
    #             case ase.Expr("PyAst_block", body):
    #                 blkeff = block_effect[expr]
    #                 iter_stmt = iter(reversed(body))
    #                 last = next(iter_stmt)
    #                 blkeff.appendleft(VarEffect().merge(expr_effect[last]))
    #                 for stmt in iter_stmt:
    #                     lasteff = blkeff[0]
    #                     # Start with the expression level effect.
    #                     eff = VarEffect().merge(expr_effect[stmt])
    #                     blkeff.appendleft(eff)
    #                     # Keep track of future uses discounting newly defined
    #                     eff.uses.update(lasteff.uses - expr_effect[stmt].defs)
    #                     last = stmt
    #                 expr_effect[expr] = blkeff[0]

    #             case ase.Expr("PyAst_Name", (name, "load", loc)):
    #                 if name not in nonlocals:
    #                     expr_effect[expr].use(name)
    #             case ase.Expr("PyAst_Name", (name, "store", loc)):
    #                 expr_effect[expr].define(name)
    #             case ase.Expr("PyAst_arg", (name, anno, loc)):
    #                 expr_effect[expr].define(name)
    #             case _ if expr.head in {"PyAst_Name", "PyAst_arg", "PyAst_block"}:
    #                 raise AssertionError(expr)
    #             case _:
    #                 for child in expr.args:
    #                     if isinstance(child, ase.Expr):
    #                         expr_effect[expr].merge(expr_effect[child])

    # expr.apply_bottomup(FindDataDependence())

    # assign_map = defaultdict(dict)

    # def track_assignments(root: SExpr):
    #     """
    #     Deal with assignments. Including the implicit ones in code-blocks.
    #     Topdown pass.
    #     """
    #     block_depth_map = defaultdict(int)

    #     varmap = {}
    #     for i, k in enumerate(arguments):
    #         varmap[k] = "arg", i, k

    #     for k in nonlocals:
    #         varmap[k] = "nonlocal", k

    #     for parents, expr in root.walk_descendants():
    #         if parents:
    #             block_depth_map[expr] = block_depth_map[parents[-1]]
    #         else: # function-def
    #             block_depth_map[expr] = len(arguments)
    #         match expr:
    #             case ase.Expr("PyAst_Assign", (val, *targets, loc)):
    #                 match val:
    #                     case ase.Expr("PyAst_Name", (name, *_)):
    #                         val = varmap[name]

    #                 assigns = assign_map[expr]
    #                 for target in targets:
    #                     [x] = expr_effect[target].defs
    #                     assigns[x] = val
    #                     varmap[x] = val
    #             case ase.Expr("PyAst_Name", (name, "load", loc)):
    #                 assigns = assign_map[expr]
    #                 assigns[name] = varmap[name]
    #             case ase.Expr("PyAst_block"):
    #                 assigns = assign_map[expr]
    #                 blkeff = block_effect[expr]
    #                 # handle importing variables on the block
    #                 first = blkeff[0]
    #                 n_imports = len(first.uses)
    #                 block_depth_map[expr] += n_imports
    #                 # handle exporting variables on the block
    #                 last = blkeff[-1]
    #                 exporting = last.defs
    #                 for i, target in enumerate(sorted(exporting)):
    #                     assigns[target] = "unpack", expr, i, target
    #                     varmap[target] = "unpack", expr, i, target
    #             case ase.Expr(name) if name in {"PyAst_Assign", "PyAst_block"}:
    #                 raise AssertionError(expr)

    # track_assignments(expr)

    return VariableInfo(
        nonlocals=set(nonlocals),
        args=arguments,
        loaded=set(loaded),
        stored=set(stored),
        # expr_effect=expr_effect,
        # block_effect=block_effect,
        # assign_map=assign_map,
    )

def convert_to_rvsdg(prgm: SExpr, varinfo: VariableInfo):

    def get_block(expr: SExpr,) -> SExpr:
        return next(expr.search_parents(lambda x: x.head == "PyAst_block"))
    def uses_io(val):
        for child in val.args:
            match child:
                case ase.Expr("var_load", (".io",)):
                    return True
        return False
    def unpack(val: SExpr, targets: Sequence[str], *, force_io=False) -> Iterable[tuple[SExpr, str]]:
        if force_io or uses_io(val):
            targets  = (".io", *targets)
        if len(targets) == 1:
            yield (val, targets[0])
        else:
            packed_name = f".packed.{val.handle}"
            yield val, packed_name
            for i, target in enumerate(targets):
                packed = ase.expr("var_load", packed_name)
                yield ase.expr("unpack", i, packed), target

    def lookup(blk: ase.Expr):
        defs = set()
        class FindLet(ase.TreeVisitor):
            def visit(self, expr: SExpr):
                match expr:
                    case ase.Expr("let", (varname, *_)):
                        defs.add(varname)

        blk.apply_topdown(FindLet())
        return defs

    class Convert2RVSDG(TreeRewriter[SExpr]):
        """Convert to RVSDG with Let-binding
        """
        def rewrite_generic(self, orig: SExpr, args: tuple[Any, ...], updated: bool) -> SExpr:
            raise NotImplementedError(orig.head)

        def rewrite_PyAst_Name(self, orig: ase.Expr, name: str, ctx: str, loc: SExpr) -> SExpr:
            if name in varinfo.nonlocals:
                return ase.expr("py_global_load", name)
            else:
                match ctx:
                    case "store":
                        return ase.expr("var_store", name)
                    case "load":
                        return ase.expr("var_load", name)
                    case _:
                        raise AssertionError(ctx)

        def rewrite_PyAst_loc(self, orig: ase.Expr, *args) -> SExpr:
            return orig

        def rewrite_PyAst_arg(self, orig: ase.Expr, name: str, annotation: SExpr, loc: SExpr) -> SExpr:
            return name

        def rewrite_PyAst_arguments(self, orig: ase.Expr, *args: SExpr) -> SExpr:
            return ase.expr("py_args", *args)

        def rewrite_PyAst_Constant_int(self, orig: ase.Expr, val: int, loc: SExpr) -> SExpr:
            return ase.expr("py_int", val)

        def rewrite_PyAst_Constant_str(self, orig: ase.Expr, val: int, loc: SExpr) -> SExpr:
            return ase.expr("py_str", val)

        def rewrite_PyAst_None(self, orig: ase.Expr, loc: SExpr) -> SExpr:
            return ase.expr("py_none")

        def rewrite_PyAst_Tuple(self, orig: ase.Expr, *rest: SExpr) -> SExpr:
            [*elts, loc] = rest
            return ase.expr("py_tuple", *elts)

        def rewrite_PyAst_Assign(self, orig: ase.Expr, val: SExpr, *rest: SExpr) -> SExpr:
            [*targets, loc] = rest
            return ase.expr("assign", val, *(t.args[0] for t in targets))

        def rewrite_PyAst_AugAssign(self, orig: ase.Expr, opname: str, left: SExpr, right: SExpr, loc: SExpr) -> SExpr:
            [name] = left.args
            return ase.expr("assign", ase.expr("py_inplace_binop", opname, self.get_io(), ase.expr("var_load", name), right), name)

        def rewrite_PyAst_callargs_pos(self, orig: ase.Expr, *args: SExpr) -> SExpr:
            return args

        def rewrite_PyAst_Call(self, orig: ase.Expr, func: SExpr, posargs: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("py_call", self.get_io(), func, *posargs)

        def rewrite_PyAst_Compare(self, orig: ase.Expr, opname: str, left: SExpr, right: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("py_compare", opname, self.get_io(), left, right)

        def rewrite_PyAst_block(self, orig: ase.Expr, *body: SExpr) -> SExpr:
            last = ase.expr("scfg_pass")
            for stmt in reversed(body):
                match stmt:
                    case ase.Expr("assign", (val, *targets)):
                        iterable = unpack(val, targets)
                        for unpacked, target in reversed(list(iterable)):
                            last = ase.expr("let", target, unpacked, last)

                    case ase.Expr("py_if", (test, blk_true, blk_false)):
                        # look up variable in each block to find variable
                        # definitions
                        defs_true = lookup(blk_true)
                        defs_false = lookup(blk_false)
                        defs = sorted(defs_true | defs_false)

                        if uses_io(test):
                            [(unpack_cond, unpack_name), (io, iovar), (cond, _)] = unpack(test, [".cond"], force_io=True)
                            stmt = ase.expr("scfg_if", cond, blk_true, blk_false)
                            stmt = ase.expr("let", iovar, io, stmt)
                            stmt = ase.expr("let", unpack_name, unpack_cond, stmt)

                        iterable = unpack(stmt, defs)
                        for unpacked, target in reversed(list(iterable)):
                            last = ase.expr("let", target, unpacked, last)
                    case ase.Expr("scfg_while", (ase.Expr() as test, loopblk)):
                        defs = sorted(lookup(loopblk))

                        iterable = unpack(stmt, defs)
                        for unpacked, target in reversed(list(iterable)):
                            last = ase.expr("let", target, unpacked, last)

                    case ase.Expr("py_return", (retval,)):
                        output = list(unpack(retval, [".retval"]))
                        match output:
                            case [(unpack_cond, unpack_name), (io, iovar), (value, valuevar)]:
                                stmt = ase.expr("py_return", ase.expr("var_load", iovar), ase.expr("var_load", valuevar))
                                stmt = ase.expr("let", valuevar, value, stmt)
                                stmt = ase.expr("let", iovar, io, stmt)
                                stmt = ase.expr("let", unpack_name, unpack_cond, stmt)
                                last = stmt
                            case [(value, valuevar)]:
                                stmt = ase.expr("py_return", self.get_io(), value)
                                # stmt = ase.expr("let", valuevar, value, stmt)
                                last = stmt
                            case _:
                                assert False
                    case ase.Expr("py_pass"):
                        last = stmt
                    case _:
                        raise AssertionError(stmt)
            return last

        def rewrite_PyAst_If(self, orig: ase.Expr, test: SExpr, body: SExpr, orelse: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("py_if", test, body, orelse)

        def rewrite_PyAst_While(self, orig: ase.Expr, *rest: SExpr) -> SExpr:
            [test, *body, loc] = rest
            match test:
                case ase.Expr("var_load", (str(testname),)):
                    pass
                case _:
                    raise AssertionError(f"unsupported while loop: {test.str()}")
            return ase.expr("scfg_while", test, *body)

        def rewrite_PyAst_UnaryOp(self, orig: ase.Expr, opname: str, val: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("py_unaryop", self.get_io(), opname, val)

        def rewrite_PyAst_BinOp(self, orig: ase.Expr, opname: str, lhs: SExpr, rhs: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("py_binop", opname, self.get_io(), lhs, rhs)

        def rewrite_PyAst_Return(self, orig: ase.Expr, retval: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("py_return", retval)

        def rewrite_PyAst_FunctionDef(self, orig: ase.Expr, fname: str, args: SExpr, body: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("py_func", fname, args, body)

        def get_io(self) -> SExpr:
            return ase.expr("var_load", ".io")


    rewriter = Convert2RVSDG()
    with prgm.tape:
        prgm.apply_bottomup(rewriter)

    rewritten = rewriter.memo[prgm]

    pp(rewritten)
    # Verify
    def verify(root: SExpr):
        seen = set()
        for parents, cur in rewritten.walk_descendants():
            if cur not in seen:
                seen.add(cur)
            # else:
            #     continue
            match cur:
                case ase.Expr("scfg_while", (ase.Expr("var_load", (str(testname),)), *_)):
                    defn: ase.Expr | None = None
                    for p in reversed(parents):
                        match p:
                            case ase.Expr("let", (str() as x, ase.Expr() as defn, *_)) if x == testname:
                                break

                    match defn:
                        case ase.Expr("py_int", (1,)):
                            break
                        case _:
                            raise AssertionError(f"unsupported loop: {cur}\ndefn {defn}")
                case ase.Expr("scfg_while"):
                    raise AssertionError(f"malformed scfg_while: {cur}")


    verify(rewritten)

    return rewritten



def convert_to_lambda(prgm: SExpr, varinfo: VariableInfo):
    lb = LamBuilder(prgm.tape)

    match prgm:
        case ase.Expr("py_func", (str(fname), ase.Expr("py_args", args), *_)):
            parameters = (".io", *args)
        case _:
            raise AssertionError(prgm.as_tuple(1))

    arity = len(parameters)


    let_uses: dict[SExpr, int] = defaultdict(int)
    parent_let_map: dict[SExpr, SExpr] = {}
    var_depth_map: dict[SExpr, int] = {}
    def find_parent_let(parents, var_load: ase.Expr, *, excludes=()):
        [varname] = var_load.args

        depth = 0
        for p in reversed(parents):
            match p:
                case ase.Expr("let", (str(name), _, body)):
                    if body.contains(var_load):
                        if name == varname:
                            return p, depth
                        if p not in excludes:
                            depth += 1
        return None, depth

    reachable = set()
    for parents, child in prgm.walk_descendants_depth_first_no_repeat():
        reachable.add(child)
        match child:
            case ase.Expr("var_load"):
                parent_let, depth = find_parent_let(parents, child)
                var_depth_map[child] = depth
                if parent_let is not None:
                    let_uses[parent_let] += 1
                    parent_let_map[child] = parent_let
                else:
                    # function arguments do not have parent let.
                    pass

    # single_use_repl = {}
    # single_use_lets = set()
    # def simplify_single_use_let(expr: SExpr, state: ase.TraverseState):
    #     match expr:
    #         case ase.Expr("var_load", (str(varname),)):
    #             first_let, _ = find_parent_let(state.parents, varname)
    #             if replaced := single_use_repl.get(first_let):
    #                 return (yield replaced)
    #             else:
    #                 return expr
    #         case ase.Expr("let", (str(varname),
    #                               ase.Expr() as value,
    #                               ase.Expr() as inner)):
    #             uses = let_uses.get(expr)
    #             if uses is not None and uses <= 1:
    #                 single_use_repl[expr] = value
    #                 updated = ase.expr("let", varname, (yield value), (yield inner))
    #                 single_use_lets.add(updated)
    #                 return updated
    #     # Update node
    #     updated_args = []
    #     for x in expr.args:
    #         updated_args.append((yield x) if isinstance(x, ase.Expr) else x)
    #     return ase.expr(expr.head, *updated_args)


    def rewrite_let_into_lambda(expr: SExpr, state: ase.TraverseState):
        match expr:
            case ase.Expr("var_load", (str(varname),)):
                blet, depth = find_parent_let(state.parents, expr)
                if blet is None:
                    return lb.arg(parameters.index(varname) + depth)
                what = lb.arg(depth)
                if what.handle == 636:
                    breakpoint()
                return what

            case ase.Expr("let", (str(varname),
                                  ase.Expr() as value,
                                  ase.Expr() as inner)):
                return lb.app(lb.lam((yield inner)), (yield value))

            case ase.Expr("py_func", (str(), ase.Expr() as py_args, ase.Expr() as body)):
                last = (yield body)
                # one extra for .io
                for _ in range(len(py_args.args) + 1):
                    last = lb.lam(last)
                return last

        # Update node
        updated_args = []
        for x in expr.args:
            updated_args.append((yield x) if isinstance(x, ase.Expr) else x)
        return ase.expr(expr.head, *updated_args)


    with prgm.tape:
        # memo = prgm.traverse(simplify_single_use_let)
        # prgm = memo[prgm]
        memo = prgm.traverse(rewrite_let_into_lambda)
        prgm = memo[prgm]

    return prgm #lb.run_abstraction_pass(prgm)



def restructure_source(function):
    ast2scfg_transformer = AST2SCFGTransformer(function)
    astcfg = ast2scfg_transformer.transform_to_ASTCFG()
    scfg = astcfg.to_SCFG()
    scfg.restructure()
    scfg2ast = SCFG2ASTTransformer()
    original_ast = unparse_code(function)[0]
    transformed_ast = scfg2ast.transform(original=original_ast, scfg=scfg)

    transformed_ast = ast.fix_missing_locations(transformed_ast)

    print(ast.unparse(transformed_ast))

    prgm = convert_to_sexpr(transformed_ast)

    varinfo = find_variable_info(prgm)
    pprint(varinfo)

    rvsdg = convert_to_rvsdg(prgm, varinfo)

    from sealir.prettyformat import html_format
    pp(rvsdg)


    # out = html_format.to_html(rvsdg)
    # with open("debug.html", "w") as fout:
    #     print(html_format.write_html(out,fout))
    # return


    lam = convert_to_lambda(rvsdg, varinfo)
    pp(lam)


    # out = html_format.to_html(lam)
    with open("debug.html", "w") as fout:
        print(html_format.write_html(fout,
                                     html_format.to_html(rvsdg),
                                     html_format.to_html(lam)))

    # return

    return lam

    # # DEMO FIND ORIGINAL AST
    # print('---py_range----')
    # def find_py_range(rvsdg):
    #     for path, node in rvsdg.walk_descendants():
    #         if node.args and node.args[0] == "py_call":
    #             match node.args[1]:
    #                 case ase.Expr("expr", ("py_global_load", "range")):
    #                     return node

    # py_range = find_py_range(rvsdg)
    # print(py_range.str())

    # print('---find md----')
    # md = next(py_range.search_parents(lambda x: x.head == ".md.rewrite"))
    # (_, _, orig) = md.args
    # loc = orig.args[-1]

    # lines, offset = inspect.getsourcelines(function)
    # line_offset = loc.args[0]

    # print(offset + line_offset, '|', lines[line_offset])
    # rvsdg.tape.render_dot(show_metadata=True).view()

######################

@dataclass(frozen=True)
class EvalLamState(ase.TraverseState):
    context: EvalCtx


class EvalIO:
    def __repr__(self) -> str:
        return "IO"


class SExprValuePair(NamedTuple):
    expr: SExpr
    value: Any

@dataclass(frozen=True)
class EvalCtx:
    args: tuple[Any, ...]
    value_map: dict[SExpr, Any] = field(default_factory=dict)
    blam_stack: list[SExprValuePair] = field(default_factory=list)

    @classmethod
    def from_arguments(cls, *args):
        return EvalCtx(args=tuple(reversed([EvalIO(), *args])))

    def make_arg_node(self):
        ba = []
        for i, v in enumerate(self.args):
            se = ase.expr("bind_arg", i)
            self.value_map[se] = v
            ba.append(se)
        return ba

    @contextmanager
    def bind_app(self, lam_expr: SExpr, argval: Any):
        self.blam_stack.append(SExprValuePair(lam_expr, argval))
        try:
            yield
        finally:
            self.blam_stack.pop()


def ensure_io(obj: Any) -> EvalIO:
    assert isinstance(obj, EvalIO)
    return obj


def lambda_evaluation(expr: ase.Expr, state: EvalLamState):
    indent = len(state.parents)
    ctx = state.context
    from sealir.lam import _app
    try:
        print(" " * indent, "EVAL", expr.handle, expr.as_tuple(2))
        match expr:
            case ase.Expr("lam", (body,)):
                retval = (yield body)
                return retval
            case ase.Expr("app", (argval, ase.Expr() as lam_func)):
                with ctx.bind_app(lam_func, (yield argval)):
                    retval = (yield lam_func)
                    return retval
            case ase.Expr("arg", (int(argidx),)):
                for i, v in enumerate(reversed(ctx.blam_stack)):
                    print(' ', i, '--', v.value)
                print(" target", -argidx - 1)
                # breakpoint()
                retval = ctx.blam_stack[-argidx - 1].value
                return retval
            case ase.Expr("unpack", (int(idx),
                                     ase.Expr() as packed_expr)):
                packed = (yield packed_expr)
                retval = packed[idx]
                return retval
            case ase.Expr("bind_arg"):
                return ctx.value_map[expr]
            case ase.Expr("py_return", (ase.Expr() as iostate,
                                        ase.Expr() as retval)):
                return (yield iostate), (yield retval)
            case ase.Expr("py_binop", (str(op),
                                       ase.Expr() as iostate,
                                       ase.Expr() as lhs,
                                       ase.Expr() as rhs)):
                ioval = ensure_io((yield iostate))
                lhsval = (yield lhs)
                rhsval = (yield rhs)
                match op:
                    case "+":
                        retval = ioval, lhsval + rhsval
                        return retval
                raise NotImplementedError(op)
            case ase.Expr("py_inplace_binop", (str(op),
                                       ase.Expr() as iostate,
                                       ase.Expr() as lhs,
                                       ase.Expr() as rhs)):
                ioval = ensure_io((yield iostate))
                lhsval = (yield lhs)
                rhsval = (yield rhs)
                match op:
                    case "+":
                        lhsval += rhsval
                        retval = ioval, lhsval
                        return retval
                raise NotImplementedError(op)
            case _:
                raise AssertionError(expr.as_tuple())
    finally:
        if "retval" in locals():
            print(" " * indent, "=>", retval)



def sum1d(n: int, m: int) -> int:
    return n


def sum1d(n: int, m: int) -> int:
    a = m + n
    return a
    # c = a
    # if m > n:
    #     a = b = n + m
    # else:
    #     a = b = n * m
    # c += a
    return c

# def sum1d(n: int) -> int:
#     c = 0
#     for i in range(n):
#         c += i
#     return c

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

def test_return_arg0():
    def udt(n: int, m: int) -> int:
        return n
    args = (12, 32)
    run(udt, args)

def test_return_arg1():
    def udt(n: int, m: int) -> int:
        return m

    args = (12, 32)
    run(udt, args)

def test_simple_add():
    def udt(n: int, m: int) -> int:
        a = n + m
        return a

    args = (12, 32)
    run(udt, args)

def test_inplace_add():
    def udt(n: int, m: int) -> int:
        a = n + m
        a += n
        return a
    args = (12, 32)
    run(udt, args)





def run(func, args):
    lam = restructure_source(func)

    # Prepare run
    lb = LamBuilder(lam.tape)

    ctx = EvalCtx.from_arguments(*args)
    with lam.tape:
        app_root = lb.app(lam, *ctx.make_arg_node())

    out = lb.format(app_root)
    print(out)

    memo = app_root.traverse(lambda_evaluation, EvalLamState(context=ctx))
    res = memo[app_root]
    print('result', res)
    got = res[1]

    assert got == func(*args)
    return got



