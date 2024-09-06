from __future__ import annotations
import sys
import ast
import inspect
from typing import Type, TypeAlias, Any, Iterator, Sequence
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


SExpr: TypeAlias = ase.Expr


def pp(expr: SExpr):
    print(pformat(expr.as_tuple(-1, dedup=True)).replace(',', '').replace('\'', ''))


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
    expr_effect: dict
    block_effect: dict
    assign_map: dict

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

    expr_effect: dict[SExpr, VarEffect] = defaultdict(VarEffect)

    # block_effect tracks the variables used by each statement in a block.
    # It shows which variable-uses are required for the remaining statements
    # in the block to execute. Variable-defs are local to that statement.
    block_effect: dict[SExpr, deque[VarEffect]] = defaultdict(deque)

    class FindDataDependence(ase.TreeVisitor):
        def visit(self, expr: SExpr):
            match expr:
                case ase.Expr("PyAst_block", body):
                    blkeff = block_effect[expr]
                    iter_stmt = iter(reversed(body))
                    last = next(iter_stmt)
                    blkeff.appendleft(VarEffect().merge(expr_effect[last]))
                    for stmt in iter_stmt:
                        lasteff = blkeff[0]
                        # Start with the expression level effect.
                        eff = VarEffect().merge(expr_effect[stmt])
                        blkeff.appendleft(eff)
                        # Keep track of future uses discounting newly defined
                        eff.uses.update(lasteff.uses - expr_effect[stmt].defs)
                        last = stmt
                    expr_effect[expr] = blkeff[0]

                case ase.Expr("PyAst_Name", (name, "load", loc)):
                    if name not in nonlocals:
                        expr_effect[expr].use(name)
                case ase.Expr("PyAst_Name", (name, "store", loc)):
                    expr_effect[expr].define(name)
                case ase.Expr("PyAst_arg", (name, anno, loc)):
                    expr_effect[expr].define(name)
                case _ if expr.head in {"PyAst_Name", "PyAst_arg", "PyAst_block"}:
                    raise AssertionError(expr)
                case _:
                    for child in expr.args:
                        if isinstance(child, ase.Expr):
                            expr_effect[expr].merge(expr_effect[child])

    expr.apply_bottomup(FindDataDependence())

    assign_map = defaultdict(dict)

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
        expr_effect=expr_effect,
        block_effect=block_effect,
        assign_map=assign_map,
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
    def unpack(val: SExpr, targets: Sequence[str], *, force_io=False) -> Iterator[tuple[SExpr, str]]:
        if force_io or uses_io(val):
            targets  = (".io", *targets)
        if len(targets) == 1:
            yield val, targets[0]
        else:
            for i, target in enumerate(targets):
                yield ase.expr("unpack", val, i), target

    def lookup(blk: ase.Expr):
        defs = set()
        class FindLet:
            def visit(self, expr: SExpr):
                match expr:
                    case ase.Expr("let", (varname, *_)):
                        defs.add(varname)

        blk.apply_topdown(FindLet())
        return defs

    # XXX: THIS IS NOT THE ACTUAL ALGORITHM. JUST A MOCK
    class Convert2RVSDG(TreeRewriter[SExpr]):
        def rewrite_generic(self, orig: SExpr, args: tuple[Any, ...], updated: bool) -> SExpr:
            raise NotImplementedError(orig.head)

        def rewrite_PyAst_Name(self, orig: ase.Expr, name: str, ctx: str, loc: SExpr) -> SExpr:
            if name in varinfo.nonlocals:
                return ase.expr("py_global_load", name)
            else:
                match ctx:
                    case "store":
                        return name
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

        def rewrite_PyAst_Assign(self, orig: ase.Expr, val: SExpr, *rest: SExpr) -> SExpr:
            [*targets, loc] = rest
            return ase.expr("assign", val, *targets)

        def rewrite_PyAst_AugAssign(self, orig: ase.Expr, opname: str, left: SExpr, right: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("assign", ase.expr("py_inplace_binop", opname, left, right), left)

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
                        for unpacked, target in reversed(list(unpack(val, targets))):
                            last = ase.expr("let", target, unpacked, last)
                    case ase.Expr("py_if", (test, blk_true, blk_false)):
                        # look up variable in each block to find variable
                        # definitions
                        defs_true = lookup(blk_true)
                        defs_false = lookup(blk_false)
                        defs = sorted(defs_true | defs_false)

                        if uses_io(test):
                            [(io, iovar), (cond, _)] = unpack(test, ["cond"], force_io=True)
                            stmt = ase.expr("scfg_if", cond, blk_true, blk_false)
                            stmt = ase.expr("let", iovar, io, stmt)

                        for i, var in reversed(list(enumerate(defs))):
                            extract = ase.expr("unpack", stmt, i)
                            last = ase.expr("let", var, extract, last)
                    case ase.Expr("scfg_while", (ase.Expr() as test, loopblk)):
                        defs = sorted(lookup(loopblk))
                        for i, var in reversed(list(enumerate(defs))):
                            extract = ase.expr("unpack", stmt, i)
                            last = ase.expr("let", var, extract, last)

                    case ase.Expr("py_pass" | "py_return"):
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

        def rewrite_PyAst_Return(self, orig: ase.Expr, val: SExpr, loc: SExpr) -> SExpr:
            return ase.expr("py_return", self.get_io(), val)

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


    # pprint(rvsdg.as_dict(), compact=True, sort_dicts=False)

    pp(rvsdg)

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


# def sum1d(n: int, m: int) -> int:
#     c = 0
#     a = n + m
#     if m > n:
#         a = b = n + m
#     else:
#         a = b = n * m
#     c += a
#     return c

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


def main():
    source = restructure_source(sum1d)
    print(source)


main()
