from __future__ import annotations

import string
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial, reduce
from types import SimpleNamespace
from typing import Any, Iterator, no_type_check

import pytest

from sealir import ase, egg_utils, scf
from sealir.itertools import first
from sealir.lam import LamBuilder
from sealir.rewriter import TreeRewriter

from .test_scf import make_sum_reduce_loop


class MakeTypeInferRules(TreeRewriter[ase.Expr]):
    flag_save_history = False
    type_expr_tree: ase.Tape

    def __init__(self, type_expr_tree):
        super().__init__()
        self.type_expr_tree = type_expr_tree

    def new_typevar(self, orig: ase.Expr) -> ase.Expr:
        with self.type_expr_tree:
            return ase.expr("typevar", orig._handle)

    def new_type(self, *args) -> ase.Expr:
        with self.type_expr_tree:
            return ase.expr("type", *args)

    def new_trait(self, *args) -> ase.Expr:
        with self.type_expr_tree:
            return ase.expr("trait", *args)

    def new_equiv(self, *args) -> ase.Expr:
        with self.type_expr_tree:
            return ase.expr("rule-equiv", *args)

    def new_proof(self, *args) -> ase.Expr:
        with self.type_expr_tree:
            return ase.expr("rule-proof", *args)

    def new_proof_of(self, *args) -> ase.Expr:
        with self.type_expr_tree:
            return ase.expr("proof-of", *args)

    def new_or(self, *args) -> ase.Expr:
        with self.type_expr_tree:
            return reduce(lambda x, y: ase.expr("or", x, y), args)

    def new_result_of(self, func: ase.Expr) -> ase.Expr:
        with self.type_expr_tree:
            return ase.expr("result-of", func)

    def new_isa(self, lhs: ase.Expr, rhs: ase.Expr) -> ase.Expr:
        with self.type_expr_tree:
            return ase.expr("isa", lhs, rhs)

    def find_arg(self, orig_body: ase.Expr) -> Iterator[ase.Expr]:
        for parents, child in orig_body.walk_descendants():
            if child.head == "arg":
                lam_depth = len([p for p in parents if p.head == "lam"])
                [argidx] = child.args
                if argidx == lam_depth:
                    yield child

    def apply_arg_rules(self, orig_body: ase.Expr, arg: ase.Expr) -> None:
        for arg_node in self.find_arg(orig_body):
            self.new_equiv(arg, self.memo[arg_node])

    def rewrite_generic(
        self, orig: ase.Expr, args: tuple[Any, ...], updated: bool
    ):
        raise NotImplementedError(orig)

    def rewrite_arg(self, orig: ase.Expr, argindex: int):
        return self.new_typevar(orig)

    def rewrite_lam(self, orig: ase.Expr, body: ase.Expr):
        [orig_body] = orig.args
        assert isinstance(orig_body, ase.Expr)
        tv = self.new_typevar(orig)
        arg_node = first(self.find_arg(orig_body))
        self.new_equiv(tv, self.new_type("Lam", body, self.memo[arg_node]))
        return tv

    def rewrite_app(self, orig: ase.Expr, lam: ase.Expr, arg: ase.Expr):
        tv = self.new_typevar(orig)
        self.new_equiv(tv, self.new_type("App", lam, arg))
        return tv

    def rewrite_expr(self, orig: ase.Expr, opname: str, *args):

        def binop_equiv(opname):
            lproto = f"HasL{opname.capitalize()}"
            rproto = f"HasR{opname.capitalize()}"
            lhs, rhs = args
            tv = self.new_typevar(orig)
            fn = self.new_type("Func", opname, lhs, rhs)
            self.new_equiv(tv, self.new_result_of(fn))

            # Traits
            Isa = self.new_isa
            self.new_proof(
                tv,
                self.new_or(
                    Isa(lhs, self.new_trait(lproto, rhs)),
                    Isa(rhs, self.new_trait(rproto, lhs)),
                ),
            )
            return tv

        def cmpop_equiv(opname):
            lproto = f"HasL{opname.capitalize()}"
            rproto = f"HasR{opname.capitalize()}"
            lhs, rhs = args
            tv = self.new_typevar(orig)
            fn = self.new_type("Func", opname, lhs, rhs)
            self.new_equiv(tv, self.new_result_of(fn))
            self.new_equiv(tv, self.new_type("Bool"))

            # Traits
            Isa = self.new_isa
            self.new_proof(
                tv,
                self.new_or(
                    Isa(lhs, self.new_trait(lproto, rhs)),
                    Isa(rhs, self.new_trait(rproto, lhs)),
                ),
            )
            return tv

        match opname:
            case "int":
                tv = self.new_typevar(orig)
                self.new_equiv(tv, self.new_type("Int"))
                return tv
            case "float":
                tv = self.new_typevar(orig)
                self.new_equiv(tv, self.new_type("Float"))
                return tv
            case "add":
                return binop_equiv("add")
            case "mul":
                return binop_equiv("mul")
            case "lt":
                return cmpop_equiv("lt")
            case "tuple":
                return self.new_type("Tuple", *args)
            case "tuple.getitem":
                [tup, idx] = args
                [_, orig_tup, orig_idx] = orig.args
                tv = self.new_typevar(orig)
                if not isinstance(orig_idx, int):
                    match orig_idx.args:
                        case ("num", constant):
                            if idx.head == "typevar":
                                orig_idx = constant
                                self.new_equiv(idx, self.new_type("Int"))
                        case _:
                            raise NotImplementedError(orig_idx)
                else:
                    self.new_equiv(
                        tv, self.new_type("TupleGetitem", tup, orig_idx)
                    )
                return tv
            case "scf.dowhile":
                [loop_body, loop_arg] = args
                [_, orig_body, _] = orig.args
                tv = self.new_typevar(orig)
                self.new_equiv(tv, self.new_type("App", loop_body, loop_arg))
                self.apply_arg_rules(orig_body, loop_arg)
                return tv
            case "scf.switch":
                [cond, arg, *bodies] = args
                self.new_equiv(cond, self.new_type("Bool"))
                branches_tvs = []
                for body in bodies:
                    branch_tv = self.new_typevar(body)
                    branches_tvs.append(branch_tv)
                    self.new_equiv(branch_tv, self.new_type("App", body, arg))
                    self.apply_arg_rules(body, arg)

                tv = self.new_typevar(orig)
                self.new_equiv(tv, *branches_tvs)
                raise AssertionError(
                    """
Solve the merge type.
Currently it doesn't propagate
"""
                )
                # self.new_equiv(tv, self.new_type("Merge", cond, *branches_tvs))
                self.new_proof(tv, self.new_proof_of(cond))
                return tv
            case _:
                raise NotImplementedError(opname)


def find_relevant_rules(root: ase.Expr, equiv_list: list[ase.Expr]):
    relset = {root}

    def update_relevant(equiv):
        nonlocal relset
        local_relset = set()
        result = False
        for _, child in equiv.walk_descendants():
            local_relset.add(child)
            if child in relset:
                result = True

        if result:
            relset.update(local_relset)
        return result

    for equiv in reversed(equiv_list):
        update_relevant(equiv)

    return sorted(
        [rule for rule in relset if rule.head == "equiv"],
        key=lambda x: x._handle,
    )


def replace_equivalent(equiv_list: list[ase.Expr]):
    from collections import defaultdict
    from pprint import pprint

    equiv_map = defaultdict(set)

    def match(tv, others):
        equiv_map[tv] |= set(others)

    for equiv in equiv_list:

        def get_typevars():
            for arg in equiv.args:
                if arg.head == "typevar":
                    yield arg

        if tvs := list(get_typevars()):
            [first, *others] = tvs
            match(first, others)

    # propagate equivalent
    while True:
        redir_map = {}
        for lhs, eqset in list(equiv_map.items()):
            for rhs in list(eqset):
                rhs = redir_map.get(rhs, rhs)
                if rhs in equiv_map:
                    eqset.update(equiv_map.pop(rhs))
                    redir_map[rhs] = lhs
        if not redir_map:
            break

    # build replacement map
    repl_map = {}
    for k, vs in equiv_map.items():
        for v in vs:
            repl_map[v] = k
    for k, v in repl_map.items():
        print("equiv", k.str(), v.str())

    # rewrite to simplify
    class Replace(TreeRewriter):
        pass
        # def rewrite_equiv(self, orig, *nodes):
        #     repl = tuple(repl_map.get(node, node) for node in nodes)
        #     if repl != nodes:
        #         print("old", *(n.str() for n in nodes))
        #         print("new", *(n.str() for n in repl))
        #         breakpoint()
        #     return ase.expr(orig.head, *repl)

    ps = Replace()
    ps.memo.update(repl_map)
    for eq in equiv_list:
        eq.apply_bottomup(ps)
    return ps.memo


@no_type_check
def do_egglog(equiv_list):

    from egglog import (
        EGraph,
        Expr,
        Set,
        String,
        Vec,
        birewrite,
        eq,
        function,
        i64,
        i64Like,
        method,
        rewrite,
        rule,
        ruleset,
        set_,
        union,
        var,
        vars_,
    )

    egraph = EGraph()

    class TypeInfo(Expr):
        # @classmethod
        # def typevar(cls, ident: i64Like) -> TypeInfo:
        #     ...

        @classmethod
        def type(cls, name: String) -> TypeInfo: ...
        @classmethod
        def typevar(cls, ident: i64Like) -> TypeInfo: ...

        @classmethod
        def tuple(cls, type_list: Vec[TypeInfo]) -> TypeInfo: ...

        @classmethod
        def arrow(cls, arg: TypeInfo, res: TypeInfo) -> TypeInfo: ...

        @classmethod
        def merge(cls, cond: TypeInfo, args: Vec[TypeInfo]) -> TypeInfo: ...

    class TypeProof(Expr):
        @classmethod
        def trait(cls, name: String, args: Vec[TypeInfo]) -> TypeProof: ...

        @classmethod
        def arrow(cls, arg: TypeProof, res: TypeProof) -> TypeProof: ...

        @classmethod
        def isa(cls, lhs: TypeProof, rhs: TypeProof) -> TypeProof: ...

        @classmethod
        def or_(self, lhs: TypeProof, rhs: TypeProof) -> TypeProof: ...

    @function
    def f_tuple_getitem(tup: TypeInfo, idx: i64Like) -> TypeInfo: ...

    @function
    def f_lam(body: TypeInfo, arg: TypeInfo) -> TypeInfo: ...

    @function
    def f_app(lam: TypeInfo, arg: TypeInfo) -> TypeInfo: ...

    @function
    def f_result_of(func: TypeInfo) -> TypeInfo: ...

    @function
    def f_proof(tv: TypeInfo) -> TypeProof: ...

    @function
    def f_equiv(a: TypeInfo, b: TypeInfo) -> TypeInfo: ...

    @function
    def f_func(opname: String, args: Vec[TypeInfo]) -> TypeInfo: ...

    def make_tuple(*elems):
        return TypeInfo.tuple(Vec(*elems))

    # def make_empty():
    #     return TypeInfo.union(Set[TypeInfo].empty())

    def make_func(opname: str, *args: TypeInfo) -> TypeInfo:
        return f_func(opname, Vec(*args))

    def make_merge(cond: TypeInfo, *args: TypeInfo) -> TypeInfo:
        return TypeInfo.merge(cond, Vec(*args))

    def make_trait(name, *args):
        if not args:
            return TypeProof.trait(name, Vec[TypeInfo].empty())
        else:
            return TypeProof.trait(name, Vec[TypeInfo](*args))

    type_facts, proof_facts, target, tvs = build_egglog_statements(
        equiv_list, SimpleNamespace(**locals())
    )

    # ty = lambda x: TypeInfo.type(x)
    # tup_type = TypeInfo.tuple(Vec(ty("Int"), ty("Real")))

    a, b, c, d = vars_("a b c d", TypeInfo)
    i, j = vars_("i j", i64)
    vec1, vec2 = vars_("vec1 vec2", Vec[TypeInfo])
    set1, set2 = vars_("set1 set2", Set[TypeInfo])
    name1 = var("name1", String)

    rules = [
        # f_equiv propagation
        rule(
            f_equiv(a, b),
        ).then(
            union(a).with_(b),
            union(f_proof(a)).with_(f_proof(b)),
        ),
        rule(eq(c).to(TypeInfo.arrow(a, b))).then(
            union(f_proof(c)).with_(TypeProof.arrow(f_proof(a), f_proof(b)))
        ),
        # Tuple rules
        rule(
            eq(a).to(TypeInfo.tuple(vec1)),
            eq(b).to(f_tuple_getitem(a, i)),
        ).then(
            f_equiv(b, vec1[i]),
        ),
        # Arrow rules
        rule(  # Given
            # a = lam b c
            eq(a).to(f_lam(b, c)),
            # d = typevar(i)
            eq(d).to(TypeInfo.typevar(i)),
            # a = b
            eq(a).to(d),
        ).then(
            # then a = c -> b
            f_equiv(a, TypeInfo.arrow(c, b)),
        ),
        # App rules
        rule(
            eq(a).to(f_app(TypeInfo.arrow(b, c), d)),
        ).then(
            f_equiv(a, c),
            f_equiv(b, d),
        ),
    ]

    rules += [
        rule(
            eq(a).to(make_func("mul", b, c)),
            eq(b).to(TypeInfo.type("Int")),
            eq(c).to(TypeInfo.type("Int")),
        ).then(
            f_equiv(f_result_of(a), TypeInfo.type("Int")),
        )
    ]
    for a, b in type_facts:
        rules.append(f_equiv(a, b))
    for a, b in proof_facts:
        rules.append(union(a).with_(b))

    # for tv in tvs:
    #     rules.append( f_proof(TypeInfo.typevar(tv)) )

    egraph.register(*rules)

    # saturate
    for i in range(100):
        if not egraph.run(5).updated:
            break

    # print("Extract target")

    # for tv in tvs:
    #     out = egraph.extract( f_proof(TypeInfo.typevar(tv)), 100 )
    #     print(out)
    # # out = egraph.extract(target, include_cost=True)
    # print(out)
    # for out in egraph.extract_multiple(target, 10):
    #     print(out)

    tyinferdata = TypeInferData(egraph)
    tyinferdata.get_eclasses()

    # namer = UnknownNamer()
    # for eclass, tyset in tyinferdata.eclass_to_types.items():
    #     pp = tyinferdata.prettyprint_typeset(tyset, namer)
    #     print(f"eclass {eclass} :: {pp}")

    # egraph.display()
    return tyinferdata


class UnknownNamer:
    def __init__(self):
        self._stored = {}

        def gen():
            for ch in string.ascii_lowercase:
                yield ch

            i = 1
            while True:
                for ch in string.ascii_lowercase:
                    yield f"{ch}{i}"
                i += 1

        self._namer = iter(gen())

    def get(self, key) -> str:
        if key not in self._stored:
            name = next(self._namer)
            self._stored[key] = name
        else:
            name = self._stored[key]
        return name


class TypeInferData:

    def __init__(self, egraph):
        self.egraph = egraph

    def get_eclasses(self):
        egraph = self.egraph
        eclass_data = egg_utils.extract_eclasses(egraph)
        self.eclass_data = eclass_data

        # determine type of each eclass
        eclass_to_types = {}
        for ec, members in eclass_data.eclasses.items():

            def is_type(x):
                return (
                    x.type == "TypeInfo"
                    and x.op != "TypeInfo.typevar"
                    and not x.op.startswith("f_")
                )

            types = {eclass_data.terms[x.key] for x in members if is_type(x)}
            if types:
                eclass_to_types[ec] = types
        self.eclass_to_types = eclass_to_types

        # build map of typevar to eclass
        typevar_to_eclass = {}
        TermRef = egg_utils.TermRef
        for ec, members in eclass_data.eclasses.items():
            for each in members:
                match each:
                    case TermRef(key, _, "TypeInfo.typevar"):
                        term = eclass_data.terms[key]
                        tvid = eclass_data.terms[term.children[0].key].op
                        typevar_to_eclass[tvid] = ec
        self.typevar_to_eclass = typevar_to_eclass

        # build map of eclass to trait
        eclass_to_trait = {}
        for ec, members in eclass_data.eclasses.items():

            def is_trait(x):
                return x.type == "TypeProof" and not x.op.startswith("f_")

            traits = set(filter(is_trait, members))
            if traits:
                eclass_to_trait[ec] = set(
                    map(lambda x: eclass_data.terms[x.key], traits)
                )
        self.eclass_to_trait = eclass_to_trait
        # pprint(self.eclass_to_trait)

        eclass_to_typevar = defaultdict(set)
        for k, v in typevar_to_eclass.items():
            eclass_to_typevar[v].add(k)

        # build map of typevar to proof-eclass
        typevar_to_proof_eclass = {}

        for ec, members in eclass_data.eclasses.items():
            proofs = [
                eclass_data.terms[x.key] for x in members if x.op == "f_proof"
            ]
            for proof in proofs:
                [child] = proof.children
                tv_eclass = child.eclass
                for k in eclass_to_typevar[tv_eclass]:
                    typevar_to_proof_eclass[k] = proof.eclass
        self.typevar_to_proof_eclass = typevar_to_proof_eclass
        # pprint(self.typevar_to_proof_eclass)

    def prettyprint_typeset(
        self, tyset: set[egg_utils.Term], namer, *, use_trait=False
    ):

        parts = []
        for term in tyset:
            text = self.pretty(term, namer, use_trait=use_trait)
            parts.append(text)
        if use_trait:
            if len(parts) > 1:
                parts = [f"({x})" for x in parts]
            return "*".join(parts)
        else:
            return "|".join(parts)

    def prettyprint_eclass(self, ec: str, namer, *, use_trait=False):
        if ec in stack:
            out = f"(cycle {ec})"
            raise ValueError(out)
            out = f"(cycle {namer.get(ec)})"
            return out

        stack.append(ec)
        with ExitStack() as raii:

            @raii.push
            def _(*exc):
                stack.pop()

            tyset: set[egg_utils.Term]
            if use_trait:

                def get_concrete(ec):
                    tys = self.eclass_to_types.get(ec)
                    # if tys:
                    #     return {t for t in tys if t.op != "TypeInfo.arrow"}
                    return tys

                if concrete := get_concrete(ec):
                    return self.prettyprint_typeset(
                        concrete, namer, use_trait=False
                    )
                else:
                    lookup = self.eclass_to_trait
            else:
                lookup = self.eclass_to_types
            if tyset := lookup.get(ec):
                return self.prettyprint_typeset(
                    tyset, namer, use_trait=use_trait
                )
            else:
                return namer.get(ec)

    def pretty(self, ty: egg_utils.Term, namer, *, use_trait) -> str:
        pp = partial(self.prettyprint_eclass, namer=namer, use_trait=use_trait)

        match ty.op:
            case "TypeInfo.type":
                [arg] = ty.children
                return arg.op
            case "TypeInfo.arrow":
                [ec_arg, ec_res] = map(lambda x: x.eclass, ty.children)
                return f"{pp(ec_arg)}->{pp(ec_res)}"
            case "TypeInfo.tuple":
                [args] = ty.children
                eclasses = [
                    x.eclass for x in self.eclass_data.terms[args.key].children
                ]
                parts = ", ".join(pp(x) for x in eclasses)
                return f"tuple[{parts}]"
            case "TypeInfo.typevar":
                [arg] = ty.children
                ident = arg.op
                return namer.get(ident)
            case "TypeInfo.merge":
                [cond, args] = ty.children
                # args must be a Vec
                eclasses = [
                    x.eclass for x in self.eclass_data.terms[args.key].children
                ]
                parts = ", ".join(pp(x) for x in eclasses)
                return f"Merge({pp(cond)}, {parts})"
            case "TypeProof.trait":
                [op, args] = ty.children
                # args must be a Vec
                eclasses = [
                    x.eclass for x in self.eclass_data.terms[args.key].children
                ]
                return f"(!{op.op} {' '.join(map(pp, eclasses))})"
            case "TypeProof.isa":
                [lhs, rhs] = ty.children
                return f"(isa {namer.get(lhs.eclass)} {pp(rhs.eclass)})"
            case "TypeProof.arrow":
                [arg, res] = ty.children
                return f"{pp(arg.eclass)}->{pp(res.eclass)}"
            case "TypeProof.or_":
                [lhs, rhs] = ty.children
                return f"{pp(lhs.eclass)}+{pp(rhs.eclass)}"
            case x:
                assert False, x


stack = []


@dataclass(frozen=True)
class TypeRef:
    eclass: str
    eclass_ty_map: dict = field(hash=False)

    def get(self):
        return self.eclass_ty_map[self.eclass]

    def __repr__(self) -> str:
        tyset = self.eclass_ty_map[self.eclass]
        if len(tyset) == 1:
            [ty] = tyset
            return str(ty)
        else:
            return f"{self.__class__.__name__}({self.eclass})"


def build_egglog_statements(equiv_list, node_dct):
    def flatten():
        for rule in equiv_list:
            args = rule.args
            assert args[0].head == "typevar", repr(rule.args)
            match len(args):
                case 2:
                    yield (rule.head, *args)
                case n:
                    assert n > 2
                    [head, *tail] = args
                    for other in tail:
                        yield rule.head, head, other

    typevars = {}

    def proc_second(rhs):
        match rhs.as_tuple():
            case ("typevar", y):
                tv = node_dct.TypeInfo.typevar(y)
                typevars[y] = tv
                return tv
            case ("trait", name, *args):
                return node_dct.make_trait(name, *map(proc_second, args))
            case ("type", *fields):
                match fields:
                    case (name,):
                        return node_dct.TypeInfo.type(name)
                    case ("TupleGetitem", tup_tv, idx):
                        assert isinstance(idx, int)
                        return node_dct.f_tuple_getitem(
                            proc_second(tup_tv), idx
                        )
                    case ("Tuple", *elems):
                        return node_dct.make_tuple(*map(proc_second, elems))
                    case ("Lam", bodyexpr, argexpr):
                        return node_dct.f_lam(
                            proc_second(bodyexpr), proc_second(argexpr)
                        )
                    case ("App", bodyexpr, argexpr):
                        return node_dct.f_app(
                            proc_second(bodyexpr), proc_second(argexpr)
                        )
                    case ("Func", opname, *args):
                        assert isinstance(opname, str)
                        return node_dct.make_func(
                            opname, *map(proc_second, args)
                        )
                    case ("Merge", cond, *args):
                        return node_dct.make_merge(
                            proc_second(cond), *map(proc_second, args)
                        )
                    case unknown:
                        raise ValueError(unknown)
            case ("result-of", func):
                return node_dct.f_result_of(proc_second(func))
            case ("proof-of", func):
                return node_dct.f_proof(proc_second(func))
            case ("isa", lhs, rhs):
                return node_dct.TypeProof.isa(
                    node_dct.f_proof(proc_second(lhs)), proc_second(rhs)
                )
            case ("or", lhs, rhs):
                return node_dct.TypeProof.or_(
                    proc_second(lhs), proc_second(rhs)
                )
            case _:
                raise ValueError(rhs.str())

    equiv_pairs = flatten()
    type_stmts = []
    proof_stmts = []

    for op, lhs, rhs in equiv_pairs:
        assert lhs.head == "typevar", (lhs.str(), rhs.str())
        match lhs.as_tuple():
            case "typevar", x:
                pass
            case _:
                raise AssertionError(f"{op} {lhs.str()} {rhs.str()}")
        first = node_dct.TypeInfo.typevar(x)
        second = proc_second(rhs)
        match op:
            case "rule-equiv":
                type_stmts.append((first, second))
            case "rule-proof":
                proof_stmts.append((node_dct.f_proof(first), second))
            case _:
                raise AssertionError(op)
    target = first
    return type_stmts, proof_stmts, target, typevars


@pytest.mark.xfail
def test_typeinfer():
    my_function = make_sum_reduce_loop

    def my_function(lambar):
        @lambar.lam_func
        def func_body(x, y):
            n = lambar.expr("mul", x, y)
            return n

        # root = lambar.app(func_body, lambar.expr("int", 1), lambar.expr("int", 2))
        # root = lambar.app(func_body, lambar.expr("int", 1))
        root = func_body
        root = lambar.run_abstraction_pass(root)
        return root

    def my_function(lambar):
        @lambar.lam_func
        def func_body(x, y):
            """
            Same as

            def func_body(x, y):
                if x < y:
                    out = x * y
                else:
                    out = x * 1
                return out
            """
            cond = lambar.expr("lt", x, y)  # (lt x y)

            @scf.region(lambar)
            def true_branch(x, y):
                return lambar.expr("mul", x, y)  # (mul x y)

            @scf.region(lambar)
            def false_branch(x, y):
                return lambar.expr(
                    "mul", x, lambar.expr("int", 1)
                )  # (mul x (int 1))

            tup = lambar.expr("tuple", x, y)
            n = lambar.expr("scf.switch", cond, tup, true_branch, false_branch)
            return n

        # root = lambar.app(func_body, lambar.expr("int", 1), lambar.expr("int", 2))
        # root = lambar.app(func_body, lambar.expr("int", 1))
        root = func_body
        root = lambar.run_abstraction_pass(root)
        return root

    lambar = LamBuilder()
    my_function(lambar)
    lambar = lambar.simplify()
    func_body = lambar.get_root()

    print(lambar.format(func_body))

    type_expr_tree = ase.Tape()
    typeinfer = MakeTypeInferRules(type_expr_tree)
    func_body.apply_bottomup(typeinfer)
    map_op_to_typevar = typeinfer.memo

    equiv_list = [
        expr
        for expr in type_expr_tree.iter_expr()
        if expr.head.startswith("rule-")
    ]

    print("equiv".center(80, "-"))
    for equiv in equiv_list:
        print(equiv.str())
    print("=" * 80)

    tyinferdata = do_egglog(equiv_list)
    namer = UnknownNamer()

    for op in sorted(map_op_to_typevar, key=lambda x: x._handle):
        tv_key = str(op._handle)

        eclass = tyinferdata.typevar_to_eclass.get(tv_key)
        print(f"typevar {tv_key} eclass {eclass}".center(80, "-"))
        print("op", op.str())
        print(
            f" type  {namer.get(eclass)} :: ",
            tyinferdata.prettyprint_eclass(eclass, namer),
        )

        trait_eclass = tyinferdata.typevar_to_proof_eclass.get(tv_key)
        print(
            f" proof {namer.get(trait_eclass)} :: ",
            last := tyinferdata.prettyprint_eclass(
                trait_eclass, namer, use_trait=True
            ),
        )
        print("trait_eclass", trait_eclass)
