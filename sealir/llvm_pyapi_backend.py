from __future__ import annotations

import ctypes as _ct
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Self

from llvmlite import binding as llvm
from llvmlite import ir

from sealir import ase, lam, rvsdg

ll_byte = ir.IntType(8)
ll_pyobject_ptr = ll_byte.as_pointer()
ll_iostate = ir.LiteralStructType([])  # empty struct


def llvm_codegen(root: ase.SExpr):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    mod = ir.Module()

    # make function
    arity, bodynode = determine_arity(root)
    assert arity >= 1
    actual_num_args = arity - 1  # due to iostate
    fnty = ir.FunctionType(
        ll_pyobject_ptr, [ll_pyobject_ptr] * actual_num_args
    )
    fn = ir.Function(mod, fnty, name="foo")

    builder = ir.IRBuilder(fn.append_basic_block())

    ctx = CodegenCtx(
        llvm_module=mod,
        llvm_func=fn,
        builder=builder,
        pyapi=PythonAPI(builder),
    )
    for i in reversed(range(actual_num_args)):
        ctx.blam_stack.append(BLamArg(argidx=i))
    ctx.blam_stack.append(BLamIOArg())  # iostate

    memo = ase.traverse(bodynode, _codegen_loop, CodegenState(context=ctx))

    llvm_ir = str(mod)
    print(llvm_ir)
    # Create JIT
    lljit = llvm.create_lljit_compiler()
    rt = (
        llvm.JITLibraryBuilder()
        .add_ir(llvm_ir)
        .export_symbol("foo")
        .add_current_process()
        .link(lljit, "foo")
    )
    ptr = rt["foo"]
    return JitCallable.from_pointer(rt, ptr, actual_num_args)


@dataclass(frozen=True)
class JitCallable:
    rt: llvm.ResourceTracker
    pyfunc: Callable

    @classmethod
    def from_pointer(
        cls, rt: llvm.ResourceTracker, ptr: int, arity: int
    ) -> Self:
        pyfunc = _ct.PYFUNCTYPE(_ct.py_object, *([_ct.py_object] * arity))(ptr)
        return JitCallable(rt=rt, pyfunc=pyfunc)

    def __call__(self, *args: Any) -> Any:
        return self.pyfunc(*args)


@dataclass(frozen=True)
class CodegenState(ase.TraverseState):
    context: CodegenCtx


def _codegen_loop(expr: ase.BasicSExpr, state: CodegenState):
    if False:
        (yield None)

    ctx = state.context
    builder = ctx.builder
    pyapi = ctx.pyapi

    def ensure_io(val):
        assert isinstance(val, BLamIOArg), val
        return val

    match expr:
        case lam.Arg(int(debruijn)):
            sp = -debruijn - 1
            sv = ctx.blam_stack[sp]
            match sv:
                case BLamIOArg():
                    return sv
                case BLamArg(int(idx)):
                    return builder.function.args[idx]
                case BLamValue(val=val):
                    return val
                case _:
                    raise NotImplementedError(sv)
        case lam.App(arg=argval, lam=lam_func):
            with ctx.bind_app(lam_func, (yield argval)):
                return (yield lam_func)
        case lam.Lam(body=body):
            return (yield body)
        case lam.Unpack(idx=int(idx), tup=packed_expr):
            packed = yield packed_expr
            retval = packed[idx]
            return retval
        case rvsdg.Py_BinOp(opname=str(op), iostate=iostate, lhs=lhs, rhs=rhs):
            ioval = ensure_io((yield iostate))
            lhsval = yield lhs
            rhsval = yield rhs
            match op:
                case "+":
                    retval = ctx.pyapi.number_add(lhsval, rhsval)
                case "-":
                    retval = ctx.pyapi.number_subtract(lhsval, rhsval)
                case "*":
                    retval = ctx.pyapi.number_multiply(lhsval, rhsval)
                case "/":
                    retval = ctx.pyapi.number_truedivide(lhsval, rhsval)
                case "//":
                    retval = ctx.pyapi.number_floordivide(lhsval, rhsval)
                case _:
                    raise NotImplementedError(op)
            return ioval, retval

        case rvsdg.Return(iostate=iostate, retval=retval):
            ensure_io((yield iostate))
            retval = yield retval
            builder.ret(retval)

        case _:
            raise NotImplementedError(ase.as_tuple(expr, depth=2))


@dataclass(frozen=True)
class BLamBase:
    pass


@dataclass(frozen=True)
class BLamIOArg(BLamBase):
    pass


@dataclass(frozen=True)
class BLamArg(BLamBase):
    argidx: int


@dataclass(frozen=True)
class BLamValue(BLamBase):
    lam: ase.SExpr
    val: Any


@dataclass(frozen=True)
class CodegenCtx:
    llvm_module: ir.Module
    llvm_func: ir.Function
    builder: ir.IRBuilder
    pyapi: PythonAPI
    blam_stack: list[BLamBase] = field(default_factory=list)

    @contextmanager
    def bind_app(self, lam_expr: ase.SExpr, argval: Any):
        self.blam_stack.append(BLamValue(lam_expr, argval))
        try:
            yield
        finally:
            self.blam_stack.pop()


def determine_arity(root: ase.SExpr):
    node = root
    arity = 0
    first_non_lam = None
    while node:
        match node:
            case lam.Lam(body=ase.SExpr() as node):
                arity += 1
            case _:
                first_non_lam = node
                break
    return arity, node


# Adapted from numba/core/pythonapi/PythonApi
class PythonAPI:

    def __init__(self, builder):
        self.module = builder.basic_block.function.module
        self.builder = builder
        # A unique mapping of serialized objects in this module
        try:
            self.module.__serialized
        except AttributeError:
            self.module.__serialized = {}

        # Initialize types
        self.pyobj = ll_pyobject_ptr
        self.pyobjptr = self.pyobj.as_pointer()
        self.voidptr = ir.PointerType(ll_byte)
        self.long = ir.IntType(_ct.sizeof(_ct.c_long) * 8)
        self.ulong = self.long
        self.longlong = ir.IntType(_ct.sizeof(_ct.c_ulonglong) * 8)
        self.ulonglong = self.longlong
        self.double = ir.DoubleType()
        self.py_ssize_t = self.longlong
        self.cstring = ir.PointerType(ll_byte)
        self.py_hash_t = self.py_ssize_t

    def _get_function(self, fnty, name):

        return _get_or_insert_function(self.module, fnty, name)

    def _get_number_operator(self, name):
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_%s" % name)
        return fn

    def _call_number_operator(self, name, lhs, rhs, inplace=False):
        if inplace:
            name = "InPlace" + name
        fn = self._get_number_operator(name)
        return self.builder.call(fn, [lhs, rhs])

    def number_add(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Add", lhs, rhs, inplace=inplace)

    def number_subtract(self, lhs, rhs, inplace=False):
        return self._call_number_operator(
            "Subtract", lhs, rhs, inplace=inplace
        )

    def number_multiply(self, lhs, rhs, inplace=False):
        return self._call_number_operator(
            "Multiply", lhs, rhs, inplace=inplace
        )

    def number_truedivide(self, lhs, rhs, inplace=False):
        return self._call_number_operator(
            "TrueDivide", lhs, rhs, inplace=inplace
        )

    def number_floordivide(self, lhs, rhs, inplace=False):
        return self._call_number_operator(
            "FloorDivide", lhs, rhs, inplace=inplace
        )


def _get_or_insert_function(module, fnty, name):
    fn = module.globals.get(name, None)
    if fn is None:
        fn = ir.Function(module, fnty, name)
    return fn
