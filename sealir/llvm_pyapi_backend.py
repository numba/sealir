from __future__ import annotations

import ctypes as _ct
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Self

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

    ctx = CodegenCtx(llvm_module=mod, llvm_func=fn, builder=builder)
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

    def __call__(self, *args):
        return self.pyfunc(*args)


@dataclass(frozen=True)
class CodegenState(ase.TraverseState):
    context: CodegenCtx


def _codegen_loop(expr: ase.BasicSExpr, state: CodegenState):
    if False:
        (yield None)

    ctx = state.context
    builder = ctx.builder

    def ensure_io(val):
        assert isinstance(val, BLamIOArg), val

    match expr:
        case lam.Arg(int(debruijn)):
            sp = -debruijn - 1
            sv = ctx.blam_stack[sp]
            match sv:
                case BLamIOArg():
                    return sv
                case BLamArg(int(idx)):
                    return builder.function.args[idx]
                case _:
                    raise NotImplementedError(sv)
        case rvsdg.Return(iostate=iostate, retval=retval):
            ensure_io((yield iostate))
            retval = yield retval
            builder.ret(retval)

        case _:
            raise NotImplementedError(ase.as_tuple(expr, depth=2))


@dataclass(frozen=True)
class BLamValue:
    pass


@dataclass(frozen=True)
class BLamIOArg(BLamValue):
    pass


@dataclass(frozen=True)
class BLamArg(BLamValue):
    argidx: int


@dataclass(frozen=True)
class CodegenCtx:
    llvm_module: ir.Module
    llvm_func: ir.Function
    builder: ir.IRBuilder

    blam_stack: list[BLamValue] = field(default_factory=list)

    @contextmanager
    def bind_app(
        self,
    ):
        raise
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
