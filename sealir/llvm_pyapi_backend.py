from __future__ import annotations

import ctypes as _ct
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
        return cls(rt=rt, pyfunc=pyfunc)

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
            # handled at compile time
            packed = yield packed_expr
            retval = packed[idx]
            return retval
        case lam.Pack(args):
            # handled at compile time
            elems = []
            for arg in args:
                elems.append((yield arg))
            retval = tuple(elems)
            return retval
        case rvsdg.Scfg_If(
            test=cond,
            then=br_true,
            orelse=br_false,
        ):
            condval = yield cond
            # unpack pybool
            condbit = builder.icmp_unsigned(
                "!=", pyapi.int32(0), pyapi.object_istrue(condval)
            )

            bb_then = builder.append_basic_block("then")
            bb_else = builder.append_basic_block("else")
            bb_endif = builder.append_basic_block("endif")

            builder.cbranch(condbit, bb_then, bb_else)
            # Then
            with builder.goto_block(bb_then):
                value_then = yield br_true
                builder.branch(bb_endif)
            # Else
            with builder.goto_block(bb_else):
                value_else = yield br_false
                builder.branch(bb_endif)
            # EndIf
            builder.position_at_end(bb_endif)
            assert len(value_then) == len(value_else)
            phis = []
            for left, right in zip(value_then, value_else, strict=True):
                if isinstance(left, BLamIOArg) or isinstance(right, BLamIOArg):
                    # handle iostate
                    assert isinstance(left, BLamIOArg) and isinstance(
                        right, BLamIOArg
                    )
                    phis.append(left)
                else:
                    # otherwise
                    assert left.type == right.type
                    phi = builder.phi(left.type)
                    phi.add_incoming(left, bb_then)
                    phi.add_incoming(right, bb_else)
                    phis.append(phi)
            return tuple(phis)
        case rvsdg.Scfg_While(body=loopblk):
            bb_before = builder.basic_block
            bb_loopbody = builder.append_basic_block("loopbody")
            bb_endloop = builder.append_basic_block("endloop")
            builder.branch(bb_loopbody)
            # loop body
            builder.position_at_end(bb_loopbody)
            # setup phi nodes for loopback variables
            loopback_pack = ctx.blam_stack[-1]
            assert isinstance(loopback_pack, BLamValue)
            phis = []
            fixups = {}
            for i, var in enumerate(loopback_pack.val):
                if isinstance(var, BLamIOArg):
                    # iostate
                    phis.append(var)
                else:
                    # otherwise
                    phi = builder.phi(var.type)
                    phi.add_incoming(var, bb_before)
                    fixups[i] = phi
                    phis.append(phi)
            # replace the top of stack
            ctx.blam_stack[-1] = replace(loopback_pack, val=tuple(phis))
            # generate body
            loopout = yield loopblk
            # get loop condition
            loopcond = builder.icmp_unsigned(
                "!=", pyapi.int32(0), pyapi.object_istrue(loopout[0])
            )
            # fix up phis
            for i, phi in fixups.items():
                phi.add_incoming(loopout[i], builder.basic_block)
            # back jump
            builder.cbranch(loopcond, bb_loopbody, bb_endloop)
            # end loop
            builder.position_at_end(bb_endloop)
            return loopout

        case rvsdg.Py_Undef():
            return ll_pyobject_ptr(None)
        case rvsdg.Py_Int(int(ival)):
            const = ir.Constant(pyapi.py_ssize_t, int(ival))
            return pyapi.long_from_ssize_t(const)
        case rvsdg.Py_Tuple(args):
            elems = []
            for arg in args:
                elems.append((yield arg))
            return pyapi.tuple_pack(elems)
        case rvsdg.Py_UnaryOp(
            opname=str(opname),
            iostate=iostate,
            arg=val,
        ):
            ioval = yield iostate
            val = yield val
            match opname:
                case "not":
                    retval = pyapi.bool_from_bool(pyapi.object_not(val))
                case _:
                    raise NotImplementedError(opname)
            return ioval, retval
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
        case rvsdg.Py_InplaceBinOp(
            opname=str(op),
            iostate=iostate,
            lhs=lhs,
            rhs=rhs,
        ):
            ioval = ensure_io((yield iostate))
            lhsval = yield lhs
            rhsval = yield rhs
            match op:
                case "+":
                    res = ctx.pyapi.number_add(lhsval, rhsval, inplace=True)
                case "*":
                    res = ctx.pyapi.number_multiply(
                        lhsval, rhsval, inplace=True
                    )
                case _:
                    raise NotImplementedError(op)
            return ioval, res
        case rvsdg.Py_Compare(
            opname=str(op),
            iostate=iostate,
            lhs=lhs,
            rhs=rhs,
        ):
            ioval = ensure_io((yield iostate))
            lhsval = yield lhs
            rhsval = yield rhs
            match op:
                case "<" | ">" | "!=" | "in":
                    res = pyapi.object_richcompare(lhsval, rhsval, op)
                case _:
                    raise NotImplementedError(op)
            return ioval, res
        case rvsdg.Return(iostate=iostate, retval=retval):
            ensure_io((yield iostate))
            retval = yield retval
            builder.ret(retval)
        case rvsdg.Py_Call(
            iostate=iostate,
            callee=callee,
            args=args,
        ):
            ioval = ensure_io((yield iostate))
            callee = yield callee
            argvals = []
            for arg in args:
                argvals.append((yield arg))
            retval = pyapi.call_function_objargs(callee, argvals)
            return ioval, retval
        case rvsdg.Py_GlobalLoad(str(glbname)):
            freezeobj = __builtins__[glbname]
            ptr = pyapi.py_ssize_t(id(freezeobj))
            obj = builder.inttoptr(
                ptr, ll_pyobject_ptr, name=f"global.{glbname}"
            )
            return obj
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
        self.int32 = ir.IntType(32)
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

    def _long_from_native_int(self, ival, func_name, native_int_type, signed):
        fnty = ir.FunctionType(self.pyobj, [native_int_type])
        fn = self._get_function(fnty, name=func_name)
        return self.builder.call(fn, [ival])

    def long_from_long(self, ival):
        func_name = "PyLong_FromLong"
        fnty = ir.FunctionType(self.pyobj, [self.long])
        fn = self._get_function(fnty, name=func_name)
        return self.builder.call(fn, [ival])

    def long_from_ulong(self, ival):
        return self._long_from_native_int(
            ival, "PyLong_FromUnsignedLong", self.long, signed=False
        )

    def long_from_ssize_t(self, ival):
        return self._long_from_native_int(
            ival, "PyLong_FromSsize_t", self.py_ssize_t, signed=True
        )

    def long_from_longlong(self, ival):
        return self._long_from_native_int(
            ival, "PyLong_FromLongLong", self.longlong, signed=True
        )

    def long_from_ulonglong(self, ival):
        return self._long_from_native_int(
            ival, "PyLong_FromUnsignedLongLong", self.ulonglong, signed=False
        )

    def long_from_signed_int(self, ival):
        """
        Return a Python integer from any native integer value.
        """
        bits = ival.type.width
        if bits <= self.long.width:
            return self.long_from_long(self.builder.sext(ival, self.long))
        elif bits <= self.longlong.width:
            return self.long_from_longlong(
                self.builder.sext(ival, self.longlong)
            )
        else:
            raise OverflowError("integer too big (%d bits)" % (bits))

    def long_from_unsigned_int(self, ival):
        """
        Same as long_from_signed_int, but for unsigned values.
        """
        bits = ival.type.width
        if bits <= self.ulong.width:
            return self.long_from_ulong(self.builder.zext(ival, self.ulong))
        elif bits <= self.ulonglong.width:
            return self.long_from_ulonglong(
                self.builder.zext(ival, self.ulonglong)
            )
        else:
            raise OverflowError("integer too big (%d bits)" % (bits))

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

    def object_richcompare(self, lhs, rhs, opstr):
        """
        Refer to Python source Include/object.h for macros definition
        of the opid.
        """
        ops = ["<", "<=", "==", "!=", ">", ">="]
        if opstr in ops:
            opid = ops.index(opstr)
            fnty = ir.FunctionType(
                self.pyobj, [self.pyobj, self.pyobj, ir.IntType(32)]
            )
            fn = self._get_function(fnty, name="PyObject_RichCompare")
            lopid = self.int32(opid)
            return self.builder.call(fn, (lhs, rhs, lopid))
        elif opstr == "is":
            bitflag = self.builder.icmp_unsigned("==", lhs, rhs)
            return self.bool_from_bool(bitflag)
        elif opstr == "is not":
            bitflag = self.builder.icmp_unsigned("!=", lhs, rhs)
            return self.bool_from_bool(bitflag)
        elif opstr in ("in", "not in"):
            fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj])
            fn = self._get_function(fnty, name="PySequence_Contains")
            status = self.builder.call(fn, (rhs, lhs))
            negone = self.int32(-1)
            is_good = self.builder.icmp_unsigned("!=", status, negone)
            # Stack allocate output and initialize to Null
            outptr = cgutils.alloca_once_value(
                self.builder, Constant(self.pyobj, None)
            )
            # If PySequence_Contains returns non-error value
            with cgutils.if_likely(self.builder, is_good):
                if opstr == "not in":
                    status = self.builder.not_(status)
                # Store the status as a boolean object
                truncated = self.builder.trunc(status, ir.IntType(1))
                self.builder.store(self.bool_from_bool(truncated), outptr)

            return self.builder.load(outptr)
        else:
            raise NotImplementedError(
                "Unknown operator {op!r}".format(op=opstr)
            )

    def object_istrue(self, obj):
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_IsTrue")
        return self.builder.call(fn, [obj])

    def object_not(self, obj):
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_Not")
        return self.builder.call(fn, [obj])

    def tuple_pack(self, items):
        fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t], var_arg=True)
        fn = self._get_function(fnty, name="PyTuple_Pack")
        n = self.py_ssize_t(len(items))
        args = [n]
        args.extend(items)
        return self.builder.call(fn, args)

    def bool_from_bool(self, bval):
        """
        Get a Python bool from a LLVM boolean.
        """
        longval = self.builder.zext(bval, self.long)
        return self.bool_from_long(longval)

    def bool_from_long(self, ival):
        fnty = ir.FunctionType(self.pyobj, [self.long])
        fn = self._get_function(fnty, name="PyBool_FromLong")
        return self.builder.call(fn, [ival])

    def call_function_objargs(self, callee, objargs):
        fnty = ir.FunctionType(self.pyobj, [self.pyobj], var_arg=True)
        fn = self._get_function(fnty, name="PyObject_CallFunctionObjArgs")
        args = [callee] + list(objargs)
        args.append(self.pyobj(None))
        return self.builder.call(fn, args)


def _get_or_insert_function(module, fnty, name):
    fn = module.globals.get(name, None)
    if fn is None:
        fn = ir.Function(module, fnty, name)
    return fn
