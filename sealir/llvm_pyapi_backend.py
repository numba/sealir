from __future__ import annotations

import ctypes as _ct
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import Any, Callable, Self

from llvmlite import binding as llvm
from llvmlite import ir

from sealir import ase, lam, rvsdg
from sealir.rvsdg import grammar as rg
from sealir.rvsdg import internal_prefix

ll_byte = ir.IntType(8)
ll_pyobject_ptr = ll_byte.as_pointer()
ll_iostate = ir.LiteralStructType([])  # empty struct


def llvm_codegen(root: ase.SExpr):
    """
    Emit LLVM using Python C-API.

    Warning:

    - This is for testing only.
    - Does NOT do proper memory management.
    - Does NOT do proper error handling.
    """
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    mod = ir.Module()

    # make function
    arity = determine_arity(root)
    bodynode = root.body
    assert arity >= 1
    actual_num_args = arity
    fnty = ir.FunctionType(
        ll_pyobject_ptr, [ll_pyobject_ptr] * actual_num_args
    )
    fn = ir.Function(mod, fnty, name="foo")

    # init entry block and builder
    builder = ir.IRBuilder(fn.append_basic_block())
    retval_slot = builder.alloca(ll_pyobject_ptr)
    builder.store(ll_pyobject_ptr(None), retval_slot)  # init retval to NULL
    bb_main = builder.append_basic_block()
    builder.branch(bb_main)
    builder.position_at_end(bb_main)

    ctx = CodegenCtx(
        llvm_module=mod,
        llvm_func=fn,
        builder=builder,
        pyapi=PythonAPI(builder),
        retval_slot=retval_slot,
        ports={},
    )
    # print(fn)
    # # Setup ports
    # argidx = 0
    # for k in bodynode.begin.ins.split():
    #     if k == internal_prefix("io"):
    #         ctx.ports[k] = IOState()
    #     else:
    #         ctx.ports[k] = ArgValue(argidx)
    #         argidx += 1
    # assert argidx == actual_num_args, (argidx, actual_num_args)

    memo = ase.traverse(bodynode, _codegen_loop, CodegenState(context=ctx))

    # builder.ret(builder.load(retval_slot))
    outports = dict(zip(bodynode.outs.split(), bodynode.ports, strict=True))
    retval = memo[outports[internal_prefix("ret")]].value
    builder.ret(retval)

    llvm_ir = str(mod)
    # print(llvm_ir)

    # llmod = llvm.parse_assembly(llvm_ir)
    # llvm.view_dot_graph(llvm.get_function_cfg(llmod.get_function("foo")), view=True)
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
    ctx = state.context
    builder = ctx.builder
    pyapi = ctx.pyapi

    def ensure_io(val):
        assert isinstance(val, IOState), val
        return val

    def debug_packs(name, packs):
        """
        Note: keep this for debugging
        """
        pyapi.printf(f"--- _EMIT_ERROR_HANDLING PACKS: {name}\n")
        for k, v in packs:
            if hasattr(v, "value"):
                pyapi.printf(f"    {k} = %p\n", v.value)
        return packs

    match expr:
        case rg.Func():
            raise TypeError

        case rg.RegionBegin(ins=ins, ports=ports):
            portvalues = []
            for p in ports:
                ctx.ports[p] = pv = yield p
                portvalues.append(pv)
            return PackedValues.make(*portvalues)

        case rg.RegionEnd(
            begin=rg.RegionBegin() as begin,
            outs=str(outs),
            ports=ports,
        ):
            yield begin
            portvalues = []
            for p in ports:
                ctx.ports[p] = pv = yield p
                portvalues.append(pv)
            return PackedValues.make(*portvalues)

        case rg.IO():
            return IOState()

        case rg.ArgRef(idx=int(idx), name=str(name)):
            return SSAValue(builder.function.args[idx])

        case rg.Unpack(val=source, idx=int(idx)):
            ports: PackedValues = yield source
            return ports[idx]

        case rg.PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
            ioval = ensure_io((yield io))
            lhsval = (yield lhs).value
            rhsval = (yield rhs).value
            match op:
                case "+":
                    res = ctx.pyapi.number_add(lhsval, rhsval)
                case "-":
                    res = ctx.pyapi.number_subtract(lhsval, rhsval)
                case "*":
                    res = ctx.pyapi.number_multiply(lhsval, rhsval)
                case "/":
                    res = ctx.pyapi.number_truedivide(lhsval, rhsval)
                case "//":
                    res = ctx.pyapi.number_floordivide(lhsval, rhsval)
                case "<" | ">" | "==" | "!=" | "in":
                    res = ctx.pyapi.object_richcompare(lhsval, rhsval, op)
                case _:
                    raise NotImplementedError(op)
            return PackedValues.make(ioval, SSAValue(res))

        case rg.PyInplaceBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
            ioval = ensure_io((yield io))
            lhsval = (yield lhs).value
            rhsval = (yield rhs).value
            match op:
                case "+":
                    res = ctx.pyapi.number_add(lhsval, rhsval, inplace=True)
                case "-":
                    res = ctx.pyapi.number_subtract(
                        lhsval, rhsval, inplace=True
                    )
                case "*":
                    res = ctx.pyapi.number_multiply(
                        lhsval, rhsval, inplace=True
                    )
                case "/":
                    res = ctx.pyapi.number_truedivide(
                        lhsval, rhsval, inplace=True
                    )
                case "//":
                    res = ctx.pyapi.number_floordivide(
                        lhsval, rhsval, inplace=True
                    )
                case _:
                    raise NotImplementedError(op)
            return PackedValues.make(ioval, SSAValue(res))

        case rg.PyUnaryOp(op=op, io=io, operand=operand):
            ioval = ensure_io((yield io))
            val = (yield operand).value
            match op:
                case "not":
                    retval = pyapi.bool_from_bool(pyapi.object_not(val))
                case "-":
                    retval = pyapi.number_negative(val)
                case _:
                    raise NotImplementedError(repr(op))
            return PackedValues.make(ioval, SSAValue(retval))

        case rg.DbgValue(value=value):
            val = yield value
            return val

        case rg.PyInt(int(ival)) | rg.PyBool(int(ival)):
            const = ir.Constant(pyapi.py_ssize_t, int(ival))
            return SSAValue(pyapi.long_from_ssize_t(const))

        case rg.PyStr(str(text)):
            module = builder.module
            encoded = bytearray(text.encode("utf-8") + b"\x00")
            byte_string = ir.Constant(
                ir.ArrayType(ll_byte, len(encoded)), encoded
            )
            unique_name = module.get_unique_name("const_string")
            gv = ir.GlobalVariable(module, byte_string.type, unique_name)
            gv.global_constant = True
            gv.initializer = byte_string
            gv.linkage = "internal"
            res = pyapi.string_from_string(builder.bitcast(gv, pyapi.cstring))
            return SSAValue(res)

        case rg.PyTuple(elems):
            elts = []
            for el in elems:
                elts.append((yield el).value)
            return SSAValue(pyapi.tuple_pack(elts))

        case rg.Undef(name=str(name)):
            return SSAValue(ll_pyobject_ptr(None))  # null pointer

        case rg.PyNone():
            return SSAValue(pyapi.make_none())

        case rg.IfElse(cond=cond, body=body, orelse=orelse, outs=outs):
            condval = (yield cond).value
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
                value_then = yield body
                builder.branch(bb_endif)
                bb_then_end = builder.basic_block
            # Else
            with builder.goto_block(bb_else):
                value_else = yield orelse
                builder.branch(bb_endif)
                bb_else_end = builder.basic_block
            # EndIf
            builder.position_at_end(bb_endif)
            assert len(value_then) == len(value_else)
            phis: list[LLVMValue] = []
            for left, right in zip(value_then, value_else, strict=True):
                if isinstance(left, IOState) or isinstance(right, IOState):
                    # handle iostate
                    assert isinstance(left, IOState) and isinstance(
                        right, IOState
                    )
                    phis.append(left)
                else:
                    # otherwise
                    left = left.value
                    right = right.value
                    assert left.type == right.type
                    phi = builder.phi(left.type)
                    phi.add_incoming(left, bb_then_end)
                    phi.add_incoming(right, bb_else_end)
                    phis.append(SSAValue(phi))
            return PackedValues.make(*phis)

        case rg.Loop(body=rg.RegionEnd() as body, outs=outs, loopvar=loopvar):
            # Note this is a tail loop.
            begin = body.begin
            loopentry_values = yield begin

            bb_before = builder.basic_block
            bb_loopbody = builder.append_basic_block("loopbody")
            bb_endloop = builder.append_basic_block("endloop")
            builder.branch(bb_loopbody)
            # loop body
            builder.position_at_end(bb_loopbody)
            # setup phi nodes for loopback variables

            phis = []
            fixups = {}
            for i, var in enumerate(loopentry_values.values):
                if isinstance(var, IOState):
                    # iostate
                    phis.append(var)
                else:
                    # otherwise
                    phi = builder.phi(var.value.type)
                    phi.add_incoming(var.value, bb_before)
                    fixups[i] = phi
                    phis.append(SSAValue(phi))

            packed_phis = PackedValues.make(*phis)

            # generate body
            loopctx = replace(ctx, ports={})
            loop_memo = {begin: packed_phis}
            memo = ase.traverse(
                body,
                _codegen_loop,
                CodegenState(context=loopctx),
                init_memo=loop_memo,
            )

            loopout_values = memo[body]
            cond_obj = loopout_values[outs.split().index(loopvar)].value

            # get loop condition
            loopcond = builder.icmp_unsigned(
                "!=", pyapi.int32(0), pyapi.object_istrue(cond_obj)
            )
            # fix up phis
            for i, phi in fixups.items():
                phi.add_incoming(loopout_values[i].value, builder.basic_block)
            # back jump
            builder.cbranch(loopcond, bb_loopbody, bb_endloop)
            # end loop
            builder.position_at_end(bb_endloop)
            # Returns the value from the loop body because this is a tail loop
            return loopout_values

        case rg.PyCall(func=func, io=io, args=args):
            ioval = ensure_io((yield io))
            callee = (yield func).value
            argvals = []
            for arg in args:
                argvals.append((yield arg).value)
            retval = pyapi.call_function_objargs(callee, argvals)
            return PackedValues.make(ioval, SSAValue(retval))

        case rg.PyLoadGlobal(io=io, name=str(varname)):
            freezeobj = __builtins__[varname]
            ptr = pyapi.py_ssize_t(id(freezeobj))
            obj = builder.inttoptr(
                ptr, ll_pyobject_ptr, name=f"global.{varname}"
            )
            return SSAValue(obj)

        case _:
            raise NotImplementedError(expr, type(expr))


@dataclass(frozen=True)
class CodegenCtx:
    llvm_module: ir.Module
    llvm_func: ir.Function
    builder: ir.IRBuilder
    pyapi: PythonAPI
    retval_slot: ir.Value

    ports: dict[str, LLVMValue]


def determine_arity(root: ase.SExpr) -> int:
    node = root
    match node:
        case rg.Func(args=rg.Args() as args):
            return len(args.arguments)
        case _:
            raise TypeError(node._head)


@dataclass(frozen=True)
class LLVMValue: ...


@dataclass(frozen=True)
class IOState(LLVMValue): ...


@dataclass(frozen=True)
class SSAValue(LLVMValue):
    value: ir.Value

    def __post_init__(self):
        assert isinstance(self.value, ir.Value)


@dataclass(frozen=True)
class PackedValues(LLVMValue):
    values: tuple[LLVMValue, ...]

    @classmethod
    def make(cls, *args) -> PackedValues:
        return cls(args)

    def __post_init__(self):
        for v in self.values:
            assert isinstance(v, (SSAValue, IOState)), type(v)

    def __getitem__(self, idx) -> SSAValue:
        return self.values[idx]

    def __len__(self) -> int:
        return len(self.values)


_EMIT_ERROR_HANDLING = True
"""
- control `handle_error`
"""


def handle_error(fn):
    if _EMIT_ERROR_HANDLING:

        @wraps(fn)
        def wrap(self, *args, **kwargs):
            res = fn(self, *args, **kwargs)
            builder: ir.IRBuilder = self.builder
            with builder.if_then(
                builder.icmp_unsigned("==", res, res.type(None))
            ):
                builder.ret(res)
            return res

        return wrap
    else:
        return fn


def handle_error_negone(fn):
    if _EMIT_ERROR_HANDLING:

        @wraps(fn)
        def wrap(self, *args, **kwargs):
            res = fn(self, *args, **kwargs)
            builder: ir.IRBuilder = self.builder
            with builder.if_then(
                builder.icmp_unsigned("==", res, res.type(-1))
            ):
                builder.ret(self.pyobj(None))
            return res

        return wrap
    else:
        return fn


# Adapted from numba/core/pythonapi/PythonApi
class PythonAPI:
    builder: ir.IRBuilder

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

    @handle_error
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

    @handle_error
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

    @handle_error
    def number_negative(self, operand):
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_Negative")
        return self.builder.call(fn, [operand])

    @handle_error
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
            outptr = _alloca_once_value(
                self.builder, ir.Constant(self.pyobj, None)
            )
            # If PySequence_Contains returns non-error value
            with self.builder.if_then(is_good):
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

    @handle_error_negone
    def object_istrue(self, obj):
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_IsTrue")
        return self.builder.call(fn, [obj])

    @handle_error_negone
    def object_not(self, obj):
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_Not")
        return self.builder.call(fn, [obj])

    @handle_error
    def tuple_pack(self, items):
        fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t], var_arg=True)
        fn = self._get_function(fnty, name="PyTuple_Pack")
        n = self.py_ssize_t(len(items))
        args = [n]
        args.extend(items)
        return self.builder.call(fn, args)

    @handle_error
    def bool_from_bool(self, bval):
        """
        Get a Python bool from a LLVM boolean.
        """
        longval = self.builder.zext(bval, self.long)
        return self.bool_from_long(longval)

    @handle_error
    def bool_from_long(self, ival):
        fnty = ir.FunctionType(self.pyobj, [self.long])
        fn = self._get_function(fnty, name="PyBool_FromLong")
        return self.builder.call(fn, [ival])

    @handle_error
    def call_function_objargs(self, callee, objargs):
        fnty = ir.FunctionType(self.pyobj, [self.pyobj], var_arg=True)
        fn = self._get_function(fnty, name="PyObject_CallFunctionObjArgs")
        args = [callee] + list(objargs)
        args.append(self.pyobj(None))
        return self.builder.call(fn, args)

    @handle_error
    def string_from_string(self, string):
        fnty = ir.FunctionType(self.pyobj, [self.cstring])
        fname = "PyUnicode_FromString"
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [string])

    def make_none(self):
        obj = self.borrow_none()
        self.incref(obj)
        return obj

    def borrow_none(self):
        return self.get_c_object("_Py_NoneStruct")

    @handle_error
    def get_c_object(self, name):
        """
        Get a Python object through its C-accessible *name*
        (e.g. "PyExc_ValueError").  The underlying variable must be
        a `PyObject *`, and the value of that pointer is returned.
        """
        # A LLVM global variable is implicitly a pointer to the declared
        # type, so fix up by using pyobj.pointee.
        return _get_c_value(
            self.builder, self.pyobj.pointee, name, dllimport=True
        )

    def incref(self, obj):
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name="Py_IncRef")
        self.builder.call(fn, [obj])

    def decref(self, obj):
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name="Py_DecRef")
        self.builder.call(fn, [obj])

    def dump(self, obj):
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name="_PyObject_Dump")
        self.builder.call(fn, [obj])

    def printf(self, fmt, *args):
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj], var_arg=True)

        module = self.module
        builder = self.builder
        encoded = bytearray(fmt.encode("utf-8") + b"\x00")
        byte_string = ir.Constant(ir.ArrayType(ll_byte, len(encoded)), encoded)
        unique_name = module.get_unique_name(f"_printf.{fmt.split()[0]}")
        gv = ir.GlobalVariable(module, byte_string.type, unique_name)
        gv.global_constant = True
        gv.initializer = byte_string
        gv.linkage = "internal"
        ptr = builder.bitcast(gv, self.cstring)

        fn = self._get_function(fnty, name="printf")
        self.builder.call(fn, [ptr, *args])


def _get_or_insert_function(module, fnty, name):
    # from numba's cgutils
    fn = module.globals.get(name, None)
    if fn is None:
        fn = ir.Function(module, fnty, name)
    return fn


def _get_c_value(builder, typ, name, dllimport=False):
    # from numba's context
    aot_mode = True
    module = builder.function.module
    try:
        gv = module.globals[name]
    except KeyError:
        unique_name = module.get_unique_name(name)
        gv = ir.GlobalVariable(module, typ, unique_name, addrspace=0)
        if dllimport and aot_mode and sys.platform == "win32":
            gv.storage_class = "dllimport"
    return gv


def _alloca_once_value(builder: ir.IRBuilder, value):
    with builder.goto_entry_block():
        slot = builder.alloca(value.type)
        builder.store(value, slot)

    builder.store(value, slot)
    return slot
