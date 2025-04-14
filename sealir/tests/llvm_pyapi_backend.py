from __future__ import annotations

import ctypes as _ct
import sys
from collections import ChainMap
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from types import MappingProxyType
from typing import Any, Callable, Mapping, Self

from sealir import ase, lam, rvsdg
from sealir.rvsdg import grammar as rg
from sealir.rvsdg import internal_prefix

import mlir.dialects.arith as arith
import mlir.dialects.func as func
import mlir.dialects.math as math

import mlir.passmanager as passmanager
import mlir.execution_engine as execution_engine
import mlir.runtime as runtime
import mlir.rewrite as rewrite
import mlir.extras as extras
import mlir._mlir_libs as mlir_libs
import mlir.ir as ir
import mlir.dialects as dialects

import ctypes

def llvm_codegen(root: ase.SExpr):
    """
    Emit LLVM using MLIR Python C-API.

    Warning:

    - This is for testing only.
    - Does NOT do proper memory management.
    - Does NOT do proper error handling.
    """
    context = ir.Context()
    loc = ir.Location.unknown(context=context)
    module = ir.Module.create(loc=loc)

    # f32 = ir.F32Type.get(context=context)
    f64 = ir.F64Type.get(context=context)
    # i32 = ir.IntegerType.get_signless(32, context=context)
    # i64 = ir.IntegerType.get_signless(64, context=context)

    module_body = ir.InsertionPoint(module.body)
    input_types = tuple([f64] * determine_arity(root))
    output_types = (f64,)

    with context, loc, module_body:
        fun = func.FuncOp("func", (input_types, output_types))
        fun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        entry = fun.add_entry_block()

    function_entry = ir.InsertionPoint(entry)

    memo = ase.traverse(root.body, _codegen_loop, CodegenState(context=context, loc=loc, fun=fun, function_entry=function_entry))

    module.dump()
    pass_man = passmanager.PassManager(context=context)
    pass_man.add("convert-func-to-llvm")
    pass_man.enable_verifier(True)
    pass_man.run(module.operation)
    module.dump()

    return JitCallable.from_pointer(module, input_types, output_types)


def get_exec_ptr(mlir_ty, val):
    if isinstance(mlir_ty, ir.F64Type):
        return ctypes.pointer(ctypes.c_double(val)) 

@dataclass(frozen=True)
class JitCallable:
    jit_func: Callable

    @classmethod
    def from_pointer(
        cls, jit_module, input_types, output_types
    ) -> Self:
        engine = execution_engine.ExecutionEngine(jit_module)

        assert len(output_types) == 1, "Execution of functions with output arguments > 1 not supported" 
        res_ptr = get_exec_ptr(output_types[0], 0.0)

        def jit_func(*input_args):
            assert len(input_args) == len(input_types)
            for arg, arg_ty in zip(input_args, input_types):
                # assert isinstance(arg, arg_ty)
                # TODO: Assert types here
                pass              
            input_exec_ptrs = [get_exec_ptr(ty, val) for ty, val in zip(input_types, input_args)]
            engine.invoke("func", *input_exec_ptrs, res_ptr)

            return res_ptr.contents.value

        return cls(jit_func)

    def __call__(self, *args: Any) -> Any:
        return self.jit_func(*args)


@dataclass(frozen=True)
class CodegenState(ase.TraverseState):
    context: int
    loc: int
    fun: int
    function_entry: int

def _codegen_loop(expr: ase.BasicSExpr, state: CodegenState):
    context = state.context

    def ensure_io(val):
        assert isinstance(val, IOState), val
        return val

    def _handle_binop(op: str, lhsval, rhsval):
        with state.function_entry, state.loc:
            match op:
                case "+":
                    res =arith.addf(lhsval, rhsval)
                case "-":
                    res = arith.subf(lhsval, rhsval)
                case "*":
                    res = arith.mulf(lhsval, rhsval)
                case "/":
                    res = arith.divf(lhsval, rhsval)
                case "//":
                    res = math.trunc(arith.divf(lhsval, rhsval))
                case "**":
                    res = math.powf(lhsval, rhsval)

                # compare
                case "<" | ">" | "==" | "!=" | "in":
                    res = ctx.pyapi.object_richcompare(lhsval, rhsval, op)
                case _:
                    raise NotImplementedError(op)

        return res

    @contextmanager
    def push(region_args):
        state.context.region_args.append(tuple(region_args))
        try:
            yield
        finally:
            state.context.region_args.pop()

    def get_region_args():
        return state.context.region_args[-1]

    match expr:
        case rg.Func(args=args, body=body):
            # Function prologue
            bb_main = builder.append_basic_block()
            builder.branch(bb_main)
            builder.position_at_end(bb_main)

            names = {
                argspec.name: builder.function.args[i]
                for i, argspec in enumerate(args.arguments)
            }
            argvalues = []
            for k in body.begin.inports:
                if k == internal_prefix("io"):
                    v = IOState()
                else:
                    v = SSAValue(names[k])
                argvalues.append(v)

            with push(argvalues):
                outs = yield body

            portnames = [p.name for p in body.ports]
            retval = outs[portnames.index(internal_prefix("ret"))]
            builder.ret(retval.value)

        case rg.RegionBegin(inports=ins):
            portvalues = []
            for i, k in enumerate(ins):
                pv = get_region_args()[i]
                portvalues.append(pv)
            return PackedValues.make(*portvalues)

        case rg.RegionEnd(
            begin=rg.RegionBegin() as begin,
            ports=ports,
        ):
            yield begin
            portvalues = []
            for p in ports:
                pv = yield p.value
                portvalues.append(pv)
            return PackedValues.make(*portvalues)

        case rg.IO():
            return IOState()

        case rg.ArgRef(idx=int(idx), name=str(name)):
            return SSAValue(state.fun.arguments[idx])

        case rg.Unpack(val=source, idx=int(idx)):
            ports: PackedValues = yield source
            return ports[idx]

        case rg.PyBinOpPure(op=op, lhs=lhs, rhs=rhs):
            lhsval = (yield lhs).value
            rhsval = (yield rhs).value
            res = _handle_binop(op, lhsval, rhsval)
            return SSAValue(res)

        case rg.PyBinOp(op=op, io=io, lhs=lhs, rhs=rhs):
            ioval = ensure_io((yield io))
            lhsval = (yield lhs).value
            rhsval = (yield rhs).value
            res = _handle_binop(op, lhsval, rhsval)
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
            global func
            with state.function_entry, state.loc:
                func.ReturnOp([val.value])
            return val

        case rg.PyInt(int(ival)) | rg.PyBool(int(ival)):
            const = llvm_ir.Constant(pyapi.py_ssize_t, int(ival))
            return SSAValue(pyapi.long_from_ssize_t(const))

        case rg.PyFloat(float(fval)):
            with state.loc, state.context, state.function_entry:
                const = arith.constant(ir.F64Type.get(), fval)
            return SSAValue(const)

        case rg.PyStr(str(text)):
            module = builder.module
            encoded = bytearray(text.encode("utf-8") + b"\x00")
            byte_string = llvm_ir.Constant(
                llvm_ir.ArrayType(ll_byte, len(encoded)), encoded
            )
            unique_name = module.get_unique_name("const_string")
            gv = llvm_ir.GlobalVariable(module, byte_string.type, unique_name)
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

        case rg.IfElse(cond=cond, body=body, orelse=orelse, operands=operands):
            condval = (yield cond).value

            # process operands
            ops = []
            for op in operands:
                ops.append((yield op))

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
                with push(ops):
                    value_then = yield body
                builder.branch(bb_endif)
                bb_then_end = builder.basic_block
            # Else
            with builder.goto_block(bb_else):
                with push(ops):
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

        case rg.Loop(body=rg.RegionEnd() as body, operands=operands):
            # process operands
            ops = []
            for op in operands:
                ops.append((yield op))

            # Note this is a tail loop.
            begin = body.begin

            with push(ops):
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
            loopctx = replace(ctx, region_args=[])
            loop_memo = {begin: packed_phis}
            memo = ase.traverse(
                body,
                _codegen_loop,
                CodegenState(context=loopctx),
                init_memo=loop_memo,
            )

            loopout_values = list(memo[body])
            portnames = [p.name for p in body.ports]
            cond_obj = loopout_values.pop(0).value

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

        case rg.PyCallPure(func=func, args=args):
            callee = (yield func).value
            argvals = []
            for arg in args:
                argvals.append((yield arg).value)
            retval = pyapi.call_function_objargs(callee, argvals)
            return SSAValue(retval)

        case rg.PyLoadGlobal(io=io, name=str(varname)):
            freezeobj = ctx.global_ns[varname]
            ptr = pyapi.py_ssize_t(id(freezeobj))
            obj = builder.inttoptr(
                ptr, ll_pyobject_ptr, name=f"global.{varname}"
            )
            return SSAValue(obj)

        case rg.PyAttr(io=io, value=value, attrname=str(attrname)):
            obj = pyapi.object_getattr_string((yield value), attrname)
            return SSAValue(obj)

        case _:
            raise NotImplementedError(expr, type(expr))

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
    value: int


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

