import ctypes
import ctypes.util
import os
import subprocess
import sys
from functools import cache
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any

import numpy as np
from llvmlite import binding as llvm  # type: ignore[import-untyped]

_empty_dict: MappingProxyType[str, Any] = MappingProxyType({})


@cache
def init_llvm():
    llvm.initialize()
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()


def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_mod(engine, mod):
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


def as_memref_descriptor(arr, ty):
    intptr_t = getattr(ctypes, f"c_int{8 * ctypes.sizeof(ctypes.c_void_p)}")
    N = arr.ndim

    ty_ptr = ctypes.POINTER(ty)

    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ty_ptr),
            ("aligned", ty_ptr),
            ("offset", intptr_t),
            ("sizes", intptr_t * N),
            ("strides", intptr_t * N),
        ]

    arg0 = ctypes.cast(arr.ctypes.data, ty_ptr)
    arg1 = arg0
    arg2 = intptr_t(0)
    arg3 = (intptr_t * N)(*arr.shape)
    arg4 = (intptr_t * N)(*arr.strides)
    return MemRefDescriptor(arg0, arg1, arg2, arg3, arg4)


class MLIRCompiler:
    def __init__(self, debug=False, print_cmds=False):
        self._debug = debug
        self._print_cmds = print_cmds

    def _run_cmd(self, cmd, in_mode, out_mode, src):
        assert in_mode in "tb"
        assert out_mode in "tb"
        with NamedTemporaryFile(mode=f"w{in_mode}") as src_file:
            src_file.write(src)
            src_file.flush()

            with NamedTemporaryFile(mode=f"r{out_mode}") as dst_file:
                full_cmd = *cmd, src_file.name, "-o", dst_file.name
                if self._print_cmds:
                    print(full_cmd)
                try:
                    subprocess.check_call(full_cmd)
                except subprocess.CalledProcessError as e:
                    print(e.output)
                    raise
                dst_file.flush()
                return dst_file.read()

    def to_llvm_dialect_with_omp_target(self, mlir_src):
        # mlir source to mlir llvm dialect source transform
        binary = ("mlir-opt",)
        if self._debug:
            dbg_cmd = (
                "--mlir-print-debuginfo",
                "--mlir-print-ir-after-all",
                "--debug-pass=Details",
            )
        else:
            dbg_cmd = ()

        options = (
            "--debugify-level=locations",
            "--use-unknown-locations=Enable",
            "--snapshot-op-locations",
            "--experimental-debug-variable-locations",
            "--dwarf-version=5",
            "--inline",
            "-affine-loop-normalize",
            "-affine-parallelize",
            "-affine-super-vectorize",
            "--affine-scalrep",
            "-lower-affine",
            "-convert-vector-to-scf",
            "-convert-linalg-to-loops",
            "-lower-affine",
            "-convert-scf-to-openmp",
            "-convert-scf-to-cf",
            "-cse",
            "-convert-openmp-to-llvm",
            "-convert-linalg-to-llvm",
            "-convert-vector-to-llvm",
            "-convert-math-to-llvm",
            "-expand-strided-metadata",
            "-lower-affine",
            "-finalize-memref-to-llvm",
            "-convert-func-to-llvm",
            "-convert-index-to-llvm",
            "-reconcile-unrealized-casts",
            "--llvm-request-c-wrappers",
        )
        full_cmd = binary + dbg_cmd + options
        return self._run_cmd(full_cmd, "t", "t", mlir_src)

    def mlir_translate_to_llvm_ir(self, mlir_src):
        # converts mlir source to llvm ir source
        binary = ("mlir-translate",)
        options = (
            "--mlir-print-local-scope",
            "--mir-debug-loc",
            "--use-unknown-locations=Enable",
            "-mlir-print-debuginfo=true",
            "--experimental-debug-variable-locations",
            "--mlir-to-llvmir",
        )
        full_cmd = binary + options
        return self._run_cmd(full_cmd, "t", "t", mlir_src)

    def llvm_ir_to_bitcode(self, llvmir_src):
        # converts llvm ir source to llvm bitcode
        binary = ("llvm-as",)
        full_cmd = binary
        return self._run_cmd(full_cmd, "t", "b", llvmir_src)  # txt to binary


class Compiler:
    def __init__(self, *, mlir_compiler_options=_empty_dict):
        init_llvm()
        # Need to load in OMP into process for the OMP backend.
        if sys.platform.startswith("linux"):
            omppath = ctypes.util.find_library("libgomp.so")
        elif sys.platform.startswith("darwin"):
            omppath = ctypes.util.find_library("iomp5")
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        ctypes.CDLL(omppath, mode=os.RTLD_NOW)

        self.ee = create_execution_engine()
        self._mlir_compiler_options = mlir_compiler_options
        self.output_logs = []

    def run_backend(self, mlir_src, symbol):
        mlir_compiler = MLIRCompiler(**self._mlir_compiler_options)
        mlir_omp = mlir_compiler.to_llvm_dialect_with_omp_target(mlir_src)

        llvm_ir = mlir_compiler.mlir_translate_to_llvm_ir(mlir_omp)
        fdata = mlir_compiler.llvm_ir_to_bitcode(llvm_ir)
        mod = llvm.parse_bitcode(fdata)
        mod = compile_mod(self.ee, mod)

        self.output_logs.append(
            dict(
                mlir_optimized=mlir_omp,
            )
        )

        address = self.ee.get_function_address(symbol)
        assert (
            address
        ), "Lookup for compiled function address is returning NULL."
        return address


def call_ufunc(addr: int, *, args, out):
    all_args = *args, out
    args_as_memref = [
        as_memref_descriptor(x, ctypes.c_double) for x in all_args
    ]
    prototype = ctypes.CFUNCTYPE(
        None, *[ctypes.POINTER(type(x)) for x in args_as_memref]
    )
    cfunc = prototype(addr)
    cfunc(*[ctypes.byref(x) for x in args_as_memref])
    return out
