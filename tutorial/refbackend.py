# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import ctypes

import numpy as np

__all__ = [
    "RefBackendLinalgOnTensorsBackend",
]

from shark.ir import Module
from shark.compiler.config import MLIR_C_RUNNER_UTILS, MLIR_RUNNER_UTILS
from shark.compiler.utils import run_pipeline_with_repro_report

from shark.execution_engine import ExecutionEngine

from shark.runtime import (
    UnrankedMemRefDescriptor,
    unranked_memref_to_numpy,
    get_unranked_memref_descriptor,
)


def assert_arg_type_is_supported(ty):
    SUPPORTED = [
        np.float32,
        np.float64,
        np.uint8,
        np.int8,
        np.int32,
        np.int64,
        np.bool_,
    ]
    assert (
        ty in SUPPORTED
    ), f"Only numpy arrays with dtypes in {SUPPORTED} are supported"


memref_type_to_np_dtype = {
    "mrf32": np.float32,
    "mrf64": np.float64,
    "mri1": np.bool_,
    "mri8": np.int8,
    "mri32": np.int32,
    "mri64": np.int64,
}
elemental_type_to_ctype = {
    "i1": ctypes.c_bool,
    "i8": ctypes.c_byte,
    "i64": ctypes.c_int,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}

CONSUME_RETURN_FUNC_PREFIX = "refbackend_consume_func_return_"


def get_return_funcs(module):
    return_prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
    return_funcs = []
    with module.context:
        for func in module.body:
            # Returns strings of the form `"refbackend.."` so `"` is deleted.
            func_name = str(func.attributes["sym_name"]).replace('"', "")
            if func_name[:return_prefix_len] == CONSUME_RETURN_FUNC_PREFIX:
                return_funcs.append(func_name)

    return return_funcs


def get_ctype_func(func_name):
    return_prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
    ret_types = func_name[return_prefix_len:].split("_")
    ctypes_arg = [None]
    for type in ret_types:
        if type in elemental_type_to_ctype:
            ctypes_arg.append(elemental_type_to_ctype[type])
        elif type in memref_type_to_np_dtype:
            ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
        else:
            assert False, f"Not supported type: {type}"

    return ctypes.CFUNCTYPE(*ctypes_arg), ret_types


class RefBackendInvoker:
    def __init__(self, module):
        shared_libs = [
            MLIR_C_RUNNER_UTILS,
            MLIR_RUNNER_UTILS,
        ]
        self.ee = ExecutionEngine(module, shared_libs=shared_libs)
        self.result = None

        return_funcs = get_return_funcs(module)

        for ret_func in return_funcs:
            ctype_wrapper, ret_types = get_ctype_func(ret_func)

            def consume_return_funcs(*args):
                self.result = tuple(
                    [
                        arg
                        if type in elemental_type_to_ctype
                        else unranked_memref_to_numpy(
                            arg, memref_type_to_np_dtype[type]
                        )
                        for arg, type in zip(args, ret_types)
                    ]
                )
                if len(self.result) == 1:
                    self.result = self.result[0]

            self.ee.register_runtime(ret_func, ctype_wrapper(consume_return_funcs))

    def __getattr__(self, function_name: str):
        def invoke(*args):
            ffi_args = []
            for arg in args:
                assert_arg_type_is_supported(arg.dtype)
                ffi_args.append(
                    ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(arg)))
                )

            self.ee.invoke(function_name, *ffi_args)
            result = self.result
            assert result is not None, "Invocation didn't produce a result"
            self.result = None
            return result

        return invoke


BUFFERIZATION_PIPELINE = lambda munge=False: list(
    filter(
        None,
        [
            # "func.func(refback-generalize-tensor-pad)",
            # Bufferize.
            "func.func(linalg-init-tensor-to-alloc-tensor)",
            "func.func(scf-bufferize)",
            # "func.func(tm-tensor-bufferize)",
            # "func.func(empty-tensor-to-alloc-tensor)",
            "func.func(linalg-bufferize)",
            "func-bufferize",
            "arith-bufferize",
            "func.func(tensor-bufferize)",
            "func.func(finalizing-bufferize)",
            "refback-munge-calling-conventions" if munge else None,
            "func.func(refback-munge-memref-copy)" if munge else None,
            # Insert global variable and instruction sequence for getting the next
            # global seed used in stateful rng.
            # "refback-insert-rng-globals",
        ],
    )
)

LOWER_LLVM_PIPELINE = [
    # Lower to LLVM
    # "func.func(tm-tensor-to-loops)",
    "func.func(convert-linalg-to-loops)",
    "func.func(lower-affine)",
    "convert-scf-to-cf",
    # "func.func(refback-expand-ops-for-llvm)",
    "func.func(arith-expand)",
    "func.func(convert-math-to-llvm)",
    # Handle some complex mlir::math ops (e.g. atan2)
    "convert-math-to-libm",
    "convert-linalg-to-llvm",
    "convert-memref-to-llvm",
    "func.func(convert-arith-to-llvm)",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
]


class RefBackendLinalgOnTensorsBackend:
    """Main entry-point for the reference backend."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def compile(imported_module: Module):
        run_pipeline_with_repro_report(
            imported_module,
            ",".join(BUFFERIZATION_PIPELINE(munge=True) + LOWER_LLVM_PIPELINE),
            "Lowering Linalg-on-Tensors IR to LLVM with RefBackend",
        )
        return imported_module

    @staticmethod
    def load(module) -> RefBackendInvoker:
        """Loads a compiled artifact into the runtime."""
        return RefBackendInvoker(module)
