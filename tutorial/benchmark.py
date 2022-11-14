"""This file contains benchmarks for sparse tensors. In particular, it
contains benchmarks for both mlir sparse tensor dialect and numpy so that they
can be compared against each other.
"""
import ctypes
import os

import numpy as np

from shark import ir
from shark.dialects import func, arith, memref, scf
from shark.execution_engine import ExecutionEngine
from shark.runtime import get_ranked_memref_descriptor
from refbackend import assert_arg_type_is_supported
from config import DEBUG


def emit_timer_func() -> func.FuncOp:
    """Returns the declaration of nanoTime function. If nanoTime function is
    used, the `MLIR_RUNNER_UTILS` and `MLIR_C_RUNNER_UTILS` must be included.
    """
    i64_type = ir.IntegerType.get_signless(64)
    nanoTime = func.FuncOp("nanoTime", ([], [i64_type]), visibility="private")
    nanoTime.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    return nanoTime


# op.operation.print(print_generic_op_form=True)


def get_kernel_func_from_module(module: ir.Module, kernel_name: str) -> func.FuncOp:
    """Takes an mlir module object and extracts the function object out of it.
    This function only works for a module with one region, one block, and one
    operation.
    """
    assert (
        len(module.operation.regions) == 1
    ), "Expected kernel module to have only one region"
    assert (
        len(module.operation.regions[0].blocks) == 1
    ), "Expected kernel module to have only one block"
    for op in module.operation.regions[0].blocks[0].operations:
        func_name = str(op.operation.attributes["sym_name"]).replace('"', "")
        if func_name == kernel_name:
            return op

    raise Exception(f"couldn't find {kernel_name}")


def emit_benchmark_wrapped_main_func(kernel_func, timer_func, num_iterations=100):
    """Takes a function and a timer function, both represented as FuncOp
    objects, and returns a new function. This new function wraps the call to
    the original function between calls to the timer_func and this wrapping
    in turn is executed inside a loop. The loop is executed
    len(kernel_func.type.results) times. This function can be used to
    create a "time measuring" variant of a function.
    """
    i64_type = ir.IntegerType.get_signless(64)
    memref_of_i64_type = ir.MemRefType.get([num_iterations], i64_type)
    wrapped_func = func.FuncOp(
        # Same signature and an extra buffer of indices to save timings.
        "main",
        (kernel_func.arguments.types + [memref_of_i64_type], []),
        visibility="public",
    )
    wrapped_func.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

    with ir.InsertionPoint(wrapped_func.add_entry_block()):
        timer_buffer = wrapped_func.arguments[-1]
        zero = arith.ConstantOp.create_index(0)
        one = arith.ConstantOp.create_index(1)
        num_iterations = arith.ConstantOp.create_index(num_iterations)
        loop = scf.ForOp(zero, num_iterations, one, [])
        with ir.InsertionPoint(loop.body):
            start = func.CallOp(timer_func, [])
            call = func.CallOp(
                kernel_func,
                list(wrapped_func.arguments[:-1]),
            )
            end = func.CallOp(timer_func, [])
            time_taken = arith.SubIOp(end, start)
            memref.StoreOp(time_taken, timer_buffer, [loop.induction_variable])
            scf.YieldOp([])
        func.ReturnOp(loop)

    return wrapped_func


class Benchmark:
    """Benchmark for mlir sparse matrix multiplication. Because its an
    MLIR benchmark we need to return both a `compiler` function and a `runner`
    function.
    """

    def __init__(self, c_runner_utils_fp, runner_utils_fp, num_iterations=100):
        assert os.path.exists(c_runner_utils_fp), (
            f"{c_runner_utils_fp} does not exist."
            f" Please pass a valid value for"
            f" MLIR_C_RUNNER_UTILS environment variable."
        )
        assert os.path.exists(runner_utils_fp), (
            f"{runner_utils_fp} does not exist."
            f" Please pass a valid value for MLIR_RUNNER_UTILS"
            f" environment variable."
        )
        self.c_runner_utils = c_runner_utils_fp
        self.runner_utils = runner_utils_fp
        self.num_iterations = num_iterations

    def wrap(self, module, kernel_name="main"):
        with ir.Context(), ir.Location.unknown():
            kernel_func = get_kernel_func_from_module(module, kernel_name)
            timer_func = emit_timer_func()
            if DEBUG:
                print(timer_func)
            wrapped_func = emit_benchmark_wrapped_main_func(
                kernel_func, timer_func, num_iterations=self.num_iterations
            )
            if DEBUG:
                print(wrapped_func)
            main_module_with_benchmark = ir.Module.parse(
                str(timer_func) + str(wrapped_func) + str(kernel_func)
            )

        return main_module_with_benchmark

    def run(self, main_module_with_benchmark, compiled_program_args: list):
        engine = ExecutionEngine(
            main_module_with_benchmark,
            3,
            shared_libs=[self.c_runner_utils, self.runner_utils],
        )
        ffi_args = []
        for arg in compiled_program_args:
            assert_arg_type_is_supported(arg.dtype)
            ffi_args.append(
                ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
            )
        np_timers_ns = np.zeros(self.num_iterations, dtype=np.int64)
        ffi_args.append(
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(np_timers_ns)))
        )
        engine.invoke("main", *ffi_args)
        return np_timers_ns
