import argparse

import numpy as np

import benchmark
from refbackend import (
    RefBackendLinalgOnTensorsBackend,
    BUFFERIZATION_PIPELINE,
    LOWER_LLVM_PIPELINE,
)
from shark.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    RankedTensorType,
    F64Type,
)
from shark.compiler.config import MLIR_C_RUNNER_UTILS, MLIR_RUNNER_UTILS
from shark.compiler.utils import run_pipeline_with_repro_report
from shark.dialects import func, linalg

M = 32
N = 32
K = 32


def build_matmul():
    with Context(), Location.unknown():
        module = Module.create()
        f64 = F64Type.get()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                RankedTensorType.get((M, N), f64),
                RankedTensorType.get((N, K), f64),
            )
            def matmul(lhs, rhs):
                out = linalg.InitTensorOp([M, K], f64)
                return linalg.matmul(lhs, rhs, outs=[out])

    return module


def lower_matmul(module, tile_size=2, munge=False):
    run_pipeline_with_repro_report(
        module,
        ",".join(BUFFERIZATION_PIPELINE(munge)),
    )
    if DEBUG:
        module.operation.print(print_generic_op_form=True)
    run_pipeline_with_repro_report(
        module,
        ",".join(
            list(
                filter(
                    None,
                    [
                        "func.func(convert-linalg-to-affine-loops)",
                        # f"func.func(affine-loop-unroll{{unroll-factor={unroll_factor} unroll-up-to-factor=1}})"
                        f"func.func(affine-loop-tile{{tile-size={tile_size}}})"
                        if tile_size > 0
                        else None,
                    ],
                )
            )
        ),
    )
    if DEBUG:
        module.operation.print(print_generic_op_form=True)
    run_pipeline_with_repro_report(
        module,
        ",".join(LOWER_LLVM_PIPELINE),
    )
    if DEBUG:
        module.operation.print(print_generic_op_form=True)
    return module


def run_matmul(module):
    invoker = RefBackendLinalgOnTensorsBackend.load(module)
    mat1 = np.round(np.random.uniform(low=0.0, high=5, size=(M, N)))
    mat2 = np.round(np.random.uniform(low=0.0, high=5, size=(N, K)))
    res = np.clip(invoker.matmul(mat1, mat2), 0, 1000)
    print("linalg result:\n")
    print(res)
    print("\nNumPy result:\n")
    print(mat1 @ mat2)


def test_matmul():
    module = build_matmul()
    if DEBUG:
        print(module)
    module = lower_matmul(module, munge=True)
    if DEBUG:
        print(module)
    run_matmul(module)


def reject_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return data[s < m]


def benchmark_matmul(tile_size):
    module = build_matmul()
    bench = benchmark.Benchmark(
        c_runner_utils_fp=MLIR_C_RUNNER_UTILS, runner_utils_fp=MLIR_RUNNER_UTILS
    )
    main_module_with_benchmark = bench.wrap(module, "matmul")
    lowered_module = lower_matmul(main_module_with_benchmark, tile_size)
    mat1 = np.round(np.random.uniform(low=0.0, high=5, size=(M, N)))
    mat2 = np.round(np.random.uniform(low=0.0, high=5, size=(N, K)))
    times = reject_outliers(bench.run(lowered_module, [mat1, mat2]))
    # don't count the first/JIT warmup run
    return np.mean(times[10:]), np.std(times[10:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark matmul with tiling")
    parser.add_argument(
        "-t",
        "--tile-size",
        default=2,
        type=int,
        help="Tile size",
    )
    parser.add_argument(
        "--tut",
        choices=[
            "benchmark",
            "test",
        ],
        default="test",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()

    if args.debug:
        benchmark.DEBUG = DEBUG = True

    if args.tut == "test":
        test_matmul()
    elif args.tut == "benchmark":
        tile_size = args.tile_size
        mean, var = benchmark_matmul(tile_size)
        print(f"For tile-size {tile_size}, runtime {mean:.2f}±{var:.2f} ns")
