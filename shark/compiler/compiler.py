from shark import ir

# noinspection PyUnresolvedReferences
from shark.dialects import (
    arith,
    linalg,
    math,
    memref,
    affine_ as affine
)
from shark.compiler.tracing.trace import trace


def mlir_trace(script_path):
    top_mlir_context = ir.Context()
    mlir_location = ir.Location.unknown(context=top_mlir_context)
    with top_mlir_context, mlir_location:
        mlir_module = ir.Module.create(loc=mlir_location)

    mlir_module = trace(
        script_path,
        top_mlir_context,
        mlir_location,
        mlir_module,
    )
    return mlir_module
