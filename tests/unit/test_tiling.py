from pi.compiler.compiler import mlir_trace
from pi.compiler.utils import run_pipeline_with_repro_report

mlir_module = mlir_trace("simple_kernels.py")

tile_size = 2

run_pipeline_with_repro_report(
    mlir_module,
    ",".join(
        list(
            filter(
                None,
                [
                    f"func.func(cse)",
                    f"func.func(loop-invariant-code-motion)",
                    f"func.func(sccp)",
                    f"func.func(affine-loop-tile{{tile-size={tile_size}}})"
                    if tile_size > 0
                    else None,
                ],
            )
        )
    ),
)

print(mlir_module)
