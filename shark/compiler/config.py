import os
import platform
from pathlib import Path

shlib_ext = "dylib" if platform.system() == "Darwin" else "so"

MLIR_C_RUNNER_UTILS = os.getenv(
    "MLIR_C_RUNNER_UTILS",
    default=str(
        (
            Path(__file__).parent.parent.parent
            / f"llvm_install/lib/libmlir_c_runner_utils.{shlib_ext}"
        ).absolute()
    ),
)
assert os.path.exists(MLIR_C_RUNNER_UTILS), "C runner utils not found"
MLIR_RUNNER_UTILS = os.getenv(
    "MLIR_RUNNER_UTILS",
    str(
        (
            Path(__file__).parent.parent.parent
            / f"llvm_install/lib/libmlir_runner_utils.{shlib_ext}"
        ).absolute()
    ),
)
assert os.path.exists(MLIR_RUNNER_UTILS), "Runner utils not found"

DEBUG = False
