import sys
import os

old_flags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
from pi import _pi_mlir

from torch_mlir import ir

torch_mlir_pybind11_module_local_id = next(d for d in dir(ir.Value) if "pybind" in d)

assert _pi_mlir.get_pybind11_module_local_id() == torch_mlir_pybind11_module_local_id
