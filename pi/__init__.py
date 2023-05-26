from .mlir import Tensor as Tensor

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops
ops = ops

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir.ops import *

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops as _VF

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops as _C

_C._nn = _C
# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops as linalg

linalg = linalg

from .mlir.utils import dtype, empty, zeros, ones, rand, randn, TensorPlaceholder
