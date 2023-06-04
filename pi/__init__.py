from .mlir import Tensor as Tensor

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops

ops = ops
ops.aten = ops

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

from .mlir.utils import (
    dtype,
    empty,
    zeros,
    ones,
    rand,
    randn,
    tensor,
    zeros_like,
    empty_like,
    TensorPlaceholder,
    memory_format,
    layout,
)

bfloat16 = dtype.bfloat16
bool = dtype.bool
complex32 = dtype.complex32
complex64 = dtype.complex64
half = float16 = dtype.float16
float = float32 = dtype.float32
double = float64 = dtype.float64
int8 = dtype.int8
int16 = dtype.int16
int32 = dtype.int32
long = int64 = dtype.int64
qint8 = dtype.qint8
quint8 = dtype.quint8
uint8 = dtype.uint8

strided: layout = layout.strided
sparse_coo: layout = layout.sparse_coo
sparse_csr: layout = layout.sparse_csr
sparse_csc: layout = layout.sparse_csc
sparse_bsr: layout = layout.sparse_bsr
sparse_bsc: layout = layout.sparse_bsc
_mkldnn: layout = layout._mkldnn

contiguous_format = memory_format.contiguous_format
preserve_format = memory_format.preserve_format
channels_last = memory_format.channels_last
channels_last_3d = memory_format.channels_last_3d

from . import nn
