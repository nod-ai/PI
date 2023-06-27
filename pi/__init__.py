import contextlib

from .mlir import Tensor

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir.constants import *

# note this import needs to be above the one below it so that e.g. ops.zeros is replaced by util.zeros
# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir.ops import *
from .mlir.utils import (
    LongTensor,
    TensorPlaceholder,
    Torch_BoolValue,
    Torch_IntValue,
    Torch_FloatValue,
    dtype,
    empty,
    layout,
    memory_format,
    ones,
    rand,
    randn,
    tensor,
    zeros,
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
int = int32 = dtype.int32
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

# hacks to comply with torch api


ops.aten = ops
ops.prim = ops
ops.prims = ops

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops as _VF

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops as special

# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops as _C

_C._nn = _C
# noinspection PyUnresolvedReferences
from .mlir._mlir_libs._pi_mlir import ops as linalg

from . import nn


def manual_seed(x):
    pass


class backends:
    class mkldnn:
        @contextlib.contextmanager
        @staticmethod
        def flags(*args, **kwargs):
            yield


FloatTensor = tensor

import builtins

builtin_int = builtins.int
builtin_bool = builtins.bool
builtin_float = builtins.float


class pi_bool_meta(type):
    def __instancecheck__(self, other):
        return isinstance(other, builtin_bool)

    def __call__(self, x, base=None):
        if isinstance(x, Tensor):
            return ops.Bool(x)
        elif isinstance(x, Torch_BoolValue):
            try:
                return builtin_bool(x)
            except:
                return x
        else:
            return builtin_bool(x)


class pi_bool(builtin_int, metaclass=pi_bool_meta):
    pass


pi_bool.__name__ = builtin_bool.__name__
pi_bool.__qualname__ = builtin_bool.__qualname__


class pi_int_meta(type):
    def __instancecheck__(self, other):
        return isinstance(other, builtin_int)

    def __call__(self, x, base=None):
        if isinstance(x, Tensor):
            return ops.Int(x)
        elif isinstance(x, Torch_IntValue):
            try:
                return builtin_int(x)
            except:
                return x
        else:
            if base is not None:
                return builtin_int(x, base)
            else:
                return builtin_int(x)


class pi_int(builtin_int, metaclass=pi_int_meta):
    pass


pi_int.__name__ = builtin_int.__name__
pi_int.__qualname__ = builtin_int.__qualname__


class pi_float_meta(type):
    def __instancecheck__(self, other):
        return isinstance(other, builtin_float)

    def __call__(self, x):
        if isinstance(x, Tensor):
            return ops.Float(x)
        elif isinstance(x, Torch_FloatValue):
            try:
                return builtin_float(x)
            except:
                return x
        else:
            return builtin_float(x)


class pi_float(builtin_float, metaclass=pi_float_meta):
    pass


pi_float.__name__ = builtin_float.__name__
pi_float.__qualname__ = builtin_float.__qualname__


@contextlib.contextmanager
def swap_pi_int_float():
    __builtins__["int"] = pi_int
    __builtins__["bool"] = pi_bool
    __builtins__["float"] = pi_float
    from . import nn

    yield

    __builtins__["int"] = builtin_int
    __builtins__["bool"] = builtin_bool
    __builtins__["float"] = builtin_float
    from . import nn
