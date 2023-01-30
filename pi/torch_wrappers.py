from torch_mlir.dialects import torch as torch_dialect

# noinspection PyUnresolvedReferences
from ._torch_wrappers import *
from . import _torch_wrappers
from ._pi_mlir import Torch_NumberType
from ._tensor import Tensor
from .types_ import (
    Torch_Value,
)


def ScalarImplicit(a: Tensor) -> _torch_wrappers.TorchNumber:
    # assert check_argument_types()
    result0_type = Torch_NumberType()
    return Torch_Value(torch_dialect.AtenScalarImplicitOp(result0_type, a).result)


__all__ = _torch_wrappers.__all__ + ["ScalarImplicit"]
