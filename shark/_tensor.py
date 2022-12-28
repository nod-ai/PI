from __future__ import annotations

from typing import List, Optional

from shark._mlir import _Torch_Tensor
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.dialects._ods_common import get_op_result_or_value
from torch_mlir.ir import (
    Value as MLIRValue,
)

from shark.types_ import parse_sizes_from_tensor_type_str, Number, Torch_ValueTensorType


class Tensor(_Torch_Tensor):
    def __init__(self, torch_tensor: MLIRValue):
        torch_tensor = get_op_result_or_value(torch_tensor)
        super(Tensor, self).__init__(torch_tensor._CAPIPtr)
        self._type = Torch_ValueTensorType(self.type)

    def size(self, dim: Optional[int] = None):
        sizes, _ = parse_sizes_from_tensor_type_str(self)
        # # in torch-mlir this is handled using IR
        # # see Utils::toPositiveDimDynamic
        # if dim < 0:
        #     dim = len(sizes) + dim
        #
        if dim is not None:
            s = sizes[dim]
            if s > 1:
                return s
            return torch_dialect.AtenSizeIntOp(self, dim).result
        else:
            return sizes

    @property
    def dtype(self):
        # TODO(max): pull from type directly
        _sizes, dtype = parse_sizes_from_tensor_type_str(self)
        return dtype

    def numel(self):
        return torch_dialect.AtenNumelOp(self).result

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(torch_dialect.AtenAddTensorOp(self, other))

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor(torch_dialect.AtenMulTensorOp(self, other))

    def flatten(self, start_dim: int, end_dim: int) -> Tensor:
        return Tensor(torch_dialect.AtenFlattenUsingIntsOp(self, start_dim, end_dim))

    def permute(self, *dims: List[int]):
        if len(dims) == 1 and isinstance(dims[0], list):
            dims = dims[0]
        return Tensor(torch_dialect.AtenPermuteOp(self, dims))

    def add(self, other: Tensor, alpha: Number) -> Tensor:
        return Tensor(torch_dialect.AtenAddTensorOp(self, other, alpha=alpha))

    def fill_(self, value: Number):
        return Tensor(torch_dialect.AtenFill_TensorOp(self, value))

    def softmax(self, dim, dtype=dtype):
        return Tensor(torch_dialect.AtenSoftmaxIntOp(self, dim, dtype))
