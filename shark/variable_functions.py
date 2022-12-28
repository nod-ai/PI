from typing import Tuple, Optional, Any, Union, List

import shark
from shark import Tensor, Number, dtype

import numpy as np
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.ir import (
    DenseElementsAttr,
)


# TODO(max): put this somewhere central
DEBUG = True

__all__ = [
    "from_numpy",
    "empty",
    "randint",
    "uniform",
    "ones",
    "mm",
    "bmm",
    "tanh",
    "addmm",
    "tensor",
    "transpose",
    "cat",
    "gather",
    "arange",
    "embedding",
    "embedding_bag",
]


def from_numpy(arr: np.ndarray):
    if DEBUG:
        arr = np.ones_like(arr, dtype=np.float32)
    attr = DenseElementsAttr.get(arr)
    vt = Tensor(torch_dialect.ValueTensorLiteralOp(attr))
    return vt


def empty(shape: Tuple[int, ...], dtype: shark.dtype = shark.dtype.float32) -> Tensor:
    # TODO(max): handle other dtypes (i.e. figure out a non-dumb way to map between np and torch_mlir and pytorch
    if np.prod(shape) == 0:
        return Tensor(None)
    else:
        return from_numpy(np.empty(shape))


def randint(low: int, high: int, size: Tuple[int, ...]) -> Tensor:
    return from_numpy(np.random.randint(low, high, size))


def uniform(low: float, high: float, size: Tuple[int, ...]) -> Tensor:
    return from_numpy(np.random.uniform(low, high, size))


def ones(size: Tuple[int, ...]) -> Tensor:
    return from_numpy(np.ones(size))


def mm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    assert out is None, "out variant not supported"
    return Tensor(torch_dialect.AtenMmOp(input, mat2))


def bmm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    assert out is None, "out variant not supported"
    return Tensor(torch_dialect.AtenBmmOp(input, mat2))


def tanh(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    assert out is None, "out variant not supported"
    return Tensor(torch_dialect.AtenTanhOp(input))


def addmm(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: Number = 1,
    alpha: Number = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    assert out is None, "out variant not supported"
    return Tensor(torch_dialect.AtenAddmmOp(input, batch1, batch2, beta, alpha))


def tensor(data: Any, dtype: Optional[dtype] = None) -> Tensor:
    return from_numpy(np.array(data, dtype=dtype))


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    return Tensor(torch_dialect.AtenTransposeIntOp(input, dim0, dim1))


def cat(
    tensors: Union[Tuple[Tensor, ...], List[Tensor]],
    dim: int = 0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    assert out is None, "out variant not supported"
    return Tensor(torch_dialect.AtenCatOp(tensors, dim))


def gather(
    input: Tensor,
    dim: int,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    assert out is None, "out variant not supported"
    return Tensor(
        torch_dialect.AtenGatherOp(input, dim, index, sparse_grad=sparse_grad)
    )


def arange(
    start: Number,
    end: Number,
    step: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Optional[dtype] = None,
) -> Tensor:
    assert out is None, "out variant not supported"
    return Tensor(torch_dialect.AtenArangeStartStepOp(start, end, step, dtype))


def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    return Tensor(
        torch_dialect.AtenEmbeddingOp(
            weight, indices, padding_idx, scale_grad_by_freq, sparse
        )
    )


def embedding_bag(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: bool,
    mode: int,
    sparse: bool,
    per_sample_weights: Optional[Tensor],
    include_last_offset: bool,
    padding_idx: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return Tensor(
        torch_dialect.AtenEmbeddingBagPaddingIdxOp(
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        )
    )
