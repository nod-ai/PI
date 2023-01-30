import functools
import re
from typing import (
    Tuple,
    Any,
    Union,
)

import numpy as np

# noinspection PyUnresolvedReferences
from pi._pi_mlir import Torch_Tensor, Torch_Value
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.ir import DenseElementsAttr

from .types_ import (
    dtype as pi_dtype,
)
from .torch_wrappers import empty, ones, zeros, tensor, device
from ._tensor import Tensor


def from_numpy(arr: np.ndarray, dtype: pi_dtype = None):
    shape = arr.shape
    if dtype is None:
        dtype = pi_dtype.from_np_type(arr.dtype)
    if dtype == pi_dtype.bool:
        arr = np.packbits(arr, axis=None, bitorder="little")
    attr = DenseElementsAttr.get(
        arr, signless=dtype.is_signless(), type=dtype.to_mlir_type(), shape=shape
    )

    vt = Tensor(torch_dialect.NonValueTensorLiteralOp(attr))
    return vt


def _np_wrapper(*size: Tuple[int, ...], **kwargs):
    factory = kwargs.get("factory", None)
    assert factory is not None
    if size == ((),) or len(size) == 0:
        return from_numpy(factory())

    if isinstance(size[0], (tuple, list)):
        assert len(size) == 1, f"malformed size tuple {size}"
        size = size[0]

    dtype = kwargs.get("dtype", None)
    try:
        if dtype is not None and factory not in (np.random.rand, np.random.randn):
            res = factory(size, dtype=dtype.to_np_type())
        else:
            res = factory(size)
    except TypeError as e:
        assert re.match(
            "'(tuple|list)' object cannot be interpreted as an integer", str(e)
        ), str(e)
        if dtype is not None and factory not in (np.random.rand, np.random.randn):
            res = factory(*size, dtype=dtype.to_np_type())
        else:
            res = factory(*size)
    return from_numpy(res, dtype=dtype)


def _torch_wrapper(*size: Tuple[int, ...], **kwargs):
    factory = kwargs.get("factory", None)
    assert factory is not None
    if size == ((),) or len(size) == 0:
        return factory()

    if isinstance(size[0], (tuple, list)):
        assert len(size) == 1, f"malformed size tuple {size}"
        size = size[0]

    dtype = kwargs.get("dtype", None)
    try:
        if dtype is not None:
            res = factory(size, dtype=dtype.value)
        else:
            res = factory(size)
    except TypeError as e:
        assert re.match(
            "'(tuple|list)' object cannot be interpreted as an integer", str(e)
        ), str(e)
        if dtype is not None:
            res = factory(*size, dtype=dtype.value)
        else:
            res = factory(*size)

    return res


# empty = functools.partial(_np_wrapper, factory=empty)
ones = functools.partial(_torch_wrapper, factory=ones)
zeros = functools.partial(_torch_wrapper, factory=zeros)
rand = functools.partial(_np_wrapper, factory=np.random.rand)
randn = functools.partial(_np_wrapper, factory=np.random.randn)


def tensor(arr, **kwargs):
    dtype = kwargs.get("dtype", None)
    if dtype is not None:
        res = np.array(arr, dtype=dtype.to_np_type())
    else:
        res = np.array(arr, dtype=np.float32)
    return from_numpy(res)


def LongTensor(data: Any) -> Tensor:
    return from_numpy(np.array(data, dtype=pi_dtype.int64.to_np_type()))


def FloatTensor(data: Any) -> Tensor:
    return from_numpy(np.array(data, dtype=pi_dtype.float64.to_np_type()))


def device(a: Union[Tensor, str]):
    if isinstance(a, Tensor):
        return device(a)
    elif isinstance(a, str):
        return Torch_Value(torch_dialect.ConstantDeviceOp(a).result)


__all__ = [
    "from_numpy",
    "empty",
    "randn",
    "rand",
    "ones",
    "tensor",
    "LongTensor",
    "FloatTensor",
    "zeros",
    "device",
]
