# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import contextlib
import functools
import inspect
import warnings
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np

from . import Torch_BoolType, Torch_FloatType, Torch_IntType, dtype
from .dialects import _torch_ops_gen as torch_dialect, torch as torch_dialect
from .ir import (
    Context,
    DenseElementsAttr,
    DictAttr,
    F16Type,
    F32Type,
    F64Type,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    RankedTensorType,
    Type,
    TypeAttr,
    register_attribute_builder,
)


@contextlib.contextmanager
def disable_multithreading(context=None):
    from . import DefaultContext

    if context is None:
        context = DefaultContext

    context.enable_multithreading(False)
    yield
    context.enable_multithreading(True)


@contextlib.contextmanager
def mlir_mod_ctx(
    src: Optional[str] = None, context: Context = None, location: Location = None
):
    if context is None:
        context = Context()
    with context:
        if location is None:
            location = Location.unknown()
        with location:
            if src is not None:
                module = Module.parse(src)
            else:
                module = Module.create()
            with InsertionPoint(module.body):
                yield module


@register_attribute_builder("AnyI64Attr")
def _anyI64Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), x)


@register_attribute_builder("I1Attr")
def _i1Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(1, context=context), x)


def infer_mlir_type(
    py_val: Union[int, float, bool, np.ndarray]
) -> Union[IntegerType, F64Type, RankedTensorType]:
    if isinstance(py_val, bool):
        return IntegerType.get_signless(1)
    elif isinstance(py_val, int):
        return IntegerType.get_signless(64)
    elif isinstance(py_val, float):
        return F64Type.get()
    elif isinstance(py_val, np.ndarray):
        dtype_ = {
            np.int8: IntegerType.get_signless(8),
            np.int16: IntegerType.get_signless(16),
            np.int32: IntegerType.get_signless(32),
            np.int64: IntegerType.get_signless(64),
            np.uintp: IndexType.get(),
            np.longlong: IndexType.get(),
            np.float16: F16Type.get(),
            np.float32: F32Type.get(),
            np.float64: F64Type.get(),
        }[py_val.dtype.type]
        return RankedTensorType.get(py_val.shape, dtype_)
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )


def attr_from_numpy(arr: np.ndarray, dtype_: dtype = None):
    shape = arr.shape
    if dtype_ is None:
        dtype_ = dtype.from_np_type(arr.dtype)
    if dtype_ == dtype.bool:
        arr = np.packbits(arr, axis=None, bitorder="little")
    return DenseElementsAttr.get(
        arr,
        signless=dtype_.is_signless(),
        type=dtype_.to_mlir_type(),
        shape=shape,
    )


@register_attribute_builder("ElementsAttr")
def _elementsAttr(x, context=None):
    if isinstance(x, np.ndarray):
        t = attr_from_numpy(x)
    else:
        t = DenseElementsAttr.get(
            np.array(x, dtype=np.float64),
            type=F64Type.get(context=context),
        )
    return t


def _np_wrapper(*args, factory=None, **kwargs):
    if "device" in kwargs:
        kwargs.pop("device")
    if "dtype" in kwargs and isinstance(kwargs["dtype"], dtype):
        kwargs["dtype"] = kwargs["dtype"].to_np_type()
    return torch_dialect.NonValueTensorLiteralOp(factory(*args, **kwargs))


empty = functools.partial(_np_wrapper, factory=np.empty)
ones = functools.partial(_np_wrapper, factory=np.ones)
zeros = functools.partial(_np_wrapper, factory=np.zeros)
rand = functools.partial(_np_wrapper, factory=np.random.rand)
randn = functools.partial(_np_wrapper, factory=np.random.randn)
tensor = functools.partial(_np_wrapper, factory=np.array)
zeros_like = functools.partial(_np_wrapper, factory=np.zeros_like)
empty_like = functools.partial(_np_wrapper, factory=np.empty_like)


class layout(Enum):
    strided = 1
    sparse_coo = 2
    sparse_csr = 3
    sparse_csc = 4
    sparse_bsr = 5
    sparse_bsc = 6
    _mkldnn = 7


class memory_format(Enum):
    contiguous_format = 0
    preserve_format = 1
    channels_last = 2
    channels_last_3d = 3


class TensorPlaceholder:
    def __init__(self, shape: Union[Tuple[int, ...], List[int]], dtype_: dtype):
        self.shape = shape
        self.dtype = dtype_

    def to_value_tensor_type(self):
        dtype_ = self.dtype.to_mlir_type()
        # TODO(max): "modernize"
        type = Type.parse(
            f"!torch.vtensor<[{','.join(map(str, self.shape))}],{dtype_}>"
        )
        return type

    def to_nonvalue_tensor_type(self):
        # TODO(max): "modernize"
        type = Type.parse(f"!torch.tensor")
        return type

    def to_value_tensor_type_bound(self):
        type_attr = TypeAttr.get(self.to_value_tensor_type())
        return DictAttr.get({"torch.type_bound": type_attr})

    def to(self, dtype_: dtype):
        self.dtype = dtype_
        return self

    def type(self, dtype_):
        return self.to(dtype_)

    def bool(self):
        return self.to(dtype.bool)

    def double(self):
        self.dtype = dtype.float64
        return self

    def int(self):
        self.dtype = dtype.int32
        return self

    def long(self):
        self.dtype = dtype.int64
        return self


ArgAnnotation = Union[type, Tuple[List[int], dtype]]


def annotations_to_placeholders(
    args: List[str], annotations: List[Optional[ArgAnnotation]]
):
    from collections import OrderedDict

    placeholders = OrderedDict()
    for annotation, arg in zip(annotations, args):
        # Skip the "self" annotation.
        if annotation is None:
            assert arg == "self"
            continue
        shape, dtype, value_tensor = annotation
        assert value_tensor, f"non-value tensors not supported {arg}"
        if not shape:
            warnings.warn(f"empty shape annotation: {shape}")
        placeholders[arg] = TensorPlaceholder(annotation[0], annotation[1])

    return placeholders


def annotate_args(annotations: List[Optional[ArgAnnotation]]):
    def decorator(fn):
        arg_spec = inspect.getfullargspec(fn)
        placeholders = annotations_to_placeholders(arg_spec.args, annotations)
        setattr(fn, "__placeholders__", placeholders)
        return fn

    return decorator


def export(fn):
    return fn
