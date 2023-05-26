# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import contextlib
import functools
from enum import Enum
from typing import Optional, Union, Tuple, List

import numpy as np

from pi.mlir import Torch_BoolType, Torch_FloatType, Torch_IntType
from .dialects import _torch_ops_gen as torch_dialect
from .dialects import torch as torch_dialect
from .ir import (
    Context,
    Location,
    BF16Type,
    ComplexType,
    register_attribute_builder,
    FloatAttr,
    IntegerAttr,
    DenseElementsAttr,
    Module,
    InsertionPoint,
    IntegerType,
    F16Type,
    F32Type,
    F64Type,
    IndexType,
    RankedTensorType,
    Type,
    DictAttr,
    TypeAttr,
)


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


class dtype(Enum):
    """
    |-------------------|--------------------|
    | Torch Type        | MLIR Type          |
    |-------------------|--------------------|
    | torch.bfloat16    | bf16               |
    | torch.bool        | i1                 |
    | torch.complex*    | complex<*>         |
    | torch.float16     | f16                |
    | torch.float32     | f32                |
    | torch.float64     | f64                |
    | torch.int16       | si16               |
    | torch.int32       | si32               |
    | torch.int64       | si64               |
    | torch.int8        | si8                |
    | torch.qint8       | !torch.qint8       |
    | torch.quint8      | !torch.quint8      |
    | torch.uint8       | ui8                |
    |-------------------|--------------------|
    """

    uint8 = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4
    float16 = 5
    float32 = 6
    float64 = 7
    # complex_half 8
    complex32 = 9
    complex64 = 10
    bool = 11
    qint8 = 12
    quint8 = 13
    # qint32 14
    bfloat16 = 15

    # qint4x2 16
    # qint2x4 17

    def to_mlir_type(self):
        match self:
            case dtype.bfloat16:
                return BF16Type.get()
            case dtype.bool:
                return IntegerType.get_signless(1)
            case dtype.complex32:
                return ComplexType.get(F32Type.get())
            case dtype.complex64:
                return ComplexType.get(F64Type.get())
            case dtype.float16:
                return F16Type.get()
            case dtype.float32:
                return F32Type.get()
            case dtype.float64:
                return F64Type.get()
            case dtype.int8:
                return IntegerType.get_signed(8)
            case dtype.int16:
                return IntegerType.get_signed(16)
            case dtype.int32:
                return IntegerType.get_signed(32)
            case dtype.int64:
                return IntegerType.get_signed(64)
            case dtype.uint8:
                return IntegerType.get_unsigned(8)
            case _:
                raise NotImplementedError(f"unimplemented to mlir_type from {self}")

    def to_torch_value_type(self):
        match self:
            case dtype.bool:
                return Torch_BoolType()
            case dtype.float16 | dtype.float32 | dtype.float64:
                return Torch_FloatType()
            case dtype.int8 | dtype.int16 | dtype.int32 | dtype.int64 | dtype.uint8:
                return Torch_IntType()
            case _:
                raise NotImplementedError(
                    f"unimplemented to torch value type from {self}"
                )

    @staticmethod
    def from_np_type(self):
        match self:
            case np.half:
                return dtype.float16
            case np.bool_:
                return dtype.bool
            case np.singlecomplex:
                return dtype.complex32
            case np.complex_:
                return dtype.complex64
            case np.float32:
                return dtype.float32
            case np.float64:
                return dtype.float64
            case np.int8:
                return dtype.int8
            case np.int16:
                return dtype.int16
            case np.int32:
                return dtype.int32
            case np.int64:
                return dtype.int64
            case np.uint8:
                return dtype.uint8
            case _:
                raise NotImplementedError(f"unrecognized dtype: {self}")

    def to_np_type(self):
        match self:
            case dtype.bfloat16 | dtype.float16:
                return np.half
            case dtype.bool:
                return np.bool_
            case dtype.complex32:
                return np.singlecomplex
            case dtype.complex64:
                return np.complex_
            case dtype.float32:
                return np.float32
            case dtype.float64:
                return np.float64
            case dtype.int8:
                return np.int8
            case dtype.int16:
                return np.int16
            case dtype.int32:
                return np.int32
            case dtype.int64:
                return np.int64
            case dtype.uint8:
                return np.uint8
            case _:
                raise NotImplementedError(f"unrecognized dtype: {self}")

    @staticmethod
    def from_mlir_type(t: str):
        match t:
            case "bf16":
                return dtype.bfloat16
            case "i1":
                return dtype.bool
            case "complex32":
                return dtype.complex32
            case "complex64":
                return dtype.complex64
            case "f16":
                return dtype.float16
            case "f32":
                return dtype.float32
            case "f64":
                return dtype.float64
            case "si8":
                return dtype.int8
            case "si16":
                return dtype.int16
            case "si32":
                return dtype.int32
            case "si64":
                return dtype.int64
            case "ui8":
                return dtype.uint8
            case _:
                raise NotImplementedError(f"Something's wrong with the internet {t}")

    # IntegerType.get_signless(32) -> i32
    # IntegerType.get_signed(32) -> si32
    # IntegerType.get_unsigned(32) -> ui32
    def is_signless(self):
        return self in {dtype.bool}


@register_attribute_builder("F64Attr")
def _f64Attr(x, context):
    return FloatAttr.get(F64Type.get(context=context), x)


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
        dtype = {
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
        return RankedTensorType.get(py_val.shape, dtype)
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
        arr, signless=dtype_.is_signless(), type=dtype_.to_mlir_type(), shape=shape
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
    return torch_dialect.NonValueTensorLiteralOp(factory(*args, **kwargs))


empty = functools.partial(_np_wrapper, factory=np.empty)
ones = functools.partial(_np_wrapper, factory=np.ones)
zeros = functools.partial(_np_wrapper, factory=np.zeros)
rand = functools.partial(_np_wrapper, factory=np.random.rand)
randn = functools.partial(_np_wrapper, factory=np.random.randn)


class TensorPlaceholder:
    def __init__(self, shape: Union[Tuple[int, ...], List[int]], dtype_: dtype):
        self.shape = shape
        self.dtype = dtype_

    def to_value_tensor_type(self):
        dtype = self.dtype.to_mlir_type()
        # TODO(max): "modernize"
        type = Type.parse(f"!torch.vtensor<[{','.join(map(str, self.shape))}],{dtype}>")
        return type

    def to_nonvalue_tensor_type(self):
        # TODO(max): "modernize"
        type = Type.parse(f"!torch.tensor")
        return type

    def to_value_tensor_type_bound(self):
        type_attr = TypeAttr.get(self.to_value_tensor_type())
        return DictAttr.get({"torch.type_bound": type_attr})

    def to(self, dtype: dtype):
        self.dtype = dtype
        return self

    def type(self, dtype):
        return self.to(dtype)

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
