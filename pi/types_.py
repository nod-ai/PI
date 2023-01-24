import builtins
import weakref
from enum import Enum
from inspect import isclass
from typing import Any, Optional, Tuple
from typing import Generic, TypeVar
from typing import Union, List, NewType, Sequence

import numpy as np

# noinspection PyUnresolvedReferences
from pi._mlir import (
    TorchListOfNonValueTensorType as TorchListOfTensor,
    TorchListOfTorchBoolType as TorchListOfTorchBool,
    TorchListOfTorchIntType as TorchListOfTorchInt,
    TorchListOfTorchFloatType as TorchListOfTorchFloat,
    TorchListOfTorchStringType as TorchListOfTorchString,
)
from pi._mlir import (
    Torch_BoolType,
    Torch_FloatType,
    Torch_IntType,
    Torch_StringType,
    Torch_Value as _Torch_Value,
    Torch_List as _Torch_List,
    Torch_Tensor as Tensor,
)
from torch_mlir import ir
from torch_mlir._mlir_libs._mlir.ir import IntegerAttr, IntegerType, FloatAttr, F64Type
from torch_mlir.ir import (
    register_attribute_builder,
)
from typeguard import (
    TypeCheckError,
    TypeCheckerCallable,
    TypeCheckMemo,
    checker_lookup_functions,
    check_type,
)


T = TypeVar("T", Torch_FloatType, Torch_BoolType, Torch_StringType, Torch_IntType)


class Torch_Value(_Torch_Value, Generic[T]):
    def __init__(self, v):
        super().__init__(v)

    @property
    def type(self) -> T:
        return super().type


class Torch_List(_Torch_List, Generic[T]):
    def __init__(self, v):
        super().__init__(v)

    @property
    def type(self) -> T:
        return super().type


class Torch_Dict:
    ...
    # def __init__(self, v):
    #     super().__init__(v)
    #
    # @property
    # def type(self) -> T:
    #     return super().type


def check_simple_torch_value(
    value: Any, origin_type: Any, type_var_args: Tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    assert len(type_var_args) == 1, f"multiple type var args to Torch_Value not handled"
    type_var_arg = type_var_args[0]
    if not isinstance(value, Torch_Value):
        raise TypeCheckError(f"Not Torch_Value: {value}")
    try:
        check_type(value.type, type_var_arg)
    except TypeCheckError:
        raise TypeCheckError(f"Not correct type param ({type_var_arg}): {value.type}")


def check_simple_torch_list(
    value: Any, origin_type: Any, type_var_args: Tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    assert len(type_var_args) == 1, f"multiple type var args to Torch_List not handled"
    type_var_arg = type_var_args[0]
    if not isinstance(value, Torch_List):
        raise TypeCheckError(f"Not Torch_Value: {value}")
    try:
        check_type(value.el_type, type_var_arg)
    except TypeCheckError:
        raise TypeCheckError(
            f"Not correct type param ({type_var_arg}): {value.el_type}"
        )


def check_tensor(
    value: Any, origin_type: Any, type_var_args: Tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    if not isinstance(value, Tensor):
        raise TypeCheckError(f"Not Torch_Tensor: {value}")


def torch_value_checker_lookup(
    origin_type: Any, type_var_args: Tuple[Any, ...], extras: Tuple[Any, ...]
) -> Optional[TypeCheckerCallable]:
    if isclass(origin_type) and issubclass(origin_type, Torch_List):
        return check_simple_torch_list
    elif isclass(origin_type) and issubclass(origin_type, Torch_Value):
        return check_simple_torch_value
    elif isclass(origin_type) and issubclass(origin_type, Tensor):
        return check_tensor

    return None


checker_lookup_functions.insert(0, torch_value_checker_lookup)


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
                return ir.BF16Type.get()
            case dtype.bool:
                return ir.IntegerType.get_signless(1)
            case dtype.complex32:
                return ir.ComplexType.get(ir.F32Type.get())
            case dtype.complex64:
                return ir.ComplexType.get(ir.F64Type.get())
            case dtype.float16:
                return ir.F16Type.get()
            case dtype.float32:
                return ir.F32Type.get()
            case dtype.float64:
                return ir.F64Type.get()
            case dtype.int8:
                return ir.IntegerType.get_signed(8)
            case dtype.int16:
                return ir.IntegerType.get_signed(16)
            case dtype.int32:
                return ir.IntegerType.get_signed(32)
            case dtype.int64:
                return ir.IntegerType.get_signed(64)
            case dtype.uint8:
                return ir.IntegerType.get_unsigned(8)
            case _:
                raise NotImplementedError("Something's wrong with the internet")

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

Size = size = Union[List[int], Tuple[int, ...]]

Generator = Any

device = Device = TorchDevice = NewType("Device", str)


class BroadcastingListCls(object):
    def __getitem__(self, types):
        return


BroadcastingList1 = BroadcastingList2 = BroadcastingList3 = BroadcastingListCls()

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched: "weakref.WeakKeyDictionary[Callable, Dict[str, Callable]]" = (
    weakref.WeakKeyDictionary()
)  # noqa: T484


def boolean_dispatch(
    arg_name, arg_index, default, if_true, if_false, module_name, func_name
):
    def fn(*args, **kwargs):
        dispatch_flag = False
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name,
    }
    return fn


class layout(Enum):
    strided = 1
    sparse_coo = 2
    sparse_csr = 3
    sparse_csc = 4
    sparse_bsr = 5
    sparse_bsc = 6
    _mkldnn = 7


strided: layout = layout.strided
sparse_coo: layout = layout.sparse_coo
sparse_csr: layout = layout.sparse_csr
sparse_csc: layout = layout.sparse_csc
sparse_bsr: layout = layout.sparse_bsr
sparse_bsc: layout = layout.sparse_bsc
_mkldnn: layout = layout._mkldnn


class memory_format(Enum):
    contiguous_format = 0
    preserve_format = 1
    channels_last = 2
    channels_last_3d = 3


contiguous_format = memory_format.contiguous_format
preserve_format = memory_format.preserve_format
channels_last = memory_format.channels_last
channels_last_3d = memory_format.channels_last_3d


@register_attribute_builder("AnyI64Attr")
def _i64Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), x)


@register_attribute_builder("I1Attr")
def _i1Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(1, context=context), int(x))


@register_attribute_builder("F64Attr")
def _f64Attr(x, context):
    return FloatAttr.get(F64Type.get(context=context), x)


# ConstantNumberOp
@register_attribute_builder("anonymous_443")
def _numberAttr(x, context):
    if isinstance(x, builtins.float):
        return FloatAttr.get(F64Type.get(context=context), x)
    elif isinstance(x, builtins.int):
        return IntegerAttr.get(IntegerType.get_signless(64, context=context), x)
    else:
        raise TypeError(f"unhandled Number/Scalar {x}")
