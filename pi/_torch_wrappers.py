import builtins
from typing import List, Optional, Any, Dict, Tuple, Union, Sequence

from ._tensor import Tensor
from .types_ import (
    dtype as pi_dtype,
    layout as pi_layout,
    memory_format as pi_memory_format,
    Torch_Value,
    Torch_List,
    Torch_Dict,
)

# noinspection PyUnresolvedReferences
from ._pi_mlir import (
    TorchListOfTorchBoolType,
    TorchListOfTorchFloatType,
    TorchListOfTorchIntType,
    TorchListOfTorchStringType,
    TorchListOfNonValueTensorType,
    TorchListOfOptionalTensorType,
    TorchOptionalBoolType,
    TorchOptionalFloatType,
    TorchOptionalIntType,
    TorchOptionalStringType,
    TorchOptionalDeviceType,
    TorchOptionalGeneratorType,
    TorchOptionalNonValueTensorType,
    Torch_AnyType,
    Torch_BoolType,
    Torch_DeviceType,
    Torch_FloatType,
    Torch_IntType,
    Torch_NumberType,
    Torch_StringType,
    Torch_NonValueTensorType,
    Torch_GeneratorType,
)
from .dispatcher import register_dispatch
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.dialects._ods_common import (
    get_op_results_or_values,
)
from typeguard import check_argument_types

TorchNumber = Union[
    Torch_Value[Torch_IntType], Torch_Value[Torch_FloatType], int, float
]


# overload Tensor
@register_dispatch
def Bool(a: Tensor) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenBoolTensorOp(a).result)


# overload float
@register_dispatch
def Bool(
    a: Union[Torch_Value[Torch_FloatType], builtins.float]
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    return Torch_Value(torch_dialect.AtenBoolFloatOp(a).result)


# overload int
@register_dispatch
def Bool(
    a: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    return Torch_Value(torch_dialect.AtenBoolIntOp(a).result)


# overload Dict_str
def Delete(
    self_: Torch_Dict, key: Union[Torch_Value[Torch_StringType], builtins.str]
) -> None:
    assert check_argument_types()
    if isinstance(key, builtins.str):
        key = torch_dialect.ConstantStrOp(key).result
    torch_dialect.AtenDeleteDictStrOp(self_, key)


# overload Tensor
@register_dispatch
def Float(a: Tensor) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenFloatTensorOp(a).result)


# overload Scalar
@register_dispatch
def Float(a: TorchNumber) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, (builtins.int, builtins.float)):
        a = torch_dialect.ConstantNumberOp(a).result
    return Torch_Value(torch_dialect.AtenFloatScalarOp(a).result)


# overload str
@register_dispatch
def Float(
    a: Union[Torch_Value[Torch_StringType], builtins.str]
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.str):
        a = torch_dialect.ConstantStrOp(a).result
    return Torch_Value(torch_dialect.AtenFloatStrOp(a).result)


def FloatImplicit(a: Tensor) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenFloatImplicitOp(a).result)


# overload Tensor
@register_dispatch
def Int(a: Tensor) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenIntTensorOp(a).result)


# overload float
@register_dispatch
def Int(
    a: Union[Torch_Value[Torch_FloatType], builtins.float]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    return Torch_Value(torch_dialect.AtenIntFloatOp(a).result)


# overload Scalar
@register_dispatch
def Int(a: TorchNumber) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, (builtins.int, builtins.float)):
        a = torch_dialect.ConstantNumberOp(a).result
    return Torch_Value(torch_dialect.AtenIntScalarOp(a).result)


# overload bool
@register_dispatch
def Int(
    a: Union[Torch_Value[Torch_BoolType], builtins.bool]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.bool):
        a = torch_dialect.ConstantBoolOp(a).result
    return Torch_Value(torch_dialect.AtenIntBoolOp(a).result)


def IntImplicit(a: Tensor) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenIntImplicitOp(a).result)


# overload Scalar
def NumToTensor(a: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(a, (builtins.int, builtins.float)):
        a = torch_dialect.ConstantNumberOp(a).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.PrimNumToTensorScalarOp(result0_type, a))


def RaiseException(
    msg: Union[Torch_Value[Torch_StringType], builtins.str],
    cls: Union[Torch_Value[Torch_StringType], builtins.str, None] = None,
) -> None:
    assert check_argument_types()
    if isinstance(msg, builtins.str):
        msg = torch_dialect.ConstantStrOp(msg).result
    if isinstance(cls, builtins.str):
        cls = torch_dialect.ConstantStrOp(cls).result
    if cls is None:
        cls = torch_dialect.ConstantNoneOp().result
    torch_dialect.PrimRaiseExceptionOp(msg, cls)


def TupleIndex(
    tup: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
    i: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[
    TorchNumber,
    Tensor,
    Torch_Value[Torch_AnyType],
    Any,
    Torch_Value[Torch_BoolType],
    builtins.bool,
    Torch_Dict,
    Torch_Value[Torch_DeviceType],
    builtins.str,
    Torch_GeneratorType,
    Torch_List,
    None,
    Torch_Value[Torch_StringType],
    Tuple,
]:
    assert check_argument_types()
    if isinstance(i, builtins.int):
        i = torch_dialect.ConstantIntOp(i).result
    result0_type = Torch_AnyType()
    return Torch_Value(torch_dialect.PrimTupleIndexOp(result0_type, tup, i).result)


def Uninitialized() -> (
    Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ]
):
    assert check_argument_types()
    result0_type = Torch_AnyType()
    return Torch_Value(
        torch_dialect.PrimUninitializedOp(
            result0_type,
        ).result
    )


# overload Tensor
@register_dispatch
def __and__(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.Aten__And__TensorOp(result0_type, self_, other))


# overload bool
@register_dispatch
def __and__(
    a: Union[Torch_Value[Torch_BoolType], builtins.bool],
    b: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.bool):
        a = torch_dialect.ConstantBoolOp(a).result
    if isinstance(b, builtins.bool):
        b = torch_dialect.ConstantBoolOp(b).result
    return Torch_Value(torch_dialect.Aten__And__BoolOp(a, b).result)


# overload str
@register_dispatch
def __contains__(
    dict: Torch_Dict, key: Union[Torch_Value[Torch_StringType], builtins.str]
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(key, builtins.str):
        key = torch_dialect.ConstantStrOp(key).result
    return Torch_Value(torch_dialect.Aten__Contains__StrOp(dict, key).result)


# overload int_list
@register_dispatch
def __contains__(
    l: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    item: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(l, (builtins.list, builtins.tuple)) and builtins.len(l):
        l = builtins.list(l)
        for i, a in enumerate(l):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                l[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        l = torch_dialect.PrimListConstructOp(ls_type, l).result
    if isinstance(item, builtins.int):
        item = torch_dialect.ConstantIntOp(item).result
    return Torch_Value(torch_dialect.Aten__Contains__IntListOp(l, item).result)


def __derive_index(
    index: Union[Torch_Value[Torch_IntType], builtins.int],
    start: Union[Torch_Value[Torch_IntType], builtins.int],
    step: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(index, builtins.int):
        index = torch_dialect.ConstantIntOp(index).result
    if isinstance(start, builtins.int):
        start = torch_dialect.ConstantIntOp(start).result
    if isinstance(step, builtins.int):
        step = torch_dialect.ConstantIntOp(step).result
    return Torch_Value(torch_dialect.Aten__DeriveIndexOp(index, start, step).result)


# overload Dict_str
@register_dispatch
def __getitem__(
    self_: Torch_Dict, key: Union[Torch_Value[Torch_StringType], builtins.str]
) -> Union[
    TorchNumber,
    Tensor,
    Torch_Value[Torch_AnyType],
    Any,
    Torch_Value[Torch_BoolType],
    builtins.bool,
    Torch_Dict,
    Torch_Value[Torch_DeviceType],
    builtins.str,
    Torch_GeneratorType,
    Torch_List,
    None,
    Torch_Value[Torch_StringType],
    Tuple,
]:
    assert check_argument_types()
    if isinstance(key, builtins.str):
        key = torch_dialect.ConstantStrOp(key).result
    result0_type = Torch_AnyType()
    return Torch_Value(
        torch_dialect.Aten__Getitem__DictStrOp(result0_type, self_, key).result
    )


# overload t
@register_dispatch
def __getitem__(
    list_: Union[Sequence[Any], Any],
    idx: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[
    TorchNumber,
    Tensor,
    Torch_Value[Torch_AnyType],
    Any,
    Torch_Value[Torch_BoolType],
    builtins.bool,
    Torch_Dict,
    Torch_Value[Torch_DeviceType],
    builtins.str,
    Torch_GeneratorType,
    Torch_List,
    None,
    Torch_Value[Torch_StringType],
    Tuple,
]:
    assert check_argument_types()
    if isinstance(idx, builtins.int):
        idx = torch_dialect.ConstantIntOp(idx).result
    result0_type = Torch_AnyType()
    return Torch_Value(
        torch_dialect.Aten__Getitem__TOp(result0_type, list_, idx).result
    )


def __is__(
    self_: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
    obj: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.Aten__Is__Op(self_, obj).result)


def __isnot__(
    self_: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
    obj: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.Aten__Isnot__Op(self_, obj).result)


def __not__(
    self_: Union[Torch_Value[Torch_BoolType], builtins.bool]
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(self_, builtins.bool):
        self_ = torch_dialect.ConstantBoolOp(self_).result
    return Torch_Value(torch_dialect.Aten__Not__Op(self_).result)


def __range_length(
    lo: Union[Torch_Value[Torch_IntType], builtins.int],
    hi: Union[Torch_Value[Torch_IntType], builtins.int],
    step: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(lo, builtins.int):
        lo = torch_dialect.ConstantIntOp(lo).result
    if isinstance(hi, builtins.int):
        hi = torch_dialect.ConstantIntOp(hi).result
    if isinstance(step, builtins.int):
        step = torch_dialect.ConstantIntOp(step).result
    return Torch_Value(torch_dialect.Aten__RangeLengthOp(lo, hi, step).result)


@register_dispatch
def _convolution(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    transposed: Union[Torch_Value[Torch_BoolType], builtins.bool],
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    groups: Union[Torch_Value[Torch_IntType], builtins.int],
    benchmark: Union[Torch_Value[Torch_BoolType], builtins.bool],
    deterministic: Union[Torch_Value[Torch_BoolType], builtins.bool],
    cudnn_enabled: Union[Torch_Value[Torch_BoolType], builtins.bool],
    allow_tf32: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(transposed, builtins.bool):
        transposed = torch_dialect.ConstantBoolOp(transposed).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    if isinstance(benchmark, builtins.bool):
        benchmark = torch_dialect.ConstantBoolOp(benchmark).result
    if isinstance(deterministic, builtins.bool):
        deterministic = torch_dialect.ConstantBoolOp(deterministic).result
    if isinstance(cudnn_enabled, builtins.bool):
        cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled).result
    if isinstance(allow_tf32, builtins.bool):
        allow_tf32 = torch_dialect.ConstantBoolOp(allow_tf32).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_ConvolutionOp(
            result0_type,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            benchmark,
            deterministic,
            cudnn_enabled,
            allow_tf32,
        )
    )


# overload deprecated
@register_dispatch
def _convolution(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    transposed: Union[Torch_Value[Torch_BoolType], builtins.bool],
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    groups: Union[Torch_Value[Torch_IntType], builtins.int],
    benchmark: Union[Torch_Value[Torch_BoolType], builtins.bool],
    deterministic: Union[Torch_Value[Torch_BoolType], builtins.bool],
    cudnn_enabled: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(transposed, builtins.bool):
        transposed = torch_dialect.ConstantBoolOp(transposed).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    if isinstance(benchmark, builtins.bool):
        benchmark = torch_dialect.ConstantBoolOp(benchmark).result
    if isinstance(deterministic, builtins.bool):
        deterministic = torch_dialect.ConstantBoolOp(deterministic).result
    if isinstance(cudnn_enabled, builtins.bool):
        cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_ConvolutionDeprecatedOp(
            result0_type,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            benchmark,
            deterministic,
            cudnn_enabled,
        )
    )


def _embedding_bag(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    mode: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    sparse: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    padding_idx: Union[Torch_Value[Torch_IntType], builtins.int] = -1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(scale_grad_by_freq, builtins.bool):
        scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq).result
    if isinstance(mode, builtins.int):
        mode = torch_dialect.ConstantIntOp(mode).result
    if isinstance(sparse, builtins.bool):
        sparse = torch_dialect.ConstantBoolOp(sparse).result
    if per_sample_weights is None:
        per_sample_weights = torch_dialect.ConstantNoneOp().result
    if isinstance(include_last_offset, builtins.bool):
        include_last_offset = torch_dialect.ConstantBoolOp(include_last_offset).result
    if isinstance(padding_idx, builtins.int):
        padding_idx = torch_dialect.ConstantIntOp(padding_idx).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    result2_type = Torch_NonValueTensorType()
    result3_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.Aten_EmbeddingBagOp(
            result0_type,
            result1_type,
            result2_type,
            result3_type,
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
    return tuple([Tensor(o) for o in op_results])


def _index_put_impl(
    self_: Tensor,
    indices: Union[Sequence[Optional[Tensor]], Tensor, None],
    values: Tensor,
    accumulate: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    unsafe: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    indices = builtins.list(indices)
    for i, a in enumerate(indices):
        if a is not None:
            assert isinstance(a, Tensor)
        else:
            indices[i] = torch_dialect.ConstantNoneOp().result
    indices = torch_dialect.PrimListConstructOp(
        TorchListOfOptionalTensorType(), indices
    ).result
    if isinstance(accumulate, builtins.bool):
        accumulate = torch_dialect.ConstantBoolOp(accumulate).result
    if isinstance(unsafe, builtins.bool):
        unsafe = torch_dialect.ConstantBoolOp(unsafe).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_IndexPutImplOp(
            result0_type, self_, indices, values, accumulate, unsafe
        )
    )


def _index_put_impl_(
    self_: Tensor,
    indices: Union[Sequence[Optional[Tensor]], Tensor, None],
    values: Tensor,
    accumulate: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    unsafe: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    indices = builtins.list(indices)
    for i, a in enumerate(indices):
        if a is not None:
            assert isinstance(a, Tensor)
        else:
            indices[i] = torch_dialect.ConstantNoneOp().result
    indices = torch_dialect.PrimListConstructOp(
        TorchListOfOptionalTensorType(), indices
    ).result
    if isinstance(accumulate, builtins.bool):
        accumulate = torch_dialect.ConstantBoolOp(accumulate).result
    if isinstance(unsafe, builtins.bool):
        unsafe = torch_dialect.ConstantBoolOp(unsafe).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_IndexPutImpl_Op(
            result0_type, self_, indices, values, accumulate, unsafe
        )
    )


def _log_softmax(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    half_to_float: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(half_to_float, builtins.bool):
        half_to_float = torch_dialect.ConstantBoolOp(half_to_float).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_LogSoftmaxOp(result0_type, self_, dim, half_to_float)
    )


def _log_softmax_backward_data(
    grad_output: Tensor,
    output: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    input_dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(input_dtype, pi_dtype):
        input_dtype = input_dtype.value
    if isinstance(input_dtype, builtins.int):
        input_dtype = torch_dialect.ConstantIntOp(input_dtype).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_LogSoftmaxBackwardDataOp(
            result0_type, grad_output, output, dim, input_dtype
        )
    )


def _reshape_alias(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.Aten_ReshapeAliasOp(result0_type, self_, size, stride))


def _reshape_alias_copy(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_ReshapeAliasCopyOp(result0_type, self_, size, stride)
    )


# overload str
@register_dispatch
def _set_item(
    l: Torch_Dict,
    idx: Union[Torch_Value[Torch_StringType], builtins.str],
    v: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
) -> None:
    assert check_argument_types()
    if isinstance(idx, builtins.str):
        idx = torch_dialect.ConstantStrOp(idx).result
    torch_dialect.Aten_SetItemStrOp(l, idx, v)


# overload t
@register_dispatch
def _set_item(
    l: Union[Sequence[Any], Any],
    idx: Union[Torch_Value[Torch_IntType], builtins.int],
    el: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
) -> Union[Sequence[Any], Any]:
    assert check_argument_types()
    if isinstance(idx, builtins.int):
        idx = torch_dialect.ConstantIntOp(idx).result
    result0_type = Torch_List.of(Torch_AnyType())
    return Torch_Value(torch_dialect.Aten_SetItemTOp(result0_type, l, idx, el).result)


def _shape_as_tensor(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.Aten_ShapeAsTensorOp(result0_type, self_))


def _softmax(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    half_to_float: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(half_to_float, builtins.bool):
        half_to_float = torch_dialect.ConstantBoolOp(half_to_float).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.Aten_SoftmaxOp(result0_type, self_, dim, half_to_float))


def _softmax_backward_data(
    grad_output: Tensor,
    output: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    input_dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(input_dtype, pi_dtype):
        input_dtype = input_dtype.value
    if isinstance(input_dtype, builtins.int):
        input_dtype = torch_dialect.ConstantIntOp(input_dtype).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_SoftmaxBackwardDataOp(
            result0_type, grad_output, output, dim, input_dtype
        )
    )


def _to_copy(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    non_blocking: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(non_blocking, builtins.bool):
        non_blocking = torch_dialect.ConstantBoolOp(non_blocking).result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.Aten_ToCopyOp(
            result0_type,
            self_,
            dtype,
            layout,
            device,
            pin_memory,
            non_blocking,
            memory_format,
        )
    )


def _unsafe_view(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.Aten_UnsafeViewOp(result0_type, self_, size))


@register_dispatch
def abs(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAbsOp(result0_type, self_))


# overload Scalar
@register_dispatch
def abs(a: TorchNumber) -> TorchNumber:
    assert check_argument_types()
    if isinstance(a, (builtins.int, builtins.float)):
        a = torch_dialect.ConstantNumberOp(a).result
    result0_type = Torch_NumberType()
    return Torch_Value(torch_dialect.PrimAbsScalarOp(result0_type, a).result)


def abs_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAbs_Op(result0_type, self_))


def adaptive_avg_pool2d(
    self_: Tensor,
    output_size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(output_size, (builtins.list, builtins.tuple)) and builtins.len(
        output_size
    ):
        output_size = builtins.list(output_size)
        for i, a in enumerate(output_size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_size = torch_dialect.PrimListConstructOp(ls_type, output_size).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAdaptiveAvgPool2dOp(result0_type, self_, output_size)
    )


# overload Tensor
@register_dispatch
def add(self_: Tensor, other: Tensor, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAddTensorOp(result0_type, self_, other, alpha))


# overload Scalar
@register_dispatch
def add(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAddScalarOp(result0_type, self_, other, alpha))


# overload str
@register_dispatch
def add(
    a: Union[Torch_Value[Torch_StringType], builtins.str],
    b: Union[Torch_Value[Torch_StringType], builtins.str],
) -> Union[Torch_Value[Torch_StringType], builtins.str]:
    assert check_argument_types()
    if isinstance(a, builtins.str):
        a = torch_dialect.ConstantStrOp(a).result
    if isinstance(b, builtins.str):
        b = torch_dialect.ConstantStrOp(b).result
    return Torch_Value(torch_dialect.AtenAddStrOp(a, b).result)


# overload int
@register_dispatch
def add(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenAddIntOp(a, b).result)


# overload float_int
@register_dispatch
def add(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenAddFloatIntOp(a, b).result)


@register_dispatch
def add(a: TorchNumber, b: TorchNumber) -> TorchNumber:
    assert check_argument_types()
    if isinstance(a, (builtins.int, builtins.float)):
        a = torch_dialect.ConstantNumberOp(a).result
    if isinstance(b, (builtins.int, builtins.float)):
        b = torch_dialect.ConstantNumberOp(b).result
    result0_type = Torch_NumberType()
    return Torch_Value(torch_dialect.AtenAddOp(result0_type, a, b).result)


# overload Tensor
@register_dispatch
def add_(self_: Tensor, other: Tensor, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAdd_TensorOp(result0_type, self_, other, alpha))


# overload Scalar
@register_dispatch
def add_(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAdd_ScalarOp(result0_type, self_, other, alpha))


def addcdiv(
    self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: TorchNumber = 1
) -> Tensor:
    assert check_argument_types()
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAddcdivOp(result0_type, self_, tensor1, tensor2, value)
    )


def addcdiv_(
    self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: TorchNumber = 1
) -> Tensor:
    assert check_argument_types()
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAddcdiv_Op(result0_type, self_, tensor1, tensor2, value)
    )


def addcmul(
    self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: TorchNumber = 1
) -> Tensor:
    assert check_argument_types()
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAddcmulOp(result0_type, self_, tensor1, tensor2, value)
    )


def addcmul_(
    self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: TorchNumber = 1
) -> Tensor:
    assert check_argument_types()
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAddcmul_Op(result0_type, self_, tensor1, tensor2, value)
    )


def addmm(
    self_: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    beta: TorchNumber = 1,
    alpha: TorchNumber = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(beta, (builtins.int, builtins.float)):
        beta = torch_dialect.ConstantNumberOp(beta).result
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAddmmOp(result0_type, self_, mat1, mat2, beta, alpha)
    )


def alias_copy(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAliasCopyOp(result0_type, self_))


@register_dispatch
def all(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAllOp(result0_type, self_))


# overload bool
@register_dispatch
def all(
    self_: Union[
        Sequence[Union[Torch_Value[Torch_BoolType], builtins.bool]],
        Torch_List[Torch_BoolType],
    ]
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(self_, (builtins.list, builtins.tuple)) and builtins.len(self_):
        self_ = builtins.list(self_)
        for i, a in enumerate(self_):
            if not isinstance(a, builtins.bool):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be bool"
            else:
                self_[i] = torch_dialect.ConstantBoolOp(a).result
        ls_type = Torch_List.of(Torch_BoolType())
        self_ = torch_dialect.PrimListConstructOp(ls_type, self_).result
    return Torch_Value(torch_dialect.AtenAllBoolOp(self_).result)


def amax(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (),
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAmaxOp(result0_type, self_, dim, keepdim))


@register_dispatch
def any(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAnyOp(result0_type, self_))


# overload dim
@register_dispatch
def any(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAnyDimOp(result0_type, self_, dim, keepdim))


# overload bool
@register_dispatch
def any(
    self_: Union[
        Sequence[Union[Torch_Value[Torch_BoolType], builtins.bool]],
        Torch_List[Torch_BoolType],
    ]
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(self_, (builtins.list, builtins.tuple)) and builtins.len(self_):
        self_ = builtins.list(self_)
        for i, a in enumerate(self_):
            if not isinstance(a, builtins.bool):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be bool"
            else:
                self_[i] = torch_dialect.ConstantBoolOp(a).result
        ls_type = Torch_List.of(Torch_BoolType())
        self_ = torch_dialect.PrimListConstructOp(ls_type, self_).result
    return Torch_Value(torch_dialect.AtenAnyBoolOp(self_).result)


# overload t
def append(
    self_: Union[Sequence[Any], Any],
    el: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
) -> Union[Sequence[Any], Any]:
    assert check_argument_types()
    result0_type = Torch_List.of(Torch_AnyType())
    return Torch_Value(torch_dialect.AtenAppendTOp(result0_type, self_, el).result)


@register_dispatch
def arange(
    end: TorchNumber,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(end, (builtins.int, builtins.float)):
        end = torch_dialect.ConstantNumberOp(end).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenArangeOp(result0_type, end, dtype, layout, device, pin_memory)
    )


# overload start
@register_dispatch
def arange(
    start: TorchNumber,
    end: TorchNumber,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(start, (builtins.int, builtins.float)):
        start = torch_dialect.ConstantNumberOp(start).result
    if isinstance(end, (builtins.int, builtins.float)):
        end = torch_dialect.ConstantNumberOp(end).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenArangeStartOp(
            result0_type, start, end, dtype, layout, device, pin_memory
        )
    )


# overload start_step
@register_dispatch
def arange(
    start: TorchNumber,
    end: TorchNumber,
    step: TorchNumber = 1,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(start, (builtins.int, builtins.float)):
        start = torch_dialect.ConstantNumberOp(start).result
    if isinstance(end, (builtins.int, builtins.float)):
        end = torch_dialect.ConstantNumberOp(end).result
    if isinstance(step, (builtins.int, builtins.float)):
        step = torch_dialect.ConstantNumberOp(step).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenArangeStartStepOp(
            result0_type, start, end, step, dtype, layout, device, pin_memory
        )
    )


# overload start_out
@register_dispatch
def arange(
    start: TorchNumber, end: TorchNumber, step: TorchNumber = 1, out: Tensor = None
) -> Tensor:
    assert check_argument_types()
    if isinstance(start, (builtins.int, builtins.float)):
        start = torch_dialect.ConstantNumberOp(start).result
    if isinstance(end, (builtins.int, builtins.float)):
        end = torch_dialect.ConstantNumberOp(end).result
    if isinstance(step, (builtins.int, builtins.float)):
        step = torch_dialect.ConstantNumberOp(step).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenArangeStartOutOp(result0_type, start, end, step, out)
    )


def argmax(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenArgmaxOp(result0_type, self_, dim, keepdim))


def as_strided_copy(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    storage_offset: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(storage_offset, builtins.int):
        storage_offset = torch_dialect.ConstantIntOp(storage_offset).result
    if storage_offset is None:
        storage_offset = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAsStridedCopyOp(
            result0_type, self_, size, stride, storage_offset
        )
    )


def as_strided_scatter(
    self_: Tensor,
    src: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    storage_offset: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(storage_offset, builtins.int):
        storage_offset = torch_dialect.ConstantIntOp(storage_offset).result
    if storage_offset is None:
        storage_offset = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAsStridedScatterOp(
            result0_type, self_, src, size, stride, storage_offset
        )
    )


def atan2(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAtan2Op(result0_type, self_, other))


def atan2_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenAtan2_Op(result0_type, self_, other))


def avg_pool2d(
    self_: Tensor,
    kernel_size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (),
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0, 0),
    ceil_mode: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    count_include_pad: Union[Torch_Value[Torch_BoolType], builtins.bool] = True,
    divisor_override: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(kernel_size, (builtins.list, builtins.tuple)) and builtins.len(
        kernel_size
    ):
        kernel_size = builtins.list(kernel_size)
        for i, a in enumerate(kernel_size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                kernel_size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        kernel_size = torch_dialect.PrimListConstructOp(ls_type, kernel_size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(ceil_mode, builtins.bool):
        ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode).result
    if isinstance(count_include_pad, builtins.bool):
        count_include_pad = torch_dialect.ConstantBoolOp(count_include_pad).result
    if isinstance(divisor_override, builtins.int):
        divisor_override = torch_dialect.ConstantIntOp(divisor_override).result
    if divisor_override is None:
        divisor_override = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenAvgPool2dOp(
            result0_type,
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    )


def baddbmm(
    self_: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    beta: TorchNumber = 1,
    alpha: TorchNumber = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(beta, (builtins.int, builtins.float)):
        beta = torch_dialect.ConstantNumberOp(beta).result
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenBaddbmmOp(result0_type, self_, batch1, batch2, beta, alpha)
    )


def baddbmm_(
    self_: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    beta: TorchNumber = 1,
    alpha: TorchNumber = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(beta, (builtins.int, builtins.float)):
        beta = torch_dialect.ConstantNumberOp(beta).result
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenBaddbmm_Op(result0_type, self_, batch1, batch2, beta, alpha)
    )


def batch_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: Union[Torch_Value[Torch_BoolType], builtins.bool],
    momentum: Union[Torch_Value[Torch_FloatType], builtins.float],
    eps: Union[Torch_Value[Torch_FloatType], builtins.float],
    cudnn_enabled: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if weight is None:
        weight = torch_dialect.ConstantNoneOp().result
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if running_mean is None:
        running_mean = torch_dialect.ConstantNoneOp().result
    if running_var is None:
        running_var = torch_dialect.ConstantNoneOp().result
    if isinstance(training, builtins.bool):
        training = torch_dialect.ConstantBoolOp(training).result
    if isinstance(momentum, builtins.float):
        momentum = torch_dialect.ConstantFloatOp(momentum).result
    if isinstance(eps, builtins.float):
        eps = torch_dialect.ConstantFloatOp(eps).result
    if isinstance(cudnn_enabled, builtins.bool):
        cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenBatchNormOp(
            result0_type,
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            cudnn_enabled,
        )
    )


@register_dispatch
def bernoulli(self_: Tensor, generator: Optional[Torch_GeneratorType] = None) -> Tensor:
    assert check_argument_types()
    if generator is None:
        generator = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBernoulliOp(result0_type, self_, generator))


# overload p
@register_dispatch
def bernoulli(
    self_: Tensor,
    p: Union[Torch_Value[Torch_FloatType], builtins.float],
    generator: Optional[Torch_GeneratorType] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(p, builtins.float):
        p = torch_dialect.ConstantFloatOp(p).result
    if generator is None:
        generator = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBernoulliPOp(result0_type, self_, p, generator))


# overload Tensor
@register_dispatch
def bernoulli(
    self_: Tensor, p: Tensor, generator: Optional[Torch_GeneratorType] = None
) -> Tensor:
    assert check_argument_types()
    if generator is None:
        generator = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenBernoulliTensorOp(result0_type, self_, p, generator)
    )


# overload float
@register_dispatch
def bernoulli_(
    self_: Tensor,
    p: Union[Torch_Value[Torch_FloatType], builtins.float] = 0.5,
    generator: Optional[Torch_GeneratorType] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(p, builtins.float):
        p = torch_dialect.ConstantFloatOp(p).result
    if generator is None:
        generator = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenBernoulli_FloatOp(result0_type, self_, p, generator)
    )


# overload Tensor
@register_dispatch
def bernoulli_(
    self_: Tensor, p: Tensor, generator: Optional[Torch_GeneratorType] = None
) -> Tensor:
    assert check_argument_types()
    if generator is None:
        generator = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenBernoulli_TensorOp(result0_type, self_, p, generator)
    )


def bincount(
    self_: Tensor,
    weights: Optional[Tensor] = None,
    minlength: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
) -> Tensor:
    assert check_argument_types()
    if weights is None:
        weights = torch_dialect.ConstantNoneOp().result
    if isinstance(minlength, builtins.int):
        minlength = torch_dialect.ConstantIntOp(minlength).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBincountOp(result0_type, self_, weights, minlength))


# overload Tensor
def bitwise_and(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBitwiseAndTensorOp(result0_type, self_, other))


# overload Tensor
def bitwise_and_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBitwiseAnd_TensorOp(result0_type, self_, other))


def bitwise_not(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBitwiseNotOp(result0_type, self_))


def bitwise_not_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBitwiseNot_Op(result0_type, self_))


# overload Tensor
def bitwise_or(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBitwiseOrTensorOp(result0_type, self_, other))


# overload Tensor
def bitwise_or_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBitwiseOr_TensorOp(result0_type, self_, other))


# overload Tensor
def bitwise_xor(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBitwiseXorTensorOp(result0_type, self_, other))


# overload Tensor
def bitwise_xor_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBitwiseXor_TensorOp(result0_type, self_, other))


def bmm(self_: Tensor, mat2: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBmmOp(result0_type, self_, mat2))


def broadcast_to(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenBroadcastToOp(result0_type, self_, size))


# overload Tensor
def bucketize(
    self_: Tensor,
    boundaries: Tensor,
    out_int32: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    right: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(out_int32, builtins.bool):
        out_int32 = torch_dialect.ConstantBoolOp(out_int32).result
    if isinstance(right, builtins.bool):
        right = torch_dialect.ConstantBoolOp(right).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenBucketizeTensorOp(
            result0_type, self_, boundaries, out_int32, right
        )
    )


def cat(
    tensors: Union[Sequence[Tensor], Tensor],
    dim: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
) -> Tensor:
    assert check_argument_types()
    if isinstance(tensors, (builtins.list, builtins.tuple)) and builtins.len(tensors):
        assert builtins.all([isinstance(a, Tensor) for a in tensors])
        ls_type = Torch_List.of(Torch_NonValueTensorType())
        tensors = torch_dialect.PrimListConstructOp(ls_type, tensors).result
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCatOp(result0_type, tensors, dim))


@register_dispatch
def ceil(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCeilOp(result0_type, self_))


# overload Scalar
@register_dispatch
def ceil(a: TorchNumber) -> TorchNumber:
    assert check_argument_types()
    if isinstance(a, (builtins.int, builtins.float)):
        a = torch_dialect.ConstantNumberOp(a).result
    result0_type = Torch_NumberType()
    return Torch_Value(torch_dialect.AtenCeilScalarOp(result0_type, a).result)


# overload float
@register_dispatch
def ceil(
    a: Union[Torch_Value[Torch_FloatType], builtins.float]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    return Torch_Value(torch_dialect.AtenCeilFloatOp(a).result)


def ceil_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCeil_Op(result0_type, self_))


@register_dispatch
def clamp(
    self_: Tensor, min: Optional[TorchNumber] = None, max: Optional[TorchNumber] = None
) -> Tensor:
    assert check_argument_types()
    if isinstance(min, (builtins.int, builtins.float)):
        min = torch_dialect.ConstantNumberOp(min).result
    if min is None:
        min = torch_dialect.ConstantNoneOp().result
    if isinstance(max, (builtins.int, builtins.float)):
        max = torch_dialect.ConstantNumberOp(max).result
    if max is None:
        max = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenClampOp(result0_type, self_, min, max))


# overload Tensor
@register_dispatch
def clamp(
    self_: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
) -> Tensor:
    assert check_argument_types()
    if min is None:
        min = torch_dialect.ConstantNoneOp().result
    if max is None:
        max = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenClampTensorOp(result0_type, self_, min, max))


@register_dispatch
def clamp_(
    self_: Tensor, min: Optional[TorchNumber] = None, max: Optional[TorchNumber] = None
) -> Tensor:
    assert check_argument_types()
    if isinstance(min, (builtins.int, builtins.float)):
        min = torch_dialect.ConstantNumberOp(min).result
    if min is None:
        min = torch_dialect.ConstantNoneOp().result
    if isinstance(max, (builtins.int, builtins.float)):
        max = torch_dialect.ConstantNumberOp(max).result
    if max is None:
        max = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenClamp_Op(result0_type, self_, min, max))


# overload Tensor
@register_dispatch
def clamp_(
    self_: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
) -> Tensor:
    assert check_argument_types()
    if min is None:
        min = torch_dialect.ConstantNoneOp().result
    if max is None:
        max = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenClamp_TensorOp(result0_type, self_, min, max))


def clamp_max(self_: Tensor, max: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(max, (builtins.int, builtins.float)):
        max = torch_dialect.ConstantNumberOp(max).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenClampMaxOp(result0_type, self_, max))


def clamp_max_(self_: Tensor, max: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(max, (builtins.int, builtins.float)):
        max = torch_dialect.ConstantNumberOp(max).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenClampMax_Op(result0_type, self_, max))


def clamp_min(self_: Tensor, min: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(min, (builtins.int, builtins.float)):
        min = torch_dialect.ConstantNumberOp(min).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenClampMinOp(result0_type, self_, min))


def clamp_min_(self_: Tensor, min: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(min, (builtins.int, builtins.float)):
        min = torch_dialect.ConstantNumberOp(min).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenClampMin_Op(result0_type, self_, min))


def clone(
    self_: Tensor,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCloneOp(result0_type, self_, memory_format))


def constant_pad_nd(
    self_: Tensor,
    pad: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    value: TorchNumber = 0,
) -> Tensor:
    assert check_argument_types()
    if isinstance(pad, (builtins.list, builtins.tuple)) and builtins.len(pad):
        pad = builtins.list(pad)
        for i, a in enumerate(pad):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                pad[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        pad = torch_dialect.PrimListConstructOp(ls_type, pad).result
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenConstantPadNdOp(result0_type, self_, pad, value))


def contiguous(
    self_: Tensor,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format
    ] = 0,
) -> Tensor:
    assert check_argument_types()
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenContiguousOp(result0_type, self_, memory_format))


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1, 1),
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0, 0),
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1, 1),
    groups: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenConv2dOp(
            result0_type, input, weight, bias, stride, padding, dilation, groups
        )
    )


def conv_transpose1d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1),
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0),
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0),
    groups: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1),
) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenConvTranspose1dOp(
            result0_type,
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )
    )


# overload input
def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1, 1),
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0, 0),
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0, 0),
    groups: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1, 1),
) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenConvTranspose2dInputOp(
            result0_type,
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )
    )


# overload input
def conv_transpose3d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1, 1, 1),
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0, 0, 0),
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0, 0, 0),
    groups: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1, 1, 1),
) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenConvTranspose3dInputOp(
            result0_type,
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )
    )


def convert_element_type(
    a: Tensor, dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype]
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.PrimsConvertElementTypeOp(result0_type, a, dtype))


def convolution(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    transposed: Union[Torch_Value[Torch_BoolType], builtins.bool],
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    groups: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(transposed, builtins.bool):
        transposed = torch_dialect.ConstantBoolOp(transposed).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenConvolutionOp(
            result0_type,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
    )


def convolution_backward(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    bias_sizes: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    transposed: Union[Torch_Value[Torch_BoolType], builtins.bool],
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    groups: Union[Torch_Value[Torch_IntType], builtins.int],
    output_mask: Union[
        Sequence[Union[Torch_Value[Torch_BoolType], builtins.bool]],
        Torch_List[Torch_BoolType],
    ],
) -> Tuple[Tensor, Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(bias_sizes, (builtins.list, builtins.tuple)) and builtins.len(
        bias_sizes
    ):
        bias_sizes = builtins.list(bias_sizes)
        for i, a in enumerate(bias_sizes):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                bias_sizes[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        bias_sizes = torch_dialect.PrimListConstructOp(ls_type, bias_sizes).result
    if bias_sizes is None:
        bias_sizes = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(transposed, builtins.bool):
        transposed = torch_dialect.ConstantBoolOp(transposed).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    if isinstance(output_mask, (builtins.list, builtins.tuple)) and builtins.len(
        output_mask
    ):
        output_mask = builtins.list(output_mask)
        for i, a in enumerate(output_mask):
            if not isinstance(a, builtins.bool):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be bool"
            else:
                output_mask[i] = torch_dialect.ConstantBoolOp(a).result
        ls_type = Torch_List.of(Torch_BoolType())
        output_mask = torch_dialect.PrimListConstructOp(ls_type, output_mask).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    result2_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenConvolutionBackwardOp(
            result0_type,
            result1_type,
            result2_type,
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
    )
    return tuple([Tensor(o) for o in op_results])


def convolution_backward_overrideable(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    transposed: Union[Torch_Value[Torch_BoolType], builtins.bool],
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    groups: Union[Torch_Value[Torch_IntType], builtins.int],
    output_mask: Union[
        Sequence[Union[Torch_Value[Torch_BoolType], builtins.bool]],
        Torch_List[Torch_BoolType],
    ],
) -> Tuple[Tensor, Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(transposed, builtins.bool):
        transposed = torch_dialect.ConstantBoolOp(transposed).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    if isinstance(output_mask, (builtins.list, builtins.tuple)) and builtins.len(
        output_mask
    ):
        output_mask = builtins.list(output_mask)
        for i, a in enumerate(output_mask):
            if not isinstance(a, builtins.bool):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be bool"
            else:
                output_mask[i] = torch_dialect.ConstantBoolOp(a).result
        ls_type = Torch_List.of(Torch_BoolType())
        output_mask = torch_dialect.PrimListConstructOp(ls_type, output_mask).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    result2_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenConvolutionBackwardOverrideableOp(
            result0_type,
            result1_type,
            result2_type,
            grad_output,
            input,
            weight,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
    )
    return tuple([Tensor(o) for o in op_results])


def convolution_overrideable(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    transposed: Union[Torch_Value[Torch_BoolType], builtins.bool],
    output_padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    groups: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(transposed, builtins.bool):
        transposed = torch_dialect.ConstantBoolOp(transposed).result
    if isinstance(output_padding, (builtins.list, builtins.tuple)) and builtins.len(
        output_padding
    ):
        output_padding = builtins.list(output_padding)
        for i, a in enumerate(output_padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_padding = torch_dialect.PrimListConstructOp(
            ls_type, output_padding
        ).result
    if isinstance(groups, builtins.int):
        groups = torch_dialect.ConstantIntOp(groups).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenConvolutionOverrideableOp(
            result0_type,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
    )


def copy(
    self_: Tensor,
    src: Tensor,
    non_blocking: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(non_blocking, builtins.bool):
        non_blocking = torch_dialect.ConstantBoolOp(non_blocking).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCopyOp(result0_type, self_, src, non_blocking))


def copy_(
    self_: Tensor,
    src: Tensor,
    non_blocking: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(non_blocking, builtins.bool):
        non_blocking = torch_dialect.ConstantBoolOp(non_blocking).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCopy_Op(result0_type, self_, src, non_blocking))


def cos(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCosOp(result0_type, self_))


def cos_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCos_Op(result0_type, self_))


def cpu(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCpuOp(result0_type, self_))


def cumsum(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenCumsumOp(result0_type, self_, dim, dtype))


def detach(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenDetachOp(result0_type, self_))


def detach_copy(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenDetachCopyOp(result0_type, self_))


def device(a: Tensor) -> Union[Torch_Value[Torch_DeviceType], builtins.str]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.PrimDeviceOp(a).result)


def diagonal_copy(
    self_: Tensor,
    offset: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    dim1: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    dim2: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(offset, builtins.int):
        offset = torch_dialect.ConstantIntOp(offset).result
    if isinstance(dim1, builtins.int):
        dim1 = torch_dialect.ConstantIntOp(dim1).result
    if isinstance(dim2, builtins.int):
        dim2 = torch_dialect.ConstantIntOp(dim2).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenDiagonalCopyOp(result0_type, self_, offset, dim1, dim2)
    )


def diagonal_scatter(
    self_: Tensor,
    src: Tensor,
    offset: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    dim1: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    dim2: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(offset, builtins.int):
        offset = torch_dialect.ConstantIntOp(offset).result
    if isinstance(dim1, builtins.int):
        dim1 = torch_dialect.ConstantIntOp(dim1).result
    if isinstance(dim2, builtins.int):
        dim2 = torch_dialect.ConstantIntOp(dim2).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenDiagonalScatterOp(
            result0_type, self_, src, offset, dim1, dim2
        )
    )


def dim(self_: Tensor) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenDimOp(self_).result)


# overload Tensor
@register_dispatch
def div(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenDivTensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def div(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenDivScalarOp(result0_type, self_, other))


# overload Tensor_mode
@register_dispatch
def div(
    self_: Tensor,
    other: Tensor,
    rounding_mode: Union[Torch_Value[Torch_StringType], builtins.str, None],
) -> Tensor:
    assert check_argument_types()
    if isinstance(rounding_mode, builtins.str):
        rounding_mode = torch_dialect.ConstantStrOp(rounding_mode).result
    if rounding_mode is None:
        rounding_mode = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenDivTensorModeOp(result0_type, self_, other, rounding_mode)
    )


# overload int
@register_dispatch
def div(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenDivIntOp(a, b).result)


# overload float
@register_dispatch
def div(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.float):
        b = torch_dialect.ConstantFloatOp(b).result
    return Torch_Value(torch_dialect.AtenDivFloatOp(a, b).result)


@register_dispatch
def div(
    a: TorchNumber, b: TorchNumber
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, (builtins.int, builtins.float)):
        a = torch_dialect.ConstantNumberOp(a).result
    if isinstance(b, (builtins.int, builtins.float)):
        b = torch_dialect.ConstantNumberOp(b).result
    return Torch_Value(torch_dialect.AtenDivOp(a, b).result)


# overload Tensor
@register_dispatch
def div_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenDiv_TensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def div_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenDiv_ScalarOp(result0_type, self_, other))


# overload Tensor_mode
@register_dispatch
def div_(
    self_: Tensor,
    other: Tensor,
    rounding_mode: Union[Torch_Value[Torch_StringType], builtins.str, None],
) -> Tensor:
    assert check_argument_types()
    if isinstance(rounding_mode, builtins.str):
        rounding_mode = torch_dialect.ConstantStrOp(rounding_mode).result
    if rounding_mode is None:
        rounding_mode = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenDiv_TensorModeOp(result0_type, self_, other, rounding_mode)
    )


def dropout(
    input: Tensor,
    p: Union[Torch_Value[Torch_FloatType], builtins.float],
    train: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if isinstance(p, builtins.float):
        p = torch_dialect.ConstantFloatOp(p).result
    if isinstance(train, builtins.bool):
        train = torch_dialect.ConstantBoolOp(train).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenDropoutOp(result0_type, input, p, train))


def dropout_(
    self_: Tensor,
    p: Union[Torch_Value[Torch_FloatType], builtins.float],
    train: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if isinstance(p, builtins.float):
        p = torch_dialect.ConstantFloatOp(p).result
    if isinstance(train, builtins.bool):
        train = torch_dialect.ConstantBoolOp(train).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenDropout_Op(result0_type, self_, p, train))


def dtype(a: Tensor) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.PrimDtypeOp(a).result)


def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: Union[Torch_Value[Torch_IntType], builtins.int] = -1,
    scale_grad_by_freq: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    sparse: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(padding_idx, builtins.int):
        padding_idx = torch_dialect.ConstantIntOp(padding_idx).result
    if isinstance(scale_grad_by_freq, builtins.bool):
        scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq).result
    if isinstance(sparse, builtins.bool):
        sparse = torch_dialect.ConstantBoolOp(sparse).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenEmbeddingOp(
            result0_type, weight, indices, padding_idx, scale_grad_by_freq, sparse
        )
    )


# overload padding_idx
def embedding_bag(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: Union[Torch_Value[Torch_BoolType], builtins.bool],
    mode: Union[Torch_Value[Torch_IntType], builtins.int],
    sparse: Union[Torch_Value[Torch_BoolType], builtins.bool],
    per_sample_weights: Optional[Tensor],
    include_last_offset: Union[Torch_Value[Torch_BoolType], builtins.bool],
    padding_idx: Union[Torch_Value[Torch_IntType], builtins.int, None],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(scale_grad_by_freq, builtins.bool):
        scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq).result
    if isinstance(mode, builtins.int):
        mode = torch_dialect.ConstantIntOp(mode).result
    if isinstance(sparse, builtins.bool):
        sparse = torch_dialect.ConstantBoolOp(sparse).result
    if per_sample_weights is None:
        per_sample_weights = torch_dialect.ConstantNoneOp().result
    if isinstance(include_last_offset, builtins.bool):
        include_last_offset = torch_dialect.ConstantBoolOp(include_last_offset).result
    if isinstance(padding_idx, builtins.int):
        padding_idx = torch_dialect.ConstantIntOp(padding_idx).result
    if padding_idx is None:
        padding_idx = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    result2_type = Torch_NonValueTensorType()
    result3_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenEmbeddingBagPaddingIdxOp(
            result0_type,
            result1_type,
            result2_type,
            result3_type,
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
    return tuple([Tensor(o) for o in op_results])


def embedding_dense_backward(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: Union[Torch_Value[Torch_IntType], builtins.int],
    padding_idx: Union[Torch_Value[Torch_IntType], builtins.int],
    scale_grad_by_freq: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if isinstance(num_weights, builtins.int):
        num_weights = torch_dialect.ConstantIntOp(num_weights).result
    if isinstance(padding_idx, builtins.int):
        padding_idx = torch_dialect.ConstantIntOp(padding_idx).result
    if isinstance(scale_grad_by_freq, builtins.bool):
        scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenEmbeddingDenseBackwardOp(
            result0_type,
            grad_output,
            indices,
            num_weights,
            padding_idx,
            scale_grad_by_freq,
        )
    )


# overload memory_format
def empty(
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenEmptyMemoryFormatOp(
            result0_type, size, dtype, layout, device, pin_memory, memory_format
        )
    )


def empty_like(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenEmptyLikeOp(
            result0_type, self_, dtype, layout, device, pin_memory, memory_format
        )
    )


# overload Tensor
@register_dispatch
def eq(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenEqTensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def eq(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenEqScalarOp(result0_type, self_, other))


# overload int_list
@register_dispatch
def eq(
    a: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    b: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, (builtins.list, builtins.tuple)) and builtins.len(a):
        a = builtins.list(a)
        for i, a in enumerate(a):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                a[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        a = torch_dialect.PrimListConstructOp(ls_type, a).result
    if isinstance(b, (builtins.list, builtins.tuple)) and builtins.len(b):
        b = builtins.list(b)
        for i, a in enumerate(b):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                b[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        b = torch_dialect.PrimListConstructOp(ls_type, b).result
    return Torch_Value(torch_dialect.AtenEqIntListOp(a, b).result)


# overload str
@register_dispatch
def eq(
    a: Union[Torch_Value[Torch_StringType], builtins.str],
    b: Union[Torch_Value[Torch_StringType], builtins.str],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.str):
        a = torch_dialect.ConstantStrOp(a).result
    if isinstance(b, builtins.str):
        b = torch_dialect.ConstantStrOp(b).result
    return Torch_Value(torch_dialect.AtenEqStrOp(a, b).result)


# overload int
@register_dispatch
def eq(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenEqIntOp(a, b).result)


# overload float
@register_dispatch
def eq(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.float):
        b = torch_dialect.ConstantFloatOp(b).result
    return Torch_Value(torch_dialect.AtenEqFloatOp(a, b).result)


# overload device
@register_dispatch
def eq(
    a: Union[Torch_Value[Torch_DeviceType], builtins.str],
    b: Union[Torch_Value[Torch_DeviceType], builtins.str],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenEqDeviceOp(a, b).result)


# overload Tensor
@register_dispatch
def eq_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenEq_TensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def eq_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenEq_ScalarOp(result0_type, self_, other))


def erf(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenErfOp(result0_type, self_))


def erf_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenErf_Op(result0_type, self_))


def exp(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenExpOp(result0_type, self_))


def exp_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenExp_Op(result0_type, self_))


def expand(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    implicit: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(implicit, builtins.bool):
        implicit = torch_dialect.ConstantBoolOp(implicit).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenExpandOp(result0_type, self_, size, implicit))


def expand_as(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenExpandAsOp(result0_type, self_, other))


def expand_copy(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    implicit: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(implicit, builtins.bool):
        implicit = torch_dialect.ConstantBoolOp(implicit).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenExpandCopyOp(result0_type, self_, size, implicit))


def expm1(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenExpm1Op(result0_type, self_))


def expm1_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenExpm1_Op(result0_type, self_))


def fft_fft(
    self_: Tensor,
    n: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    dim: Union[Torch_Value[Torch_IntType], builtins.int] = -1,
    norm: Union[Torch_Value[Torch_StringType], builtins.str, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(n, builtins.int):
        n = torch_dialect.ConstantIntOp(n).result
    if n is None:
        n = torch_dialect.ConstantNoneOp().result
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(norm, builtins.str):
        norm = torch_dialect.ConstantStrOp(norm).result
    if norm is None:
        norm = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFftFftOp(result0_type, self_, n, dim, norm))


# overload Scalar
@register_dispatch
def fill(self_: Tensor, value: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFillScalarOp(result0_type, self_, value))


# overload Tensor
@register_dispatch
def fill(self_: Tensor, value: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFillTensorOp(result0_type, self_, value))


# overload Scalar
@register_dispatch
def fill_(self_: Tensor, value: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFill_ScalarOp(result0_type, self_, value))


# overload Tensor
@register_dispatch
def fill_(self_: Tensor, value: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFill_TensorOp(result0_type, self_, value))


# overload using_ints
def flatten(
    self_: Tensor,
    start_dim: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    end_dim: Union[Torch_Value[Torch_IntType], builtins.int] = -1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(start_dim, builtins.int):
        start_dim = torch_dialect.ConstantIntOp(start_dim).result
    if isinstance(end_dim, builtins.int):
        end_dim = torch_dialect.ConstantIntOp(end_dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenFlattenUsingIntsOp(result0_type, self_, start_dim, end_dim)
    )


def flip(
    self_: Tensor,
    dims: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dims, (builtins.list, builtins.tuple)) and builtins.len(dims):
        dims = builtins.list(dims)
        for i, a in enumerate(dims):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dims[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dims = torch_dialect.PrimListConstructOp(ls_type, dims).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFlipOp(result0_type, self_, dims))


def floor(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFloorOp(result0_type, self_))


def floor_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFloor_Op(result0_type, self_))


@register_dispatch
def floor_divide(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFloorDivideOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def floor_divide(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFloorDivideScalarOp(result0_type, self_, other))


# overload int
def floordiv(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenFloordivIntOp(a, b).result)


# overload Scalar
def fmod(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFmodScalarOp(result0_type, self_, other))


# overload Scalar
def fmod_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenFmod_ScalarOp(result0_type, self_, other))


# overload dim
def frobenius_norm(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenFrobeniusNormDimOp(result0_type, self_, dim, keepdim)
    )


def full(
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    fill_value: TorchNumber,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(fill_value, (builtins.int, builtins.float)):
        fill_value = torch_dialect.ConstantNumberOp(fill_value).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenFullOp(
            result0_type, size, fill_value, dtype, layout, device, pin_memory
        )
    )


def full_like(
    self_: Tensor,
    fill_value: TorchNumber,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(fill_value, (builtins.int, builtins.float)):
        fill_value = torch_dialect.ConstantNumberOp(fill_value).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenFullLikeOp(
            result0_type,
            self_,
            fill_value,
            dtype,
            layout,
            device,
            pin_memory,
            memory_format,
        )
    )


def gather(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    index: Tensor,
    sparse_grad: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(sparse_grad, builtins.bool):
        sparse_grad = torch_dialect.ConstantBoolOp(sparse_grad).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenGatherOp(result0_type, self_, dim, index, sparse_grad)
    )


# overload Tensor
@register_dispatch
def ge(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGeTensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def ge(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGeScalarOp(result0_type, self_, other))


# overload int
@register_dispatch
def ge(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenGeIntOp(a, b).result)


# overload float
@register_dispatch
def ge(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.float):
        b = torch_dialect.ConstantFloatOp(b).result
    return Torch_Value(torch_dialect.AtenGeFloatOp(a, b).result)


# overload float_int
@register_dispatch
def ge(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenGeFloatIntOp(a, b).result)


# overload Tensor
@register_dispatch
def ge_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGe_TensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def ge_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGe_ScalarOp(result0_type, self_, other))


def gelu(
    self_: Tensor,
    approximate: Union[Torch_Value[Torch_StringType], builtins.str] = "none",
) -> Tensor:
    assert check_argument_types()
    if isinstance(approximate, builtins.str):
        approximate = torch_dialect.ConstantStrOp(approximate).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGeluOp(result0_type, self_, approximate))


def gelu_backward(
    grad_output: Tensor,
    self_: Tensor,
    approximate: Union[Torch_Value[Torch_StringType], builtins.str] = "none",
) -> Tensor:
    assert check_argument_types()
    if isinstance(approximate, builtins.str):
        approximate = torch_dialect.ConstantStrOp(approximate).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenGeluBackwardOp(result0_type, grad_output, self_, approximate)
    )


# overload default_str
def get(
    self_: Torch_Dict,
    key: Union[Torch_Value[Torch_StringType], builtins.str],
    default_value: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
) -> Union[
    TorchNumber,
    Tensor,
    Torch_Value[Torch_AnyType],
    Any,
    Torch_Value[Torch_BoolType],
    builtins.bool,
    Torch_Dict,
    Torch_Value[Torch_DeviceType],
    builtins.str,
    Torch_GeneratorType,
    Torch_List,
    None,
    Torch_Value[Torch_StringType],
    Tuple,
]:
    assert check_argument_types()
    if isinstance(key, builtins.str):
        key = torch_dialect.ConstantStrOp(key).result
    result0_type = Torch_AnyType()
    return Torch_Value(
        torch_dialect.AtenGetDefaultStrOp(
            result0_type, self_, key, default_value
        ).result
    )


# overload Tensor
@register_dispatch
def gt(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGtTensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def gt(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGtScalarOp(result0_type, self_, other))


# overload int
@register_dispatch
def gt(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenGtIntOp(a, b).result)


# overload float
@register_dispatch
def gt(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.float):
        b = torch_dialect.ConstantFloatOp(b).result
    return Torch_Value(torch_dialect.AtenGtFloatOp(a, b).result)


# overload float_int
@register_dispatch
def gt(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenGtFloatIntOp(a, b).result)


# overload Tensor
@register_dispatch
def gt_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGt_TensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def gt_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenGt_ScalarOp(result0_type, self_, other))


def hardsigmoid(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenHardsigmoidOp(result0_type, self_))


def hardsigmoid_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenHardsigmoid_Op(result0_type, self_))


def hardswish(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenHardswishOp(result0_type, self_))


def hardswish_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenHardswish_Op(result0_type, self_))


def hardtanh(
    self_: Tensor, min_val: TorchNumber = -1, max_val: TorchNumber = 1
) -> Tensor:
    assert check_argument_types()
    if isinstance(min_val, (builtins.int, builtins.float)):
        min_val = torch_dialect.ConstantNumberOp(min_val).result
    if isinstance(max_val, (builtins.int, builtins.float)):
        max_val = torch_dialect.ConstantNumberOp(max_val).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenHardtanhOp(result0_type, self_, min_val, max_val))


def hardtanh_(
    self_: Tensor, min_val: TorchNumber = -1, max_val: TorchNumber = 1
) -> Tensor:
    assert check_argument_types()
    if isinstance(min_val, (builtins.int, builtins.float)):
        min_val = torch_dialect.ConstantNumberOp(min_val).result
    if isinstance(max_val, (builtins.int, builtins.float)):
        max_val = torch_dialect.ConstantNumberOp(max_val).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenHardtanh_Op(result0_type, self_, min_val, max_val))


def hardtanh_backward(
    grad_output: Tensor, self_: Tensor, min_val: TorchNumber, max_val: TorchNumber
) -> Tensor:
    assert check_argument_types()
    if isinstance(min_val, (builtins.int, builtins.float)):
        min_val = torch_dialect.ConstantNumberOp(min_val).result
    if isinstance(max_val, (builtins.int, builtins.float)):
        max_val = torch_dialect.ConstantNumberOp(max_val).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenHardtanhBackwardOp(
            result0_type, grad_output, self_, min_val, max_val
        )
    )


# overload Tensor
@register_dispatch
def index(
    self_: Tensor, indices: Union[Sequence[Optional[Tensor]], Tensor, None]
) -> Tensor:
    assert check_argument_types()
    indices = builtins.list(indices)
    for i, a in enumerate(indices):
        if a is not None:
            assert isinstance(a, Tensor)
        else:
            indices[i] = torch_dialect.ConstantNoneOp().result
    indices = torch_dialect.PrimListConstructOp(
        TorchListOfOptionalTensorType(), indices
    ).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenIndexTensorOp(result0_type, self_, indices))


# overload Tensor_hacked_twin
@register_dispatch
def index(self_: Tensor, indices: Union[Sequence[Tensor], Tensor]) -> Tensor:
    assert check_argument_types()
    if isinstance(indices, (builtins.list, builtins.tuple)) and builtins.len(indices):
        assert builtins.all([isinstance(a, Tensor) for a in indices])
        ls_type = Torch_List.of(Torch_NonValueTensorType())
        indices = torch_dialect.PrimListConstructOp(ls_type, indices).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenIndexTensorHackedTwinOp(result0_type, self_, indices)
    )


@register_dispatch
def index_put(
    self_: Tensor,
    indices: Union[Sequence[Optional[Tensor]], Tensor, None],
    values: Tensor,
    accumulate: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    indices = builtins.list(indices)
    for i, a in enumerate(indices):
        if a is not None:
            assert isinstance(a, Tensor)
        else:
            indices[i] = torch_dialect.ConstantNoneOp().result
    indices = torch_dialect.PrimListConstructOp(
        TorchListOfOptionalTensorType(), indices
    ).result
    if isinstance(accumulate, builtins.bool):
        accumulate = torch_dialect.ConstantBoolOp(accumulate).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenIndexPutOp(result0_type, self_, indices, values, accumulate)
    )


# overload hacked_twin
@register_dispatch
def index_put(
    self_: Tensor,
    indices: Union[Sequence[Tensor], Tensor],
    values: Tensor,
    accumulate: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(indices, (builtins.list, builtins.tuple)) and builtins.len(indices):
        assert builtins.all([isinstance(a, Tensor) for a in indices])
        ls_type = Torch_List.of(Torch_NonValueTensorType())
        indices = torch_dialect.PrimListConstructOp(ls_type, indices).result
    if isinstance(accumulate, builtins.bool):
        accumulate = torch_dialect.ConstantBoolOp(accumulate).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenIndexPutHackedTwinOp(
            result0_type, self_, indices, values, accumulate
        )
    )


@register_dispatch
def index_put_(
    self_: Tensor,
    indices: Union[Sequence[Optional[Tensor]], Tensor, None],
    values: Tensor,
    accumulate: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    indices = builtins.list(indices)
    for i, a in enumerate(indices):
        if a is not None:
            assert isinstance(a, Tensor)
        else:
            indices[i] = torch_dialect.ConstantNoneOp().result
    indices = torch_dialect.PrimListConstructOp(
        TorchListOfOptionalTensorType(), indices
    ).result
    if isinstance(accumulate, builtins.bool):
        accumulate = torch_dialect.ConstantBoolOp(accumulate).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenIndexPut_Op(result0_type, self_, indices, values, accumulate)
    )


# overload hacked_twin
@register_dispatch
def index_put_(
    self_: Tensor,
    indices: Union[Sequence[Tensor], Tensor],
    values: Tensor,
    accumulate: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(indices, (builtins.list, builtins.tuple)) and builtins.len(indices):
        assert builtins.all([isinstance(a, Tensor) for a in indices])
        ls_type = Torch_List.of(Torch_NonValueTensorType())
        indices = torch_dialect.PrimListConstructOp(ls_type, indices).result
    if isinstance(accumulate, builtins.bool):
        accumulate = torch_dialect.ConstantBoolOp(accumulate).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenIndexPut_HackedTwinOp(
            result0_type, self_, indices, values, accumulate
        )
    )


def index_select(
    self_: Tensor, dim: Union[Torch_Value[Torch_IntType], builtins.int], index: Tensor
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenIndexSelectOp(result0_type, self_, dim, index))


# overload t
def insert(
    self_: Union[Sequence[Any], Any],
    idx: Union[Torch_Value[Torch_IntType], builtins.int],
    el: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ],
) -> None:
    assert check_argument_types()
    if isinstance(idx, builtins.int):
        idx = torch_dialect.ConstantIntOp(idx).result
    torch_dialect.AtenInsertTOp(self_, idx, el)


def is_floating_point(
    self_: Tensor,
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenIsFloatingPointOp(self_).result)


def item(self_: Tensor) -> TorchNumber:
    assert check_argument_types()
    result0_type = Torch_NumberType()
    return Torch_Value(torch_dialect.AtenItemOp(result0_type, self_).result)


def join(
    self_: Union[Torch_Value[Torch_StringType], builtins.str],
    values: Union[
        Sequence[Union[Torch_Value[Torch_StringType], builtins.str]],
        Torch_List[Torch_StringType],
    ],
) -> Union[Torch_Value[Torch_StringType], builtins.str]:
    assert check_argument_types()
    if isinstance(self_, builtins.str):
        self_ = torch_dialect.ConstantStrOp(self_).result
    if isinstance(values, (builtins.list, builtins.tuple)) and builtins.len(values):
        values = builtins.list(values)
        for i, a in enumerate(values):
            if not isinstance(a, builtins.str):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be str"
            else:
                values[i] = torch_dialect.ConstantStrOp(a).result
        ls_type = Torch_List.of(Torch_StringType())
        values = torch_dialect.PrimListConstructOp(ls_type, values).result
    return Torch_Value(torch_dialect.AtenJoinOp(self_, values).result)


# overload str
def keys(
    self_: Torch_Dict,
) -> Union[
    Sequence[Union[Torch_Value[Torch_StringType], builtins.str]],
    Torch_List[Torch_StringType],
]:
    assert check_argument_types()
    result0_type = Torch_List.of(Torch_StringType())
    return Torch_Value(torch_dialect.AtenKeysStrOp(result0_type, self_).result)


def layer_norm(
    input: Tensor,
    normalized_shape: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: Union[Torch_Value[Torch_FloatType], builtins.float] = 1.0000000000000001e-05,
    cudnn_enable: Union[Torch_Value[Torch_BoolType], builtins.bool] = True,
) -> Tensor:
    assert check_argument_types()
    if isinstance(normalized_shape, (builtins.list, builtins.tuple)) and builtins.len(
        normalized_shape
    ):
        normalized_shape = builtins.list(normalized_shape)
        for i, a in enumerate(normalized_shape):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                normalized_shape[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        normalized_shape = torch_dialect.PrimListConstructOp(
            ls_type, normalized_shape
        ).result
    if weight is None:
        weight = torch_dialect.ConstantNoneOp().result
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(eps, builtins.float):
        eps = torch_dialect.ConstantFloatOp(eps).result
    if isinstance(cudnn_enable, builtins.bool):
        cudnn_enable = torch_dialect.ConstantBoolOp(cudnn_enable).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenLayerNormOp(
            result0_type, input, normalized_shape, weight, bias, eps, cudnn_enable
        )
    )


def layout(a: Tensor) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.PrimLayoutOp(a).result)


# overload Tensor
@register_dispatch
def le(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLeTensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def le(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLeScalarOp(result0_type, self_, other))


# overload int
@register_dispatch
def le(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenLeIntOp(a, b).result)


# overload Tensor
@register_dispatch
def le_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLe_TensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def le_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLe_ScalarOp(result0_type, self_, other))


def leaky_relu(self_: Tensor, negative_slope: TorchNumber = 0.01) -> Tensor:
    assert check_argument_types()
    if isinstance(negative_slope, (builtins.int, builtins.float)):
        negative_slope = torch_dialect.ConstantNumberOp(negative_slope).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLeakyReluOp(result0_type, self_, negative_slope))


def leaky_relu_(self_: Tensor, negative_slope: TorchNumber = 0.01) -> Tensor:
    assert check_argument_types()
    if isinstance(negative_slope, (builtins.int, builtins.float)):
        negative_slope = torch_dialect.ConstantNumberOp(negative_slope).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLeakyRelu_Op(result0_type, self_, negative_slope))


def leaky_relu_backward(
    grad_output: Tensor,
    self_: Tensor,
    negative_slope: TorchNumber,
    self_is_result: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Tensor:
    assert check_argument_types()
    if isinstance(negative_slope, (builtins.int, builtins.float)):
        negative_slope = torch_dialect.ConstantNumberOp(negative_slope).result
    if isinstance(self_is_result, builtins.bool):
        self_is_result = torch_dialect.ConstantBoolOp(self_is_result).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenLeakyReluBackwardOp(
            result0_type, grad_output, self_, negative_slope, self_is_result
        )
    )


# overload Tensor
@register_dispatch
def len(t: Tensor) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenLenTensorOp(t).result)


# overload str
@register_dispatch
def len(
    s: Union[Torch_Value[Torch_StringType], builtins.str]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(s, builtins.str):
        s = torch_dialect.ConstantStrOp(s).result
    return Torch_Value(torch_dialect.AtenLenStrOp(s).result)


# overload t
@register_dispatch
def len(
    a: Union[Sequence[Any], Any]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenLenTOp(a).result)


# overload Tensor
def lerp(self_: Tensor, end: Tensor, weight: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLerpTensorOp(result0_type, self_, end, weight))


# overload Tensor
def lerp_(self_: Tensor, end: Tensor, weight: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLerp_TensorOp(result0_type, self_, end, weight))


def lift_fresh_copy(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLiftFreshCopyOp(result0_type, self_))


def vector_norm(
    self_: Tensor,
    ord: TorchNumber = 2,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ] = None,
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(ord, (builtins.int, builtins.float)):
        ord = torch_dialect.ConstantNumberOp(ord).result
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenLinalgVectorNormOp(
            result0_type, self_, ord, dim, keepdim, dtype
        )
    )


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    assert check_argument_types()
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLinearOp(result0_type, input, weight, bias))


# overload t
def list(l: Union[Sequence[Any], Any]) -> Union[Sequence[Any], Any]:
    assert check_argument_types()
    result0_type = Torch_List.of(Torch_AnyType())
    return Torch_Value(torch_dialect.AtenListTOp(result0_type, l).result)


@register_dispatch
def log(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogOp(result0_type, self_))


# overload int
@register_dispatch
def log(
    a: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    return Torch_Value(torch_dialect.AtenLogIntOp(a).result)


def log1p(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLog1pOp(result0_type, self_))


def log1p_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLog1p_Op(result0_type, self_))


def log2(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLog2Op(result0_type, self_))


def log2_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLog2_Op(result0_type, self_))


def log_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLog_Op(result0_type, self_))


# overload int
def log_softmax(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogSoftmaxIntOp(result0_type, self_, dim, dtype))


def logical_and(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogicalAndOp(result0_type, self_, other))


def logical_and_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogicalAnd_Op(result0_type, self_, other))


def logical_not(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogicalNotOp(result0_type, self_))


def logical_not_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogicalNot_Op(result0_type, self_))


def logical_or(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogicalOrOp(result0_type, self_, other))


def logical_or_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogicalOr_Op(result0_type, self_, other))


def logical_xor(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogicalXorOp(result0_type, self_, other))


def logical_xor_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogicalXor_Op(result0_type, self_, other))


def logsumexp(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLogsumexpOp(result0_type, self_, dim, keepdim))


# overload Tensor
@register_dispatch
def lt(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLtTensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def lt(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLtScalarOp(result0_type, self_, other))


# overload int
@register_dispatch
def lt(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenLtIntOp(a, b).result)


# overload float
@register_dispatch
def lt(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.float):
        b = torch_dialect.ConstantFloatOp(b).result
    return Torch_Value(torch_dialect.AtenLtFloatOp(a, b).result)


# overload float_int
@register_dispatch
def lt(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenLtFloatIntOp(a, b).result)


# overload Tensor
@register_dispatch
def lt_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLt_TensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def lt_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenLt_ScalarOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def masked_fill(self_: Tensor, mask: Tensor, value: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenMaskedFillScalarOp(result0_type, self_, mask, value)
    )


# overload Tensor
@register_dispatch
def masked_fill(self_: Tensor, mask: Tensor, value: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenMaskedFillTensorOp(result0_type, self_, mask, value)
    )


# overload Scalar
@register_dispatch
def masked_fill_(self_: Tensor, mask: Tensor, value: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenMaskedFill_ScalarOp(result0_type, self_, mask, value)
    )


# overload Tensor
@register_dispatch
def masked_fill_(self_: Tensor, mask: Tensor, value: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenMaskedFill_TensorOp(result0_type, self_, mask, value)
    )


def masked_select(self_: Tensor, mask: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMaskedSelectOp(result0_type, self_, mask))


def matmul(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMatmulOp(result0_type, self_, other))


@register_dispatch
def max(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMaxOp(result0_type, self_))


# overload dim
@register_dispatch
def max(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tuple[Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenMaxDimOp(result0_type, result1_type, self_, dim, keepdim)
    )
    return tuple([Tensor(o) for o in op_results])


# overload self_int
@register_dispatch
def max(
    self_: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(self_, (builtins.list, builtins.tuple)) and builtins.len(self_):
        self_ = builtins.list(self_)
        for i, a in enumerate(self_):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                self_[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        self_ = torch_dialect.PrimListConstructOp(ls_type, self_).result
    return Torch_Value(torch_dialect.PrimMaxSelfIntOp(self_).result)


# overload int
@register_dispatch
def max(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.PrimMaxIntOp(a, b).result)


def max_pool2d(
    self_: Tensor,
    kernel_size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (),
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0, 0),
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1, 1),
    ceil_mode: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(kernel_size, (builtins.list, builtins.tuple)) and builtins.len(
        kernel_size
    ):
        kernel_size = builtins.list(kernel_size)
        for i, a in enumerate(kernel_size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                kernel_size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        kernel_size = torch_dialect.PrimListConstructOp(ls_type, kernel_size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(ceil_mode, builtins.bool):
        ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenMaxPool2dOp(
            result0_type, self_, kernel_size, stride, padding, dilation, ceil_mode
        )
    )


def max_pool2d_with_indices(
    self_: Tensor,
    kernel_size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (),
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (0, 0),
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (1, 1),
    ceil_mode: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tuple[Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(kernel_size, (builtins.list, builtins.tuple)) and builtins.len(
        kernel_size
    ):
        kernel_size = builtins.list(kernel_size)
        for i, a in enumerate(kernel_size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                kernel_size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        kernel_size = torch_dialect.PrimListConstructOp(ls_type, kernel_size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(ceil_mode, builtins.bool):
        ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenMaxPool2dWithIndicesOp(
            result0_type,
            result1_type,
            self_,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
    )
    return tuple([Tensor(o) for o in op_results])


def max_pool2d_with_indices_backward(
    grad_output: Tensor,
    self_: Tensor,
    kernel_size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    padding: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dilation: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    ceil_mode: Union[Torch_Value[Torch_BoolType], builtins.bool],
    indices: Tensor,
) -> Tensor:
    assert check_argument_types()
    if isinstance(kernel_size, (builtins.list, builtins.tuple)) and builtins.len(
        kernel_size
    ):
        kernel_size = builtins.list(kernel_size)
        for i, a in enumerate(kernel_size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                kernel_size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        kernel_size = torch_dialect.PrimListConstructOp(ls_type, kernel_size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(padding, (builtins.list, builtins.tuple)) and builtins.len(padding):
        padding = builtins.list(padding)
        for i, a in enumerate(padding):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                padding[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        padding = torch_dialect.PrimListConstructOp(ls_type, padding).result
    if isinstance(dilation, (builtins.list, builtins.tuple)) and builtins.len(dilation):
        dilation = builtins.list(dilation)
        for i, a in enumerate(dilation):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dilation[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dilation = torch_dialect.PrimListConstructOp(ls_type, dilation).result
    if isinstance(ceil_mode, builtins.bool):
        ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenMaxPool2dWithIndicesBackwardOp(
            result0_type,
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices,
        )
    )


def maximum(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMaximumOp(result0_type, self_, other))


# overload dim
@register_dispatch
def mean(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ],
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMeanDimOp(result0_type, self_, dim, keepdim, dtype))


@register_dispatch
def mean(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMeanOp(result0_type, self_, dtype))


# overload self_int
@register_dispatch
def min(
    self_: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(self_, (builtins.list, builtins.tuple)) and builtins.len(self_):
        self_ = builtins.list(self_)
        for i, a in enumerate(self_):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                self_[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        self_ = torch_dialect.PrimListConstructOp(ls_type, self_).result
    return Torch_Value(torch_dialect.PrimMinSelfIntOp(self_).result)


# overload int
@register_dispatch
def min(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.PrimMinIntOp(a, b).result)


def minimum(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMinimumOp(result0_type, self_, other))


def mish(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMishOp(result0_type, self_))


def mm(self_: Tensor, mat2: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMmOp(result0_type, self_, mat2))


def mse_loss(
    self_: Tensor,
    target: Tensor,
    reduction: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(reduction, builtins.int):
        reduction = torch_dialect.ConstantIntOp(reduction).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMseLossOp(result0_type, self_, target, reduction))


# overload Tensor
@register_dispatch
def mul(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMulTensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def mul(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMulScalarOp(result0_type, self_, other))


# overload int
@register_dispatch
def mul(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenMulIntOp(a, b).result)


# overload float
@register_dispatch
def mul(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.float):
        b = torch_dialect.ConstantFloatOp(b).result
    return Torch_Value(torch_dialect.AtenMulFloatOp(a, b).result)


# overload Tensor
@register_dispatch
def mul_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMul_TensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def mul_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMul_ScalarOp(result0_type, self_, other))


def mv(self_: Tensor, vec: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenMvOp(result0_type, self_, vec))


def narrow(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    start: Union[Torch_Value[Torch_IntType], builtins.int],
    length: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(start, builtins.int):
        start = torch_dialect.ConstantIntOp(start).result
    if isinstance(length, builtins.int):
        length = torch_dialect.ConstantIntOp(length).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenNarrowOp(result0_type, self_, dim, start, length))


def native_batch_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: Union[Torch_Value[Torch_BoolType], builtins.bool],
    momentum: Union[Torch_Value[Torch_FloatType], builtins.float],
    eps: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Tuple[Tensor, Tensor, Tensor]:
    assert check_argument_types()
    if weight is None:
        weight = torch_dialect.ConstantNoneOp().result
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if running_mean is None:
        running_mean = torch_dialect.ConstantNoneOp().result
    if running_var is None:
        running_var = torch_dialect.ConstantNoneOp().result
    if isinstance(training, builtins.bool):
        training = torch_dialect.ConstantBoolOp(training).result
    if isinstance(momentum, builtins.float):
        momentum = torch_dialect.ConstantFloatOp(momentum).result
    if isinstance(eps, builtins.float):
        eps = torch_dialect.ConstantFloatOp(eps).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    result2_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenNativeBatchNormOp(
            result0_type,
            result1_type,
            result2_type,
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
        )
    )
    return tuple([Tensor(o) for o in op_results])


def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: Union[Torch_Value[Torch_BoolType], builtins.bool],
    eps: Union[Torch_Value[Torch_FloatType], builtins.float],
    output_mask: Union[
        Sequence[Union[Torch_Value[Torch_BoolType], builtins.bool]],
        Torch_List[Torch_BoolType],
    ],
) -> Tuple[Tensor, Tensor, Tensor]:
    assert check_argument_types()
    if weight is None:
        weight = torch_dialect.ConstantNoneOp().result
    if running_mean is None:
        running_mean = torch_dialect.ConstantNoneOp().result
    if running_var is None:
        running_var = torch_dialect.ConstantNoneOp().result
    if save_mean is None:
        save_mean = torch_dialect.ConstantNoneOp().result
    if save_invstd is None:
        save_invstd = torch_dialect.ConstantNoneOp().result
    if isinstance(train, builtins.bool):
        train = torch_dialect.ConstantBoolOp(train).result
    if isinstance(eps, builtins.float):
        eps = torch_dialect.ConstantFloatOp(eps).result
    if isinstance(output_mask, (builtins.list, builtins.tuple)) and builtins.len(
        output_mask
    ):
        output_mask = builtins.list(output_mask)
        for i, a in enumerate(output_mask):
            if not isinstance(a, builtins.bool):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be bool"
            else:
                output_mask[i] = torch_dialect.ConstantBoolOp(a).result
        ls_type = Torch_List.of(Torch_BoolType())
        output_mask = torch_dialect.PrimListConstructOp(ls_type, output_mask).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    result2_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenNativeBatchNormBackwardOp(
            result0_type,
            result1_type,
            result2_type,
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            train,
            eps,
            output_mask,
        )
    )
    return tuple([Tensor(o) for o in op_results])


def native_dropout(
    input: Tensor,
    p: Union[Torch_Value[Torch_FloatType], builtins.float],
    train: Union[Torch_Value[Torch_BoolType], builtins.bool, None],
) -> Tuple[Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(p, builtins.float):
        p = torch_dialect.ConstantFloatOp(p).result
    if isinstance(train, builtins.bool):
        train = torch_dialect.ConstantBoolOp(train).result
    if train is None:
        train = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenNativeDropoutOp(result0_type, result1_type, input, p, train)
    )
    return tuple([Tensor(o) for o in op_results])


def native_dropout_backward(
    grad_output: Tensor,
    mask: Tensor,
    scale: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Tensor:
    assert check_argument_types()
    if isinstance(scale, builtins.float):
        scale = torch_dialect.ConstantFloatOp(scale).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenNativeDropoutBackwardOp(
            result0_type, grad_output, mask, scale
        )
    )


def native_layer_norm(
    input: Tensor,
    normalized_shape: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Tuple[Tensor, Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(normalized_shape, (builtins.list, builtins.tuple)) and builtins.len(
        normalized_shape
    ):
        normalized_shape = builtins.list(normalized_shape)
        for i, a in enumerate(normalized_shape):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                normalized_shape[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        normalized_shape = torch_dialect.PrimListConstructOp(
            ls_type, normalized_shape
        ).result
    if weight is None:
        weight = torch_dialect.ConstantNoneOp().result
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(eps, builtins.float):
        eps = torch_dialect.ConstantFloatOp(eps).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    result2_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenNativeLayerNormOp(
            result0_type,
            result1_type,
            result2_type,
            input,
            normalized_shape,
            weight,
            bias,
            eps,
        )
    )
    return tuple([Tensor(o) for o in op_results])


def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: Union[
        Sequence[Union[Torch_Value[Torch_BoolType], builtins.bool]],
        Torch_List[Torch_BoolType],
    ],
) -> Tuple[Tensor, Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(normalized_shape, (builtins.list, builtins.tuple)) and builtins.len(
        normalized_shape
    ):
        normalized_shape = builtins.list(normalized_shape)
        for i, a in enumerate(normalized_shape):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                normalized_shape[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        normalized_shape = torch_dialect.PrimListConstructOp(
            ls_type, normalized_shape
        ).result
    if weight is None:
        weight = torch_dialect.ConstantNoneOp().result
    if bias is None:
        bias = torch_dialect.ConstantNoneOp().result
    if isinstance(output_mask, (builtins.list, builtins.tuple)) and builtins.len(
        output_mask
    ):
        output_mask = builtins.list(output_mask)
        for i, a in enumerate(output_mask):
            if not isinstance(a, builtins.bool):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be bool"
            else:
                output_mask[i] = torch_dialect.ConstantBoolOp(a).result
        ls_type = Torch_List.of(Torch_BoolType())
        output_mask = torch_dialect.PrimListConstructOp(ls_type, output_mask).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    result2_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenNativeLayerNormBackwardOp(
            result0_type,
            result1_type,
            result2_type,
            grad_out,
            input,
            normalized_shape,
            mean,
            rstd,
            weight,
            bias,
            output_mask,
        )
    )
    return tuple([Tensor(o) for o in op_results])


# overload Tensor
@register_dispatch
def ne(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenNeTensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def ne(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenNeScalarOp(result0_type, self_, other))


# overload int_list
@register_dispatch
def ne(
    a: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    b: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, (builtins.list, builtins.tuple)) and builtins.len(a):
        a = builtins.list(a)
        for i, a in enumerate(a):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                a[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        a = torch_dialect.PrimListConstructOp(ls_type, a).result
    if isinstance(b, (builtins.list, builtins.tuple)) and builtins.len(b):
        b = builtins.list(b)
        for i, a in enumerate(b):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                b[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        b = torch_dialect.PrimListConstructOp(ls_type, b).result
    return Torch_Value(torch_dialect.AtenNeIntListOp(a, b).result)


# overload int
@register_dispatch
def ne(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenNeIntOp(a, b).result)


# overload float_int
@register_dispatch
def ne(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenNeFloatIntOp(a, b).result)


# overload bool
@register_dispatch
def ne(
    a: Union[Torch_Value[Torch_BoolType], builtins.bool],
    b: Union[Torch_Value[Torch_BoolType], builtins.bool],
) -> Union[Torch_Value[Torch_BoolType], builtins.bool]:
    assert check_argument_types()
    if isinstance(a, builtins.bool):
        a = torch_dialect.ConstantBoolOp(a).result
    if isinstance(b, builtins.bool):
        b = torch_dialect.ConstantBoolOp(b).result
    return Torch_Value(torch_dialect.AtenNeBoolOp(a, b).result)


# overload Tensor
@register_dispatch
def ne_(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenNe_TensorOp(result0_type, self_, other))


# overload Scalar
@register_dispatch
def ne_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenNe_ScalarOp(result0_type, self_, other))


@register_dispatch
def neg(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenNegOp(result0_type, self_))


# overload int
@register_dispatch
def neg(
    a: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    return Torch_Value(torch_dialect.AtenNegIntOp(a).result)


# overload float
@register_dispatch
def neg(
    a: Union[Torch_Value[Torch_FloatType], builtins.float]
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    return Torch_Value(torch_dialect.AtenNegFloatOp(a).result)


def neg_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenNeg_Op(result0_type, self_))


def new_empty(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenNewEmptyOp(
            result0_type, self_, size, dtype, layout, device, pin_memory
        )
    )


def new_empty_strided(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    stride: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(stride, (builtins.list, builtins.tuple)) and builtins.len(stride):
        stride = builtins.list(stride)
        for i, a in enumerate(stride):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                stride[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        stride = torch_dialect.PrimListConstructOp(ls_type, stride).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenNewEmptyStridedOp(
            result0_type, self_, size, stride, dtype, layout, device, pin_memory
        )
    )


def new_ones(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenNewOnesOp(
            result0_type, self_, size, dtype, layout, device, pin_memory
        )
    )


def new_zeros(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenNewZerosOp(
            result0_type, self_, size, dtype, layout, device, pin_memory
        )
    )


def nll_loss_backward(
    grad_output: Tensor,
    self_: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: Union[Torch_Value[Torch_IntType], builtins.int],
    ignore_index: Union[Torch_Value[Torch_IntType], builtins.int],
    total_weight: Tensor,
) -> Tensor:
    assert check_argument_types()
    if weight is None:
        weight = torch_dialect.ConstantNoneOp().result
    if isinstance(reduction, builtins.int):
        reduction = torch_dialect.ConstantIntOp(reduction).result
    if isinstance(ignore_index, builtins.int):
        ignore_index = torch_dialect.ConstantIntOp(ignore_index).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenNllLossBackwardOp(
            result0_type,
            grad_output,
            self_,
            target,
            weight,
            reduction,
            ignore_index,
            total_weight,
        )
    )


def nll_loss_forward(
    self_: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: Union[Torch_Value[Torch_IntType], builtins.int],
    ignore_index: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tuple[Tensor, Tensor]:
    assert check_argument_types()
    if weight is None:
        weight = torch_dialect.ConstantNoneOp().result
    if isinstance(reduction, builtins.int):
        reduction = torch_dialect.ConstantIntOp(reduction).result
    if isinstance(ignore_index, builtins.int):
        ignore_index = torch_dialect.ConstantIntOp(ignore_index).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenNllLossForwardOp(
            result0_type, result1_type, self_, target, weight, reduction, ignore_index
        )
    )
    return tuple([Tensor(o) for o in op_results])


# overload ScalarOpt_dim
def norm(
    self_: Tensor,
    p: Optional[TorchNumber],
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(p, (builtins.int, builtins.float)):
        p = torch_dialect.ConstantNumberOp(p).result
    if p is None:
        p = torch_dialect.ConstantNoneOp().result
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenNormScalarOptDimOp(result0_type, self_, p, dim, keepdim)
    )


def numel(self_: Tensor) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenNumelOp(self_).result)


def numpy_T(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenNumpyTOp(result0_type, self_))


def ones(
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenOnesOp(result0_type, size, dtype, layout, device, pin_memory)
    )


def ones_like(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenOnesLikeOp(
            result0_type, self_, dtype, layout, device, pin_memory, memory_format
        )
    )


def pad(
    self_: Tensor,
    pad: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    mode: Union[Torch_Value[Torch_StringType], builtins.str] = "constant",
    value: Union[Torch_Value[Torch_FloatType], builtins.float, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(pad, (builtins.list, builtins.tuple)) and builtins.len(pad):
        pad = builtins.list(pad)
        for i, a in enumerate(pad):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                pad[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        pad = torch_dialect.PrimListConstructOp(ls_type, pad).result
    if isinstance(mode, builtins.str):
        mode = torch_dialect.ConstantStrOp(mode).result
    if isinstance(value, builtins.float):
        value = torch_dialect.ConstantFloatOp(value).result
    if value is None:
        value = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenPadOp(result0_type, self_, pad, mode, value))


def permute(
    self_: Tensor,
    dims: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dims, (builtins.list, builtins.tuple)) and builtins.len(dims):
        dims = builtins.list(dims)
        for i, a in enumerate(dims):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dims[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dims = torch_dialect.PrimListConstructOp(ls_type, dims).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenPermuteOp(result0_type, self_, dims))


def permute_copy(
    self_: Tensor,
    dims: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dims, (builtins.list, builtins.tuple)) and builtins.len(dims):
        dims = builtins.list(dims)
        for i, a in enumerate(dims):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dims[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dims = torch_dialect.PrimListConstructOp(ls_type, dims).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenPermuteCopyOp(result0_type, self_, dims))


# overload Tensor_Scalar
@register_dispatch
def pow(self_: Tensor, exponent: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(exponent, (builtins.int, builtins.float)):
        exponent = torch_dialect.ConstantNumberOp(exponent).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenPowTensorScalarOp(result0_type, self_, exponent))


# overload Tensor_Tensor
@register_dispatch
def pow(self_: Tensor, exponent: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenPowTensorTensorOp(result0_type, self_, exponent))


# overload int_float
@register_dispatch
def pow(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.float):
        b = torch_dialect.ConstantFloatOp(b).result
    return Torch_Value(torch_dialect.AtenPowIntFloatOp(a, b).result)


def prelu(self_: Tensor, weight: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenPreluOp(result0_type, self_, weight))


def rand_like(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenRandLikeOp(
            result0_type, self_, dtype, layout, device, pin_memory, memory_format
        )
    )


# overload low
def randint(
    low: Union[Torch_Value[Torch_IntType], builtins.int],
    high: Union[Torch_Value[Torch_IntType], builtins.int],
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = 4,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(low, builtins.int):
        low = torch_dialect.ConstantIntOp(low).result
    if isinstance(high, builtins.int):
        high = torch_dialect.ConstantIntOp(high).result
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenRandintLowOp(
            result0_type, low, high, size, dtype, layout, device, pin_memory
        )
    )


@register_dispatch
def randn(
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenRandnOp(result0_type, size, dtype, layout, device, pin_memory)
    )


# overload generator
@register_dispatch
def randn(
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    generator: Optional[Torch_GeneratorType],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if generator is None:
        generator = torch_dialect.ConstantNoneOp().result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenRandnGeneratorOp(
            result0_type, size, generator, dtype, layout, device, pin_memory
        )
    )


def randn_like(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenRandnLikeOp(
            result0_type, self_, dtype, layout, device, pin_memory, memory_format
        )
    )


def reciprocal(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenReciprocalOp(result0_type, self_))


def reciprocal_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenReciprocal_Op(result0_type, self_))


def relu(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenReluOp(result0_type, self_))


def relu6(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRelu6Op(result0_type, self_))


def relu6_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRelu6_Op(result0_type, self_))


def relu_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRelu_Op(result0_type, self_))


# overload int
@register_dispatch
def remainder(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenRemainderIntOp(a, b).result)


# overload Scalar
@register_dispatch
def remainder(self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRemainderScalarOp(result0_type, self_, other))


def repeat(
    self_: Tensor,
    repeats: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(repeats, (builtins.list, builtins.tuple)) and builtins.len(repeats):
        repeats = builtins.list(repeats)
        for i, a in enumerate(repeats):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                repeats[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        repeats = torch_dialect.PrimListConstructOp(ls_type, repeats).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRepeatOp(result0_type, self_, repeats))


def reshape(
    self_: Tensor,
    shape: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(shape, (builtins.list, builtins.tuple)) and builtins.len(shape):
        shape = builtins.list(shape)
        for i, a in enumerate(shape):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                shape[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        shape = torch_dialect.PrimListConstructOp(ls_type, shape).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenReshapeOp(result0_type, self_, shape))


def resize_(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenResize_Op(result0_type, self_, size, memory_format))


def roll(
    self_: Tensor,
    shifts: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dims: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ] = (),
) -> Tensor:
    assert check_argument_types()
    if isinstance(shifts, (builtins.list, builtins.tuple)) and builtins.len(shifts):
        shifts = builtins.list(shifts)
        for i, a in enumerate(shifts):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                shifts[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        shifts = torch_dialect.PrimListConstructOp(ls_type, shifts).result
    if isinstance(dims, (builtins.list, builtins.tuple)) and builtins.len(dims):
        dims = builtins.list(dims)
        for i, a in enumerate(dims):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dims[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dims = torch_dialect.PrimListConstructOp(ls_type, dims).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRollOp(result0_type, self_, shifts, dims))


def round(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRoundOp(result0_type, self_))


def round_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRound_Op(result0_type, self_))


def rsqrt(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRsqrtOp(result0_type, self_))


def rsqrt_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRsqrt_Op(result0_type, self_))


# overload Scalar
def rsub(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenRsubScalarOp(result0_type, self_, other, alpha))


def scatter_add(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    index: Tensor,
    src: Tensor,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenScatterAddOp(result0_type, self_, dim, index, src))


def scatter_add_(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    index: Tensor,
    src: Tensor,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenScatterAdd_Op(result0_type, self_, dim, index, src))


# overload two
def scatter_reduce(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    index: Tensor,
    src: Tensor,
    reduce: Union[Torch_Value[Torch_StringType], builtins.str],
    include_self: Union[Torch_Value[Torch_BoolType], builtins.bool] = True,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(reduce, builtins.str):
        reduce = torch_dialect.ConstantStrOp(reduce).result
    if isinstance(include_self, builtins.bool):
        include_self = torch_dialect.ConstantBoolOp(include_self).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenScatterReduceTwoOp(
            result0_type, self_, dim, index, src, reduce, include_self
        )
    )


# overload two
def scatter_reduce_(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    index: Tensor,
    src: Tensor,
    reduce: Union[Torch_Value[Torch_StringType], builtins.str],
    include_self: Union[Torch_Value[Torch_BoolType], builtins.bool] = True,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(reduce, builtins.str):
        reduce = torch_dialect.ConstantStrOp(reduce).result
    if isinstance(include_self, builtins.bool):
        include_self = torch_dialect.ConstantBoolOp(include_self).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenScatterReduce_TwoOp(
            result0_type, self_, dim, index, src, reduce, include_self
        )
    )


# overload int
def select(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    index: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(index, builtins.int):
        index = torch_dialect.ConstantIntOp(index).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSelectIntOp(result0_type, self_, dim, index))


# overload int
def select_copy(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    index: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(index, builtins.int):
        index = torch_dialect.ConstantIntOp(index).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSelectCopyIntOp(result0_type, self_, dim, index))


def select_scatter(
    self_: Tensor,
    src: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    index: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(index, builtins.int):
        index = torch_dialect.ConstantIntOp(index).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenSelectScatterOp(result0_type, self_, src, dim, index)
    )


def sigmoid(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSigmoidOp(result0_type, self_))


def sigmoid_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSigmoid_Op(result0_type, self_))


def silu(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSiluOp(result0_type, self_))


def silu_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSilu_Op(result0_type, self_))


def sin(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSinOp(result0_type, self_))


def sin_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSin_Op(result0_type, self_))


@register_dispatch
def size(
    self_: Tensor,
) -> Union[
    Sequence[Union[Torch_Value[Torch_IntType], builtins.int]], Torch_List[Torch_IntType]
]:
    assert check_argument_types()
    result0_type = Torch_List.of(Torch_IntType())
    return Torch_Value(torch_dialect.AtenSizeOp(result0_type, self_).result)


# overload int
@register_dispatch
def size(
    self_: Tensor, dim: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    return Torch_Value(torch_dialect.AtenSizeIntOp(self_, dim).result)


# overload Tensor
@register_dispatch
def slice(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    start: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    end: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    step: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(start, builtins.int):
        start = torch_dialect.ConstantIntOp(start).result
    if start is None:
        start = torch_dialect.ConstantNoneOp().result
    if isinstance(end, builtins.int):
        end = torch_dialect.ConstantIntOp(end).result
    if end is None:
        end = torch_dialect.ConstantNoneOp().result
    if isinstance(step, builtins.int):
        step = torch_dialect.ConstantIntOp(step).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenSliceTensorOp(result0_type, self_, dim, start, end, step)
    )


# overload t
@register_dispatch
def slice(
    l: Union[Sequence[Any], Any],
    start: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    end: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    step: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
) -> Union[Sequence[Any], Any]:
    assert check_argument_types()
    if isinstance(start, builtins.int):
        start = torch_dialect.ConstantIntOp(start).result
    if start is None:
        start = torch_dialect.ConstantNoneOp().result
    if isinstance(end, builtins.int):
        end = torch_dialect.ConstantIntOp(end).result
    if end is None:
        end = torch_dialect.ConstantNoneOp().result
    if isinstance(step, builtins.int):
        step = torch_dialect.ConstantIntOp(step).result
    result0_type = Torch_List.of(Torch_AnyType())
    return Torch_Value(
        torch_dialect.AtenSliceTOp(result0_type, l, start, end, step).result
    )


# overload Tensor
def slice_copy(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    start: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    end: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    step: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(start, builtins.int):
        start = torch_dialect.ConstantIntOp(start).result
    if start is None:
        start = torch_dialect.ConstantNoneOp().result
    if isinstance(end, builtins.int):
        end = torch_dialect.ConstantIntOp(end).result
    if end is None:
        end = torch_dialect.ConstantNoneOp().result
    if isinstance(step, builtins.int):
        step = torch_dialect.ConstantIntOp(step).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenSliceCopyTensorOp(result0_type, self_, dim, start, end, step)
    )


def slice_scatter(
    self_: Tensor,
    src: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
    start: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    end: Union[Torch_Value[Torch_IntType], builtins.int, None] = None,
    step: Union[Torch_Value[Torch_IntType], builtins.int] = 1,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(start, builtins.int):
        start = torch_dialect.ConstantIntOp(start).result
    if start is None:
        start = torch_dialect.ConstantNoneOp().result
    if isinstance(end, builtins.int):
        end = torch_dialect.ConstantIntOp(end).result
    if end is None:
        end = torch_dialect.ConstantNoneOp().result
    if isinstance(step, builtins.int):
        step = torch_dialect.ConstantIntOp(step).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenSliceScatterOp(
            result0_type, self_, src, dim, start, end, step
        )
    )


# overload int
def softmax(
    self_: Tensor,
    dim: Union[Torch_Value[Torch_IntType], builtins.int],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSoftmaxIntOp(result0_type, self_, dim, dtype))


def softplus(
    self_: Tensor, beta: TorchNumber = 1, threshold: TorchNumber = 20
) -> Tensor:
    assert check_argument_types()
    if isinstance(beta, (builtins.int, builtins.float)):
        beta = torch_dialect.ConstantNumberOp(beta).result
    if isinstance(threshold, (builtins.int, builtins.float)):
        threshold = torch_dialect.ConstantNumberOp(threshold).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSoftplusOp(result0_type, self_, beta, threshold))


# overload int
def sort(
    self_: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    reverse: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> None:
    assert check_argument_types()
    if isinstance(self_, (builtins.list, builtins.tuple)) and builtins.len(self_):
        self_ = builtins.list(self_)
        for i, a in enumerate(self_):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                self_[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        self_ = torch_dialect.PrimListConstructOp(ls_type, self_).result
    if isinstance(reverse, builtins.bool):
        reverse = torch_dialect.ConstantBoolOp(reverse).result
    torch_dialect.AtenSortIntOp(self_, reverse)


@register_dispatch
def sqrt(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSqrtOp(result0_type, self_))


# overload int
@register_dispatch
def sqrt(
    a: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    return Torch_Value(torch_dialect.AtenSqrtIntOp(a).result)


def sqrt_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSqrt_Op(result0_type, self_))


def square(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSquareOp(result0_type, self_))


def square_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSquare_Op(result0_type, self_))


# overload dim
@register_dispatch
def squeeze(
    self_: Tensor, dim: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSqueezeDimOp(result0_type, self_, dim))


@register_dispatch
def squeeze(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSqueezeOp(result0_type, self_))


@register_dispatch
def squeeze_copy(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSqueezeCopyOp(result0_type, self_))


# overload dim
@register_dispatch
def squeeze_copy(
    self_: Tensor, dim: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSqueezeCopyDimOp(result0_type, self_, dim))


def stack(
    tensors: Union[Sequence[Tensor], Tensor],
    dim: Union[Torch_Value[Torch_IntType], builtins.int] = 0,
) -> Tensor:
    assert check_argument_types()
    if isinstance(tensors, (builtins.list, builtins.tuple)) and builtins.len(tensors):
        assert builtins.all([isinstance(a, Tensor) for a in tensors])
        ls_type = Torch_List.of(Torch_NonValueTensorType())
        tensors = torch_dialect.PrimListConstructOp(ls_type, tensors).result
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenStackOp(result0_type, tensors, dim))


@register_dispatch
def std(
    self_: Tensor, unbiased: Union[Torch_Value[Torch_BoolType], builtins.bool] = True
) -> Tensor:
    assert check_argument_types()
    if isinstance(unbiased, builtins.bool):
        unbiased = torch_dialect.ConstantBoolOp(unbiased).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenStdOp(result0_type, self_, unbiased))


# overload dim
@register_dispatch
def std(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ],
    unbiased: Union[Torch_Value[Torch_BoolType], builtins.bool] = True,
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(unbiased, builtins.bool):
        unbiased = torch_dialect.ConstantBoolOp(unbiased).result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenStdDimOp(result0_type, self_, dim, unbiased, keepdim)
    )


# overload correction
@register_dispatch
def std(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ] = None,
    correction: Optional[TorchNumber] = None,
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(correction, (builtins.int, builtins.float)):
        correction = torch_dialect.ConstantNumberOp(correction).result
    if correction is None:
        correction = torch_dialect.ConstantNoneOp().result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenStdCorrectionOp(result0_type, self_, dim, correction, keepdim)
    )


def str(
    elem: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ]
) -> Union[Torch_Value[Torch_StringType], builtins.str]:
    assert check_argument_types()
    return Torch_Value(torch_dialect.AtenStrOp(elem).result)


# overload Tensor
@register_dispatch
def sub(self_: Tensor, other: Tensor, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSubTensorOp(result0_type, self_, other, alpha))


# overload Scalar
@register_dispatch
def sub(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSubScalarOp(result0_type, self_, other, alpha))


# overload int
@register_dispatch
def sub(
    a: Union[Torch_Value[Torch_IntType], builtins.int],
    b: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Union[Torch_Value[Torch_IntType], builtins.int]:
    assert check_argument_types()
    if isinstance(a, builtins.int):
        a = torch_dialect.ConstantIntOp(a).result
    if isinstance(b, builtins.int):
        b = torch_dialect.ConstantIntOp(b).result
    return Torch_Value(torch_dialect.AtenSubIntOp(a, b).result)


# overload float
@register_dispatch
def sub(
    a: Union[Torch_Value[Torch_FloatType], builtins.float],
    b: Union[Torch_Value[Torch_FloatType], builtins.float],
) -> Union[Torch_Value[Torch_FloatType], builtins.float]:
    assert check_argument_types()
    if isinstance(a, builtins.float):
        a = torch_dialect.ConstantFloatOp(a).result
    if isinstance(b, builtins.float):
        b = torch_dialect.ConstantFloatOp(b).result
    return Torch_Value(torch_dialect.AtenSubFloatOp(a, b).result)


@register_dispatch
def sub(a: TorchNumber, b: TorchNumber) -> TorchNumber:
    assert check_argument_types()
    if isinstance(a, (builtins.int, builtins.float)):
        a = torch_dialect.ConstantNumberOp(a).result
    if isinstance(b, (builtins.int, builtins.float)):
        b = torch_dialect.ConstantNumberOp(b).result
    result0_type = Torch_NumberType()
    return Torch_Value(torch_dialect.AtenSubOp(result0_type, a, b).result)


# overload Tensor
@register_dispatch
def sub_(self_: Tensor, other: Tensor, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSub_TensorOp(result0_type, self_, other, alpha))


# overload Scalar
@register_dispatch
def sub_(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    if isinstance(alpha, (builtins.int, builtins.float)):
        alpha = torch_dialect.ConstantNumberOp(alpha).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSub_ScalarOp(result0_type, self_, other, alpha))


@register_dispatch
def sum(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenSumOp(result0_type, self_, dtype))


# overload dim_IntList
@register_dispatch
def sum(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ],
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenSumDimIntListOp(result0_type, self_, dim, keepdim, dtype)
    )


def t(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTOp(result0_type, self_))


def t_copy(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTCopyOp(result0_type, self_))


def tanh(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTanhOp(result0_type, self_))


def tanh_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTanh_Op(result0_type, self_))


def tanh_backward(grad_output: Tensor, output: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTanhBackwardOp(result0_type, grad_output, output))


@register_dispatch
def tensor(
    data: Union[Sequence[Any], Any],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    requires_grad: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(requires_grad, builtins.bool):
        requires_grad = torch_dialect.ConstantBoolOp(requires_grad).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenTensorOp(result0_type, data, dtype, device, requires_grad)
    )


# overload bool
@register_dispatch
def tensor(
    t: Union[Torch_Value[Torch_BoolType], builtins.bool],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    requires_grad: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(t, builtins.bool):
        t = torch_dialect.ConstantBoolOp(t).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(requires_grad, builtins.bool):
        requires_grad = torch_dialect.ConstantBoolOp(requires_grad).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenTensorBoolOp(result0_type, t, dtype, device, requires_grad)
    )


# overload int
@register_dispatch
def tensor(
    t: Union[Torch_Value[Torch_IntType], builtins.int],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    requires_grad: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(t, builtins.int):
        t = torch_dialect.ConstantIntOp(t).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(requires_grad, builtins.bool):
        requires_grad = torch_dialect.ConstantBoolOp(requires_grad).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenTensorIntOp(result0_type, t, dtype, device, requires_grad)
    )


# overload float
@register_dispatch
def tensor(
    t: Union[Torch_Value[Torch_FloatType], builtins.float],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    requires_grad: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(t, builtins.float):
        t = torch_dialect.ConstantFloatOp(t).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(requires_grad, builtins.bool):
        requires_grad = torch_dialect.ConstantBoolOp(requires_grad).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenTensorFloatOp(result0_type, t, dtype, device, requires_grad)
    )


def threshold(self_: Tensor, threshold: TorchNumber, value: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(threshold, (builtins.int, builtins.float)):
        threshold = torch_dialect.ConstantNumberOp(threshold).result
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenThresholdOp(result0_type, self_, threshold, value))


def threshold_(self_: Tensor, threshold: TorchNumber, value: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(threshold, (builtins.int, builtins.float)):
        threshold = torch_dialect.ConstantNumberOp(threshold).result
    if isinstance(value, (builtins.int, builtins.float)):
        value = torch_dialect.ConstantNumberOp(value).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenThreshold_Op(result0_type, self_, threshold, value))


def threshold_backward(
    grad_output: Tensor, self_: Tensor, threshold: TorchNumber
) -> Tensor:
    assert check_argument_types()
    if isinstance(threshold, (builtins.int, builtins.float)):
        threshold = torch_dialect.ConstantNumberOp(threshold).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenThresholdBackwardOp(
            result0_type, grad_output, self_, threshold
        )
    )


# overload dtype
@register_dispatch
def to(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype],
    non_blocking: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    copy: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if isinstance(non_blocking, builtins.bool):
        non_blocking = torch_dialect.ConstantBoolOp(non_blocking).result
    if isinstance(copy, builtins.bool):
        copy = torch_dialect.ConstantBoolOp(copy).result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenToDtypeOp(
            result0_type, self_, dtype, non_blocking, copy, memory_format
        )
    )


# overload dtype_layout
@register_dispatch
def to(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    non_blocking: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    copy: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(non_blocking, builtins.bool):
        non_blocking = torch_dialect.ConstantBoolOp(non_blocking).result
    if isinstance(copy, builtins.bool):
        copy = torch_dialect.ConstantBoolOp(copy).result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenToDtypeLayoutOp(
            result0_type,
            self_,
            dtype,
            layout,
            device,
            pin_memory,
            non_blocking,
            copy,
            memory_format,
        )
    )


# overload other
@register_dispatch
def to(
    self_: Tensor,
    other: Tensor,
    non_blocking: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    copy: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(non_blocking, builtins.bool):
        non_blocking = torch_dialect.ConstantBoolOp(non_blocking).result
    if isinstance(copy, builtins.bool):
        copy = torch_dialect.ConstantBoolOp(copy).result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenToOtherOp(
            result0_type, self_, other, non_blocking, copy, memory_format
        )
    )


# overload prim_Device
@register_dispatch
def to(
    self_: Tensor,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    non_blocking: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    copy: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(non_blocking, builtins.bool):
        non_blocking = torch_dialect.ConstantBoolOp(non_blocking).result
    if isinstance(copy, builtins.bool):
        copy = torch_dialect.ConstantBoolOp(copy).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenToPrimDeviceOp(
            result0_type, self_, device, dtype, non_blocking, copy
        )
    )


# overload device
@register_dispatch
def to(
    self_: Tensor,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype],
    non_blocking: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    copy: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if isinstance(non_blocking, builtins.bool):
        non_blocking = torch_dialect.ConstantBoolOp(non_blocking).result
    if isinstance(copy, builtins.bool):
        copy = torch_dialect.ConstantBoolOp(copy).result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenToDeviceOp(
            result0_type, self_, device, dtype, non_blocking, copy, memory_format
        )
    )


def topk(
    self_: Tensor,
    k: Union[Torch_Value[Torch_IntType], builtins.int],
    dim: Union[Torch_Value[Torch_IntType], builtins.int] = -1,
    largest: Union[Torch_Value[Torch_BoolType], builtins.bool] = True,
    sorted: Union[Torch_Value[Torch_BoolType], builtins.bool] = True,
) -> Tuple[Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(k, builtins.int):
        k = torch_dialect.ConstantIntOp(k).result
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    if isinstance(largest, builtins.bool):
        largest = torch_dialect.ConstantBoolOp(largest).result
    if isinstance(sorted, builtins.bool):
        sorted = torch_dialect.ConstantBoolOp(sorted).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenTopkOp(
            result0_type, result1_type, self_, k, dim, largest, sorted
        )
    )
    return tuple([Tensor(o) for o in op_results])


# overload int
def transpose(
    self_: Tensor,
    dim0: Union[Torch_Value[Torch_IntType], builtins.int],
    dim1: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim0, builtins.int):
        dim0 = torch_dialect.ConstantIntOp(dim0).result
    if isinstance(dim1, builtins.int):
        dim1 = torch_dialect.ConstantIntOp(dim1).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTransposeIntOp(result0_type, self_, dim0, dim1))


# overload int
def transpose_copy(
    self_: Tensor,
    dim0: Union[Torch_Value[Torch_IntType], builtins.int],
    dim1: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim0, builtins.int):
        dim0 = torch_dialect.ConstantIntOp(dim0).result
    if isinstance(dim1, builtins.int):
        dim1 = torch_dialect.ConstantIntOp(dim1).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTransposeCopyIntOp(result0_type, self_, dim0, dim1))


def triu(
    self_: Tensor, diagonal: Union[Torch_Value[Torch_IntType], builtins.int] = 0
) -> Tensor:
    assert check_argument_types()
    if isinstance(diagonal, builtins.int):
        diagonal = torch_dialect.ConstantIntOp(diagonal).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTriuOp(result0_type, self_, diagonal))


def triu_(
    self_: Tensor, diagonal: Union[Torch_Value[Torch_IntType], builtins.int] = 0
) -> Tensor:
    assert check_argument_types()
    if isinstance(diagonal, builtins.int):
        diagonal = torch_dialect.ConstantIntOp(diagonal).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTriu_Op(result0_type, self_, diagonal))


def type_as(self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenTypeAsOp(result0_type, self_, other))


def unchecked_cast(
    x: Union[
        TorchNumber,
        Tensor,
        Torch_Value[Torch_AnyType],
        Any,
        Torch_Value[Torch_BoolType],
        builtins.bool,
        Torch_Dict,
        Torch_Value[Torch_DeviceType],
        builtins.str,
        Torch_GeneratorType,
        Torch_List,
        None,
        Torch_Value[Torch_StringType],
        Tuple,
    ]
) -> Union[
    TorchNumber,
    Tensor,
    Torch_Value[Torch_AnyType],
    Any,
    Torch_Value[Torch_BoolType],
    builtins.bool,
    Torch_Dict,
    Torch_Value[Torch_DeviceType],
    builtins.str,
    Torch_GeneratorType,
    Torch_List,
    None,
    Torch_Value[Torch_StringType],
    Tuple,
]:
    assert check_argument_types()
    result0_type = Torch_AnyType()
    return Torch_Value(torch_dialect.PrimUncheckedCastOp(result0_type, x).result)


def unfold_copy(
    self_: Tensor,
    dimension: Union[Torch_Value[Torch_IntType], builtins.int],
    size: Union[Torch_Value[Torch_IntType], builtins.int],
    step: Union[Torch_Value[Torch_IntType], builtins.int],
) -> Tensor:
    assert check_argument_types()
    if isinstance(dimension, builtins.int):
        dimension = torch_dialect.ConstantIntOp(dimension).result
    if isinstance(size, builtins.int):
        size = torch_dialect.ConstantIntOp(size).result
    if isinstance(step, builtins.int):
        step = torch_dialect.ConstantIntOp(step).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenUnfoldCopyOp(result0_type, self_, dimension, size, step)
    )


def uniform(
    self_: Tensor,
    from_: Union[Torch_Value[Torch_FloatType], builtins.float] = 0.0,
    to: Union[Torch_Value[Torch_FloatType], builtins.float] = 1.0,
    generator: Optional[Torch_GeneratorType] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(from_, builtins.float):
        from_ = torch_dialect.ConstantFloatOp(from_).result
    if isinstance(to, builtins.float):
        to = torch_dialect.ConstantFloatOp(to).result
    if generator is None:
        generator = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenUniformOp(result0_type, self_, from_, to, generator)
    )


def uniform_(
    self_: Tensor,
    from_: Union[Torch_Value[Torch_FloatType], builtins.float] = 0.0,
    to: Union[Torch_Value[Torch_FloatType], builtins.float] = 1.0,
    generator: Optional[Torch_GeneratorType] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(from_, builtins.float):
        from_ = torch_dialect.ConstantFloatOp(from_).result
    if isinstance(to, builtins.float):
        to = torch_dialect.ConstantFloatOp(to).result
    if generator is None:
        generator = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenUniform_Op(result0_type, self_, from_, to, generator)
    )


def unsqueeze(
    self_: Tensor, dim: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenUnsqueezeOp(result0_type, self_, dim))


def unsqueeze_(
    self_: Tensor, dim: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenUnsqueeze_Op(result0_type, self_, dim))


def unsqueeze_copy(
    self_: Tensor, dim: Union[Torch_Value[Torch_IntType], builtins.int]
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, builtins.int):
        dim = torch_dialect.ConstantIntOp(dim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenUnsqueezeCopyOp(result0_type, self_, dim))


def upsample_nearest2d(
    self_: Tensor,
    output_size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    scales_h: Union[Torch_Value[Torch_FloatType], builtins.float, None] = None,
    scales_w: Union[Torch_Value[Torch_FloatType], builtins.float, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(output_size, (builtins.list, builtins.tuple)) and builtins.len(
        output_size
    ):
        output_size = builtins.list(output_size)
        for i, a in enumerate(output_size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_size = torch_dialect.PrimListConstructOp(ls_type, output_size).result
    if isinstance(scales_h, builtins.float):
        scales_h = torch_dialect.ConstantFloatOp(scales_h).result
    if scales_h is None:
        scales_h = torch_dialect.ConstantNoneOp().result
    if isinstance(scales_w, builtins.float):
        scales_w = torch_dialect.ConstantFloatOp(scales_w).result
    if scales_w is None:
        scales_w = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenUpsampleNearest2dOp(
            result0_type, self_, output_size, scales_h, scales_w
        )
    )


def upsample_nearest2d_backward(
    grad_output: Tensor,
    output_size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    input_size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    scales_h: Union[Torch_Value[Torch_FloatType], builtins.float, None] = None,
    scales_w: Union[Torch_Value[Torch_FloatType], builtins.float, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(output_size, (builtins.list, builtins.tuple)) and builtins.len(
        output_size
    ):
        output_size = builtins.list(output_size)
        for i, a in enumerate(output_size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                output_size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        output_size = torch_dialect.PrimListConstructOp(ls_type, output_size).result
    if isinstance(input_size, (builtins.list, builtins.tuple)) and builtins.len(
        input_size
    ):
        input_size = builtins.list(input_size)
        for i, a in enumerate(input_size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                input_size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        input_size = torch_dialect.PrimListConstructOp(ls_type, input_size).result
    if isinstance(scales_h, builtins.float):
        scales_h = torch_dialect.ConstantFloatOp(scales_h).result
    if scales_h is None:
        scales_h = torch_dialect.ConstantNoneOp().result
    if isinstance(scales_w, builtins.float):
        scales_w = torch_dialect.ConstantFloatOp(scales_w).result
    if scales_w is None:
        scales_w = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenUpsampleNearest2dBackwardOp(
            result0_type, grad_output, output_size, input_size, scales_h, scales_w
        )
    )


@register_dispatch
def var(
    self_: Tensor, unbiased: Union[Torch_Value[Torch_BoolType], builtins.bool] = True
) -> Tensor:
    assert check_argument_types()
    if isinstance(unbiased, builtins.bool):
        unbiased = torch_dialect.ConstantBoolOp(unbiased).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenVarOp(result0_type, self_, unbiased))


# overload dim
@register_dispatch
def var(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ],
    unbiased: Union[Torch_Value[Torch_BoolType], builtins.bool] = True,
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(unbiased, builtins.bool):
        unbiased = torch_dialect.ConstantBoolOp(unbiased).result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenVarDimOp(result0_type, self_, dim, unbiased, keepdim)
    )


# overload correction
@register_dispatch
def var(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ] = None,
    correction: Optional[TorchNumber] = None,
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(correction, (builtins.int, builtins.float)):
        correction = torch_dialect.ConstantNumberOp(correction).result
    if correction is None:
        correction = torch_dialect.ConstantNoneOp().result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenVarCorrectionOp(result0_type, self_, dim, correction, keepdim)
    )


@register_dispatch
def var(
    inp: Tensor,
    dims: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ],
    correction: Union[Torch_Value[Torch_FloatType], builtins.float],
    output_dtype: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_dtype, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dims, (builtins.list, builtins.tuple)) and builtins.len(dims):
        dims = builtins.list(dims)
        for i, a in enumerate(dims):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dims[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dims = torch_dialect.PrimListConstructOp(ls_type, dims).result
    if dims is None:
        dims = torch_dialect.ConstantNoneOp().result
    if isinstance(correction, builtins.float):
        correction = torch_dialect.ConstantFloatOp(correction).result
    if isinstance(output_dtype, pi_dtype):
        output_dtype = output_dtype.value
    if isinstance(output_dtype, builtins.int):
        output_dtype = torch_dialect.ConstantIntOp(output_dtype).result
    if output_dtype is None:
        output_dtype = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.PrimsVarOp(result0_type, inp, dims, correction, output_dtype)
    )


# overload correction
@register_dispatch
def var_mean(
    self_: Tensor,
    dim: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
        None,
    ] = None,
    correction: Optional[TorchNumber] = None,
    keepdim: Union[Torch_Value[Torch_BoolType], builtins.bool] = False,
) -> Tuple[Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(dim, (builtins.list, builtins.tuple)) and builtins.len(dim):
        dim = builtins.list(dim)
        for i, a in enumerate(dim):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                dim[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        dim = torch_dialect.PrimListConstructOp(ls_type, dim).result
    if dim is None:
        dim = torch_dialect.ConstantNoneOp().result
    if isinstance(correction, (builtins.int, builtins.float)):
        correction = torch_dialect.ConstantNumberOp(correction).result
    if correction is None:
        correction = torch_dialect.ConstantNoneOp().result
    if isinstance(keepdim, builtins.bool):
        keepdim = torch_dialect.ConstantBoolOp(keepdim).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenVarMeanCorrectionOp(
            result0_type, result1_type, self_, dim, correction, keepdim
        )
    )
    return tuple([Tensor(o) for o in op_results])


@register_dispatch
def var_mean(
    self_: Tensor, unbiased: Union[Torch_Value[Torch_BoolType], builtins.bool] = True
) -> Tuple[Tensor, Tensor]:
    assert check_argument_types()
    if isinstance(unbiased, builtins.bool):
        unbiased = torch_dialect.ConstantBoolOp(unbiased).result
    result0_type = Torch_NonValueTensorType()
    result1_type = Torch_NonValueTensorType()
    op_results = get_op_results_or_values(
        torch_dialect.AtenVarMeanOp(result0_type, result1_type, self_, unbiased)
    )
    return tuple([Tensor(o) for o in op_results])


def view(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenViewOp(result0_type, self_, size))


@register_dispatch
def view_copy(
    self_: Tensor,
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenViewCopyOp(result0_type, self_, size))


# overload dtype
@register_dispatch
def view_copy(
    self_: Tensor, dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype]
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenViewCopyDtypeOp(result0_type, self_, dtype))


# overload self
@register_dispatch
def where(condition: Tensor, self_: Tensor, other: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenWhereSelfOp(result0_type, condition, self_, other))


# overload Scalar
@register_dispatch
def where(condition: Tensor, self_: TorchNumber, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(self_, (builtins.int, builtins.float)):
        self_ = torch_dialect.ConstantNumberOp(self_).result
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenWhereScalarOp(result0_type, condition, self_, other)
    )


# overload ScalarOther
@register_dispatch
def where(condition: Tensor, self_: Tensor, other: TorchNumber) -> Tensor:
    assert check_argument_types()
    if isinstance(other, (builtins.int, builtins.float)):
        other = torch_dialect.ConstantNumberOp(other).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenWhereScalarOtherOp(result0_type, condition, self_, other)
    )


# overload ScalarSelf
@register_dispatch
def where(condition: Tensor, self_: TorchNumber, other: Tensor) -> Tensor:
    assert check_argument_types()
    if isinstance(self_, (builtins.int, builtins.float)):
        self_ = torch_dialect.ConstantNumberOp(self_).result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenWhereScalarSelfOp(result0_type, condition, self_, other)
    )


def zero(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenZeroOp(result0_type, self_))


def zero_(self_: Tensor) -> Tensor:
    assert check_argument_types()
    result0_type = Torch_NonValueTensorType()
    return Tensor(torch_dialect.AtenZero_Op(result0_type, self_))


def zeros(
    size: Union[
        Sequence[Union[Torch_Value[Torch_IntType], builtins.int]],
        Torch_List[Torch_IntType],
    ],
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(size, (builtins.list, builtins.tuple)) and builtins.len(size):
        size = builtins.list(size)
        for i, a in enumerate(size):
            if not isinstance(a, builtins.int):
                assert isinstance(a, Torch_Value), f"wrong type: {a}; should be int"
            else:
                size[i] = torch_dialect.ConstantIntOp(a).result
        ls_type = Torch_List.of(Torch_IntType())
        size = torch_dialect.PrimListConstructOp(ls_type, size).result
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenZerosOp(result0_type, size, dtype, layout, device, pin_memory)
    )


def zeros_like(
    self_: Tensor,
    dtype: Union[Torch_Value[Torch_IntType], builtins.int, pi_dtype, None] = None,
    layout: Union[Torch_Value[Torch_IntType], builtins.int, pi_layout, None] = None,
    device: Union[Torch_Value[Torch_DeviceType], builtins.str, None] = None,
    pin_memory: Union[Torch_Value[Torch_BoolType], builtins.bool, None] = None,
    memory_format: Union[
        Torch_Value[Torch_IntType], builtins.int, pi_memory_format, None
    ] = None,
) -> Tensor:
    assert check_argument_types()
    if isinstance(dtype, pi_dtype):
        dtype = dtype.value
    if isinstance(dtype, builtins.int):
        dtype = torch_dialect.ConstantIntOp(dtype).result
    if dtype is None:
        dtype = torch_dialect.ConstantNoneOp().result
    if isinstance(layout, pi_layout):
        layout = layout.value
    if isinstance(layout, builtins.int):
        layout = torch_dialect.ConstantIntOp(layout).result
    if layout is None:
        layout = torch_dialect.ConstantNoneOp().result
    if isinstance(device, builtins.str):
        device = torch_dialect.ConstantStrOp(device).result
    if device is None:
        device = torch_dialect.ConstantNoneOp().result
    if isinstance(pin_memory, builtins.bool):
        pin_memory = torch_dialect.ConstantBoolOp(pin_memory).result
    if pin_memory is None:
        pin_memory = torch_dialect.ConstantNoneOp().result
    if isinstance(memory_format, pi_memory_format):
        memory_format = memory_format.value
    if isinstance(memory_format, builtins.int):
        memory_format = torch_dialect.ConstantIntOp(memory_format).result
    if memory_format is None:
        memory_format = torch_dialect.ConstantNoneOp().result
    result0_type = Torch_NonValueTensorType()
    return Tensor(
        torch_dialect.AtenZerosLikeOp(
            result0_type, self_, dtype, layout, device, pin_memory, memory_format
        )
    )


__all__ = [
    "Bool",
    "Delete",
    "Float",
    "FloatImplicit",
    "Int",
    "IntImplicit",
    "NumToTensor",
    "RaiseException",
    "TupleIndex",
    "Uninitialized",
    "__and__",
    "__contains__",
    "__derive_index",
    "__getitem__",
    "__is__",
    "__isnot__",
    "__not__",
    "__range_length",
    "_convolution",
    "_embedding_bag",
    "_index_put_impl",
    "_index_put_impl_",
    "_log_softmax",
    "_log_softmax_backward_data",
    "_reshape_alias",
    "_reshape_alias_copy",
    "_set_item",
    "_shape_as_tensor",
    "_softmax",
    "_softmax_backward_data",
    "_to_copy",
    "_unsafe_view",
    "abs",
    "abs_",
    "adaptive_avg_pool2d",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_",
    "addcmul",
    "addcmul_",
    "addmm",
    "alias_copy",
    "all",
    "amax",
    "any",
    "append",
    "arange",
    "argmax",
    "as_strided_copy",
    "as_strided_scatter",
    "atan2",
    "atan2_",
    "avg_pool2d",
    "baddbmm",
    "baddbmm_",
    "batch_norm",
    "bernoulli",
    "bernoulli_",
    "bincount",
    "bitwise_and",
    "bitwise_and_",
    "bitwise_not",
    "bitwise_not_",
    "bitwise_or",
    "bitwise_or_",
    "bitwise_xor",
    "bitwise_xor_",
    "bmm",
    "broadcast_to",
    "bucketize",
    "cat",
    "ceil",
    "ceil_",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clone",
    "constant_pad_nd",
    "contiguous",
    "conv2d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convert_element_type",
    "convolution",
    "convolution_backward",
    "convolution_backward_overrideable",
    "convolution_overrideable",
    "copy",
    "copy_",
    "cos",
    "cos_",
    "cpu",
    "cumsum",
    "detach",
    "detach_copy",
    "device",
    "diagonal_copy",
    "diagonal_scatter",
    "dim",
    "div",
    "div_",
    "dropout",
    "dropout_",
    "dtype",
    "embedding",
    "embedding_bag",
    "embedding_dense_backward",
    "empty",
    "empty_like",
    "eq",
    "eq_",
    "erf",
    "erf_",
    "exp",
    "exp_",
    "expand",
    "expand_as",
    "expand_copy",
    "expm1",
    "expm1_",
    "fft_fft",
    "fill",
    "fill_",
    "flatten",
    "flip",
    "floor",
    "floor_",
    "floor_divide",
    "floordiv",
    "fmod",
    "fmod_",
    "frobenius_norm",
    "full",
    "full_like",
    "gather",
    "ge",
    "ge_",
    "gelu",
    "gelu_backward",
    "get",
    "gt",
    "gt_",
    "hardsigmoid",
    "hardsigmoid_",
    "hardswish",
    "hardswish_",
    "hardtanh",
    "hardtanh_",
    "hardtanh_backward",
    "index",
    "index_put",
    "index_put_",
    "index_select",
    "insert",
    "is_floating_point",
    "item",
    "join",
    "keys",
    "layer_norm",
    "layout",
    "le",
    "le_",
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_backward",
    "len",
    "lerp",
    "lerp_",
    "lift_fresh_copy",
    "linear",
    "list",
    "log",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log_",
    "log_softmax",
    "logical_and",
    "logical_and_",
    "logical_not",
    "logical_not_",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
    "logsumexp",
    "lt",
    "lt_",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "matmul",
    "max",
    "max_pool2d",
    "max_pool2d_with_indices",
    "max_pool2d_with_indices_backward",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mish",
    "mm",
    "mse_loss",
    "mul",
    "mul_",
    "mv",
    "narrow",
    "native_batch_norm",
    "native_batch_norm_backward",
    "native_dropout",
    "native_dropout_backward",
    "native_layer_norm",
    "native_layer_norm_backward",
    "ne",
    "ne_",
    "neg",
    "neg_",
    "new_empty",
    "new_empty_strided",
    "new_ones",
    "new_zeros",
    "nll_loss_backward",
    "nll_loss_forward",
    "norm",
    "numel",
    "numpy_T",
    "ones",
    "ones_like",
    "pad",
    "permute",
    "permute_copy",
    "pow",
    "prelu",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu6",
    "relu6_",
    "relu_",
    "remainder",
    "repeat",
    "reshape",
    "resize_",
    "roll",
    "round",
    "round_",
    "rsqrt",
    "rsqrt_",
    "rsub",
    "scatter_add",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "select",
    "select_copy",
    "select_scatter",
    "sigmoid",
    "sigmoid_",
    "silu",
    "silu_",
    "sin",
    "sin_",
    "size",
    "slice",
    "slice_copy",
    "slice_scatter",
    "softmax",
    "softplus",
    "sort",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "squeeze",
    "squeeze_copy",
    "stack",
    "std",
    "str",
    "sub",
    "sub_",
    "sum",
    "t",
    "t_copy",
    "tanh",
    "tanh_",
    "tanh_backward",
    "tensor",
    "threshold",
    "threshold_",
    "threshold_backward",
    "to",
    "topk",
    "transpose",
    "transpose_copy",
    "triu",
    "triu_",
    "type_as",
    "unchecked_cast",
    "unfold_copy",
    "uniform",
    "uniform_",
    "unsqueeze",
    "unsqueeze_",
    "unsqueeze_copy",
    "upsample_nearest2d",
    "upsample_nearest2d_backward",
    "var",
    "var_mean",
    "vector_norm",
    "view",
    "view_copy",
    "where",
    "zero",
    "zero_",
    "zeros",
    "zeros_like",
]
