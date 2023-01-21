import builtins
from typing import List, Optional, Any, Dict, Tuple, Union

from ._tensor import Tensor, ScalarImplicit
from .types_ import (
    is_a_torch_tensor,
    Device,
    Generator,
    dtype as pi_dtype,
    layout as pi_layout,
    memory_format as pi_memory_format,
    TorchBool,
    TorchInt,
    TorchFloat,
    TorchNumber,
    TorchString,
)
from .dispatcher import dispatch
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.dialects._ods_common import (
    get_op_results_or_values,
)
# overload Tensor
@dispatch
def Bool(a: Tensor) -> TorchBool:
    assert is_a_torch_tensor(a), f'`a` should be a Tensor but is {type(a)}'
    return TorchBool(torch_dialect.AtenBoolTensorOp(a).result)
    
# overload float
@dispatch
def Bool(a: Union[TorchFloat, float]) -> TorchBool:
    return TorchBool(torch_dialect.AtenBoolFloatOp(a).result)
    
# overload int
@dispatch
def Bool(a: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenBoolIntOp(a).result)
    
# overload Dict_str
def Delete(self_: Dict[str, Tensor], key: Union[TorchString, str]) -> None:
    torch_dialect.AtenDeleteDictStrOp(self_, key)
    
# overload Tensor
@dispatch
def Float(a: Tensor) -> TorchFloat:
    assert is_a_torch_tensor(a), f'`a` should be a Tensor but is {type(a)}'
    return TorchFloat(torch_dialect.AtenFloatTensorOp(a).result)
    
# overload Scalar
@dispatch
def Float(a: TorchNumber) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenFloatScalarOp(a).result)
    
# overload str
@dispatch
def Float(a: Union[TorchString, str]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenFloatStrOp(a).result)
    
def FloatImplicit(a: Tensor) -> TorchFloat:
    assert is_a_torch_tensor(a), f'`a` should be a Tensor but is {type(a)}'
    return TorchFloat(torch_dialect.AtenFloatImplicitOp(a).result)
    
# overload Tensor
@dispatch
def Int(a: Tensor) -> TorchInt:
    assert is_a_torch_tensor(a), f'`a` should be a Tensor but is {type(a)}'
    return TorchInt(torch_dialect.AtenIntTensorOp(a).result)
    
# overload float
@dispatch
def Int(a: Union[TorchFloat, float]) -> TorchInt:
    return TorchInt(torch_dialect.AtenIntFloatOp(a).result)
    
# overload Scalar
@dispatch
def Int(a: TorchNumber) -> TorchInt:
    return TorchInt(torch_dialect.AtenIntScalarOp(a).result)
    
# overload bool
@dispatch
def Int(a: Union[TorchBool, bool]) -> TorchInt:
    return TorchInt(torch_dialect.AtenIntBoolOp(a).result)
    
def IntImplicit(a: Tensor) -> TorchInt:
    assert is_a_torch_tensor(a), f'`a` should be a Tensor but is {type(a)}'
    return TorchInt(torch_dialect.AtenIntImplicitOp(a).result)
    
# overload Scalar
def NumToTensor(a: TorchNumber) -> Tensor:
    return Tensor(torch_dialect.PrimNumToTensorScalarOp(a))
    
def RaiseException(msg: Union[TorchString, str], cls: Optional[Union[TorchString, str]] = None) -> None:
    torch_dialect.PrimRaiseExceptionOp(msg, cls)
    
def TupleIndex(tup: Any, i: Union[TorchInt, int]) -> Any:
    return Any(torch_dialect.PrimTupleIndexOp(tup, i).result)
    
def Uninitialized() -> Any:
    return Any(torch_dialect.PrimUninitializedOp().result)
    
# overload Tensor
@dispatch
def __and__(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.Aten__And__TensorOp(self_, other))
    
# overload bool
@dispatch
def __and__(a: Union[TorchBool, bool], b: Union[TorchBool, bool]) -> TorchBool:
    return TorchBool(torch_dialect.Aten__And__BoolOp(a, b).result)
    
# overload str
@dispatch
def __contains__(dict: Dict[str, Tensor], key: Union[TorchString, str]) -> TorchBool:
    return TorchBool(torch_dialect.Aten__Contains__StrOp(dict, key).result)
    
# overload int_list
@dispatch
def __contains__(l: List[Union[TorchInt, int]], item: Union[TorchInt, int]) -> TorchBool:
    if not isinstance(l, (tuple, builtins.list)):
        l = [l]
    return TorchBool(torch_dialect.Aten__Contains__IntListOp(l, item).result)
    
def __derive_index(index: Union[TorchInt, int], start: Union[TorchInt, int], step: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.Aten__DeriveIndexOp(index, start, step).result)
    
# overload Dict_str
@dispatch
def __getitem__(self_: Dict[str, Tensor], key: Union[TorchString, str]) -> Tensor:
    return Tensor(torch_dialect.Aten__Getitem__DictStrOp(self_, key).result)
    
# overload t
@dispatch
def __getitem__(list_: List[Tensor], idx: Union[TorchInt, int]) -> Tensor:
    return Tensor(torch_dialect.Aten__Getitem__TOp(list_, idx).result)
    
def __is__(self_: Tensor, obj: Tensor) -> TorchBool:
    return TorchBool(torch_dialect.Aten__Is__Op(self_, obj).result)
    
def __isnot__(self_: Tensor, obj: Tensor) -> TorchBool:
    return TorchBool(torch_dialect.Aten__Isnot__Op(self_, obj).result)
    
def __not__(self_: Union[TorchBool, bool]) -> TorchBool:
    return TorchBool(torch_dialect.Aten__Not__Op(self_).result)
    
def __range_length(lo: Union[TorchInt, int], hi: Union[TorchInt, int], step: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.Aten__RangeLengthOp(lo, hi, step).result)
    
@dispatch
def _convolution(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[Union[TorchInt, int]], padding: List[Union[TorchInt, int]], dilation: List[Union[TorchInt, int]], transposed: Union[TorchBool, bool], output_padding: List[Union[TorchInt, int]], groups: Union[TorchInt, int], benchmark: Union[TorchBool, bool], deterministic: Union[TorchBool, bool], cudnn_enabled: Union[TorchBool, bool], allow_tf32: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    return Tensor(torch_dialect.Aten_ConvolutionOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32))
    
# overload deprecated
@dispatch
def _convolution(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[Union[TorchInt, int]], padding: List[Union[TorchInt, int]], dilation: List[Union[TorchInt, int]], transposed: Union[TorchBool, bool], output_padding: List[Union[TorchInt, int]], groups: Union[TorchInt, int], benchmark: Union[TorchBool, bool], deterministic: Union[TorchBool, bool], cudnn_enabled: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    return Tensor(torch_dialect.Aten_ConvolutionDeprecatedOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled))
    
def _embedding_bag(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: Union[TorchBool, bool] = False, mode: Union[TorchInt, int] = 0, sparse: Union[TorchBool, bool] = False, per_sample_weights: Optional[Tensor] = None, include_last_offset: Union[TorchBool, bool] = False, padding_idx: Union[TorchInt, int] = -1) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    assert is_a_torch_tensor(indices), f'`indices` should be a Tensor but is {type(indices)}'
    assert is_a_torch_tensor(offsets), f'`offsets` should be a Tensor but is {type(offsets)}'
    if per_sample_weights is not None:
        assert is_a_torch_tensor(per_sample_weights), f'`per_sample_weights` should be a Tensor but is {type(per_sample_weights)}'
    op_results = get_op_results_or_values(torch_dialect.Aten_EmbeddingBagOp(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def _index_put_impl(self_: Tensor, indices: List[Optional[Tensor]], values: Tensor, accumulate: Union[TorchBool, bool] = False, unsafe: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in indices)
    assert is_a_torch_tensor(values), f'`values` should be a Tensor but is {type(values)}'
    return Tensor(torch_dialect.Aten_IndexPutImplOp(self_, indices, values, accumulate, unsafe))
    
def _index_put_impl_(self_: Tensor, indices: List[Optional[Tensor]], values: Tensor, accumulate: Union[TorchBool, bool] = False, unsafe: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in indices)
    assert is_a_torch_tensor(values), f'`values` should be a Tensor but is {type(values)}'
    return Tensor(torch_dialect.Aten_IndexPutImpl_Op(self_, indices, values, accumulate, unsafe))
    
def _log_softmax(self_: Tensor, dim: Union[TorchInt, int], half_to_float: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.Aten_LogSoftmaxOp(self_, dim, half_to_float))
    
def _log_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: Union[TorchInt, int], input_dtype: pi_dtype) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(output), f'`output` should be a Tensor but is {type(output)}'
    if input_dtype is not None:
        assert isinstance(input_dtype, pi_dtype), f'expected pi_dtype, got {type(input_dtype)}'
        input_dtype = input_dtype.value
    return Tensor(torch_dialect.Aten_LogSoftmaxBackwardDataOp(grad_output, output, dim, input_dtype))
    
def _reshape_alias(self_: Tensor, size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    return Tensor(torch_dialect.Aten_ReshapeAliasOp(self_, size, stride))
    
def _reshape_alias_copy(self_: Tensor, size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    return Tensor(torch_dialect.Aten_ReshapeAliasCopyOp(self_, size, stride))
    
# overload str
@dispatch
def _set_item(l: Dict[str, Tensor], idx: Union[TorchString, str], v: Tensor) -> None:
    torch_dialect.Aten_SetItemStrOp(l, idx, v)
    
# overload t
@dispatch
def _set_item(l: List[Tensor], idx: Union[TorchInt, int], el: Tensor) -> List[Tensor]:
    return List[Tensor](torch_dialect.Aten_SetItemTOp(l, idx, el).result)
    
def _shape_as_tensor(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.Aten_ShapeAsTensorOp(self_))
    
def _softmax(self_: Tensor, dim: Union[TorchInt, int], half_to_float: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.Aten_SoftmaxOp(self_, dim, half_to_float))
    
def _softmax_backward_data(grad_output: Tensor, output: Tensor, dim: Union[TorchInt, int], input_dtype: pi_dtype) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(output), f'`output` should be a Tensor but is {type(output)}'
    if input_dtype is not None:
        assert isinstance(input_dtype, pi_dtype), f'expected pi_dtype, got {type(input_dtype)}'
        input_dtype = input_dtype.value
    return Tensor(torch_dialect.Aten_SoftmaxBackwardDataOp(grad_output, output, dim, input_dtype))
    
def _to_copy(self_: Tensor, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, non_blocking: Union[TorchBool, bool] = False, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.Aten_ToCopyOp(self_, dtype, layout, device, pin_memory, non_blocking, memory_format))
    
def _unsafe_view(self_: Tensor, size: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    return Tensor(torch_dialect.Aten_UnsafeViewOp(self_, size))
    
@dispatch
def abs(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenAbsOp(self_))
    
# overload Scalar
@dispatch
def abs(a: TorchNumber) -> TorchNumber:
    return TorchNumber(torch_dialect.PrimAbsScalarOp(a).result)
    
def abs_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenAbs_Op(self_))
    
def adaptive_avg_pool2d(self_: Tensor, output_size: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(output_size, (tuple, builtins.list)):
        output_size = [output_size]
    return Tensor(torch_dialect.AtenAdaptiveAvgPool2dOp(self_, output_size))
    
# overload Tensor
@dispatch
def add(self_: Tensor, other: Tensor, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenAddTensorOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def add(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenAddScalarOp(self_, other, alpha))
    
# overload t
@dispatch
def add(a: List[Tensor], b: List[Tensor]) -> List[Tensor]:
    return List[Tensor](torch_dialect.AtenAddTOp(a, b).result)
    
# overload str
@dispatch
def add(a: Union[TorchString, str], b: Union[TorchString, str]) -> TorchString:
    return TorchString(torch_dialect.AtenAddStrOp(a, b).result)
    
# overload int
@dispatch
def add(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.AtenAddIntOp(a, b).result)
    
# overload float_int
@dispatch
def add(a: Union[TorchFloat, float], b: Union[TorchInt, int]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenAddFloatIntOp(a, b).result)
    
@dispatch
def add(a: TorchNumber, b: TorchNumber) -> TorchNumber:
    return TorchNumber(torch_dialect.AtenAddOp(a, b).result)
    
# overload Tensor
@dispatch
def add_(self_: Tensor, other: Tensor, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenAdd_TensorOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def add_(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenAdd_ScalarOp(self_, other, alpha))
    
def addcdiv(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(tensor1), f'`tensor1` should be a Tensor but is {type(tensor1)}'
    assert is_a_torch_tensor(tensor2), f'`tensor2` should be a Tensor but is {type(tensor2)}'
    return Tensor(torch_dialect.AtenAddcdivOp(self_, tensor1, tensor2, value))
    
def addcdiv_(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(tensor1), f'`tensor1` should be a Tensor but is {type(tensor1)}'
    assert is_a_torch_tensor(tensor2), f'`tensor2` should be a Tensor but is {type(tensor2)}'
    return Tensor(torch_dialect.AtenAddcdiv_Op(self_, tensor1, tensor2, value))
    
def addcmul(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(tensor1), f'`tensor1` should be a Tensor but is {type(tensor1)}'
    assert is_a_torch_tensor(tensor2), f'`tensor2` should be a Tensor but is {type(tensor2)}'
    return Tensor(torch_dialect.AtenAddcmulOp(self_, tensor1, tensor2, value))
    
def addcmul_(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(tensor1), f'`tensor1` should be a Tensor but is {type(tensor1)}'
    assert is_a_torch_tensor(tensor2), f'`tensor2` should be a Tensor but is {type(tensor2)}'
    return Tensor(torch_dialect.AtenAddcmul_Op(self_, tensor1, tensor2, value))
    
def addmm(self_: Tensor, mat1: Tensor, mat2: Tensor, beta: TorchNumber = 1, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(mat1), f'`mat1` should be a Tensor but is {type(mat1)}'
    assert is_a_torch_tensor(mat2), f'`mat2` should be a Tensor but is {type(mat2)}'
    return Tensor(torch_dialect.AtenAddmmOp(self_, mat1, mat2, beta, alpha))
    
def alias_copy(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenAliasCopyOp(self_))
    
@dispatch
def all(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenAllOp(self_))
    
# overload bool
@dispatch
def all(self_: List[Union[TorchBool, bool]]) -> TorchBool:
    return TorchBool(torch_dialect.AtenAllBoolOp(self_).result)
    
def amax(self_: Tensor, dim: List[Union[TorchInt, int]] = (), keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    return Tensor(torch_dialect.AtenAmaxOp(self_, dim, keepdim))
    
@dispatch
def any(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenAnyOp(self_))
    
# overload dim
@dispatch
def any(self_: Tensor, dim: Union[TorchInt, int], keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenAnyDimOp(self_, dim, keepdim))
    
# overload bool
@dispatch
def any(self_: List[Union[TorchBool, bool]]) -> TorchBool:
    return TorchBool(torch_dialect.AtenAnyBoolOp(self_).result)
    
# overload t
def append(self_: List[Tensor], el: Tensor) -> List[Tensor]:
    return List[Tensor](torch_dialect.AtenAppendTOp(self_, el).result)
    
@dispatch
def arange(end: TorchNumber, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenArangeOp(end, dtype, layout, device, pin_memory))
    
# overload start
@dispatch
def arange(start: TorchNumber, end: TorchNumber, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenArangeStartOp(start, end, dtype, layout, device, pin_memory))
    
# overload start_step
@dispatch
def arange(start: TorchNumber, end: TorchNumber, step: TorchNumber = 1, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenArangeStartStepOp(start, end, step, dtype, layout, device, pin_memory))
    
# overload start_out
@dispatch
def arange(start: TorchNumber, end: TorchNumber, step: TorchNumber = 1, out: Optional[Tensor] = None) -> Tensor:
    assert is_a_torch_tensor(out), f'`out` should be a Tensor but is {type(out)}'
    return Tensor(torch_dialect.AtenArangeStartOutOp(start, end, step, out))
    
def argmax(self_: Tensor, dim: Optional[Union[TorchInt, int]] = None, keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenArgmaxOp(self_, dim, keepdim))
    
def as_strided_copy(self_: Tensor, size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]], storage_offset: Optional[Union[TorchInt, int]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    return Tensor(torch_dialect.AtenAsStridedCopyOp(self_, size, stride, storage_offset))
    
def as_strided_scatter(self_: Tensor, src: Tensor, size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]], storage_offset: Optional[Union[TorchInt, int]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(src), f'`src` should be a Tensor but is {type(src)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    return Tensor(torch_dialect.AtenAsStridedScatterOp(self_, src, size, stride, storage_offset))
    
def atan2(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenAtan2Op(self_, other))
    
def atan2_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenAtan2_Op(self_, other))
    
def avg_pool2d(self_: Tensor, kernel_size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]] = (), padding: List[Union[TorchInt, int]] = (0, 0), ceil_mode: Union[TorchBool, bool] = False, count_include_pad: Union[TorchBool, bool] = True, divisor_override: Optional[Union[TorchInt, int]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(kernel_size, (tuple, builtins.list)):
        kernel_size = [kernel_size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    return Tensor(torch_dialect.AtenAvgPool2dOp(self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override))
    
def baddbmm(self_: Tensor, batch1: Tensor, batch2: Tensor, beta: TorchNumber = 1, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(batch1), f'`batch1` should be a Tensor but is {type(batch1)}'
    assert is_a_torch_tensor(batch2), f'`batch2` should be a Tensor but is {type(batch2)}'
    return Tensor(torch_dialect.AtenBaddbmmOp(self_, batch1, batch2, beta, alpha))
    
def baddbmm_(self_: Tensor, batch1: Tensor, batch2: Tensor, beta: TorchNumber = 1, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(batch1), f'`batch1` should be a Tensor but is {type(batch1)}'
    assert is_a_torch_tensor(batch2), f'`batch2` should be a Tensor but is {type(batch2)}'
    return Tensor(torch_dialect.AtenBaddbmm_Op(self_, batch1, batch2, beta, alpha))
    
def batch_norm(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: Union[TorchBool, bool], momentum: Union[TorchFloat, float], eps: Union[TorchFloat, float], cudnn_enabled: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    if weight is not None:
        assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if running_mean is not None:
        assert is_a_torch_tensor(running_mean), f'`running_mean` should be a Tensor but is {type(running_mean)}'
    if running_var is not None:
        assert is_a_torch_tensor(running_var), f'`running_var` should be a Tensor but is {type(running_var)}'
    return Tensor(torch_dialect.AtenBatchNormOp(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled))
    
@dispatch
def bernoulli(self_: Tensor, generator: Optional[Generator] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenBernoulliOp(self_, generator))
    
# overload Tensor
@dispatch
def bernoulli(self_: Tensor, p: Tensor, generator: Optional[Generator] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(p), f'`p` should be a Tensor but is {type(p)}'
    return Tensor(torch_dialect.AtenBernoulliTensorOp(self_, p, generator))
    
# overload float
@dispatch
def bernoulli_(self_: Tensor, p: Union[TorchFloat, float] = 0.5, generator: Optional[Generator] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenBernoulli_FloatOp(self_, p, generator))
    
# overload Tensor
@dispatch
def bernoulli_(self_: Tensor, p: Tensor, generator: Optional[Generator] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(p), f'`p` should be a Tensor but is {type(p)}'
    return Tensor(torch_dialect.AtenBernoulli_TensorOp(self_, p, generator))
    
def bincount(self_: Tensor, weights: Optional[Tensor] = None, minlength: Union[TorchInt, int] = 0) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if weights is not None:
        assert is_a_torch_tensor(weights), f'`weights` should be a Tensor but is {type(weights)}'
    return Tensor(torch_dialect.AtenBincountOp(self_, weights, minlength))
    
# overload Tensor
def bitwise_and(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenBitwiseAndTensorOp(self_, other))
    
# overload Tensor
def bitwise_and_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenBitwiseAnd_TensorOp(self_, other))
    
def bitwise_not(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenBitwiseNotOp(self_))
    
def bitwise_not_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenBitwiseNot_Op(self_))
    
# overload Tensor
def bitwise_or(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenBitwiseOrTensorOp(self_, other))
    
# overload Tensor
def bitwise_or_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenBitwiseOr_TensorOp(self_, other))
    
# overload Tensor
def bitwise_xor(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenBitwiseXorTensorOp(self_, other))
    
# overload Tensor
def bitwise_xor_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenBitwiseXor_TensorOp(self_, other))
    
def bmm(self_: Tensor, mat2: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(mat2), f'`mat2` should be a Tensor but is {type(mat2)}'
    return Tensor(torch_dialect.AtenBmmOp(self_, mat2))
    
def broadcast_to(self_: Tensor, size: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    return Tensor(torch_dialect.AtenBroadcastToOp(self_, size))
    
# overload Tensor
def bucketize(self_: Tensor, boundaries: Tensor, out_int32: Union[TorchBool, bool] = False, right: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(boundaries), f'`boundaries` should be a Tensor but is {type(boundaries)}'
    return Tensor(torch_dialect.AtenBucketizeTensorOp(self_, boundaries, out_int32, right))
    
def cat(tensors: List[Tensor], dim: Union[TorchInt, int] = 0) -> Tensor:
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in tensors)
    return Tensor(torch_dialect.AtenCatOp(tensors, dim))
    
@dispatch
def ceil(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenCeilOp(self_))
    
# overload Scalar
@dispatch
def ceil(a: TorchNumber) -> TorchNumber:
    return TorchNumber(torch_dialect.AtenCeilScalarOp(a).result)
    
# overload float
@dispatch
def ceil(a: Union[TorchFloat, float]) -> TorchInt:
    return TorchInt(torch_dialect.AtenCeilFloatOp(a).result)
    
def ceil_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenCeil_Op(self_))
    
def clamp(self_: Tensor, min: Optional[TorchNumber] = None, max: Optional[TorchNumber] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenClampOp(self_, min, max))
    
def clamp_(self_: Tensor, min: Optional[TorchNumber] = None, max: Optional[TorchNumber] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenClamp_Op(self_, min, max))
    
def clamp_max(self_: Tensor, max: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenClampMaxOp(self_, max))
    
def clamp_max_(self_: Tensor, max: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenClampMax_Op(self_, max))
    
def clamp_min(self_: Tensor, min: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenClampMinOp(self_, min))
    
def clamp_min_(self_: Tensor, min: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenClampMin_Op(self_, min))
    
def clone(self_: Tensor, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenCloneOp(self_, memory_format))
    
def constant_pad_nd(self_: Tensor, pad: List[Union[TorchInt, int]], value: TorchNumber = 0) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(pad, (tuple, builtins.list)):
        pad = [pad]
    return Tensor(torch_dialect.AtenConstantPadNdOp(self_, pad, value))
    
def contiguous(self_: Tensor, memory_format: pi_memory_format) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenContiguousOp(self_, memory_format))
    
def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[Union[TorchInt, int]] = (1, 1), padding: List[Union[TorchInt, int]] = (0, 0), dilation: List[Union[TorchInt, int]] = (1, 1), groups: Union[TorchInt, int] = 1) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    return Tensor(torch_dialect.AtenConv2dOp(input, weight, bias, stride, padding, dilation, groups))
    
def conv_transpose1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[Union[TorchInt, int]] = (1), padding: List[Union[TorchInt, int]] = (0), output_padding: List[Union[TorchInt, int]] = (0), groups: Union[TorchInt, int] = 1, dilation: List[Union[TorchInt, int]] = (1)) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    return Tensor(torch_dialect.AtenConvTranspose1dOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
# overload input
def conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[Union[TorchInt, int]] = (1, 1), padding: List[Union[TorchInt, int]] = (0, 0), output_padding: List[Union[TorchInt, int]] = (0, 0), groups: Union[TorchInt, int] = 1, dilation: List[Union[TorchInt, int]] = (1, 1)) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    return Tensor(torch_dialect.AtenConvTranspose2dInputOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
# overload input
def conv_transpose3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[Union[TorchInt, int]] = (1, 1, 1), padding: List[Union[TorchInt, int]] = (0, 0, 0), output_padding: List[Union[TorchInt, int]] = (0, 0, 0), groups: Union[TorchInt, int] = 1, dilation: List[Union[TorchInt, int]] = (1, 1, 1)) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    return Tensor(torch_dialect.AtenConvTranspose3dInputOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
def convert_element_type(a: Tensor, dtype: pi_dtype) -> Tensor:
    assert is_a_torch_tensor(a), f'`a` should be a Tensor but is {type(a)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.PrimsConvertElementTypeOp(a, dtype))
    
def convolution(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[Union[TorchInt, int]], padding: List[Union[TorchInt, int]], dilation: List[Union[TorchInt, int]], transposed: Union[TorchBool, bool], output_padding: List[Union[TorchInt, int]], groups: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    return Tensor(torch_dialect.AtenConvolutionOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups))
    
def convolution_backward(grad_output: Tensor, input: Tensor, weight: Tensor, bias_sizes: Optional[List[Union[TorchInt, int]]], stride: List[Union[TorchInt, int]], padding: List[Union[TorchInt, int]], dilation: List[Union[TorchInt, int]], transposed: Union[TorchBool, bool], output_padding: List[Union[TorchInt, int]], groups: Union[TorchInt, int], output_mask: List[Union[TorchBool, bool]]) -> Tuple[Tensor, Tensor, Tensor]:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias_sizes is not None and not isinstance(bias_sizes, (tuple, builtins.list)):
        bias_sizes = [bias_sizes]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    op_results = get_op_results_or_values(torch_dialect.AtenConvolutionBackwardOp(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def convolution_backward_overrideable(grad_output: Tensor, input: Tensor, weight: Tensor, stride: List[Union[TorchInt, int]], padding: List[Union[TorchInt, int]], dilation: List[Union[TorchInt, int]], transposed: Union[TorchBool, bool], output_padding: List[Union[TorchInt, int]], groups: Union[TorchInt, int], output_mask: List[Union[TorchBool, bool]]) -> Tuple[Tensor, Tensor, Tensor]:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    op_results = get_op_results_or_values(torch_dialect.AtenConvolutionBackwardOverrideableOp(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def convolution_overrideable(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[Union[TorchInt, int]], padding: List[Union[TorchInt, int]], dilation: List[Union[TorchInt, int]], transposed: Union[TorchBool, bool], output_padding: List[Union[TorchInt, int]], groups: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    if not isinstance(output_padding, (tuple, builtins.list)):
        output_padding = [output_padding]
    return Tensor(torch_dialect.AtenConvolutionOverrideableOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups))
    
def copy(self_: Tensor, src: Tensor, non_blocking: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(src), f'`src` should be a Tensor but is {type(src)}'
    return Tensor(torch_dialect.AtenCopyOp(self_, src, non_blocking))
    
def copy_(self_: Tensor, src: Tensor, non_blocking: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(src), f'`src` should be a Tensor but is {type(src)}'
    return Tensor(torch_dialect.AtenCopy_Op(self_, src, non_blocking))
    
def cos(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenCosOp(self_))
    
def cos_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenCos_Op(self_))
    
def cpu(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenCpuOp(self_))
    
def cumsum(self_: Tensor, dim: Union[TorchInt, int], dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenCumsumOp(self_, dim, dtype))
    
def detach(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenDetachOp(self_))
    
def detach_copy(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenDetachCopyOp(self_))
    
def diagonal_copy(self_: Tensor, offset: Union[TorchInt, int] = 0, dim1: Union[TorchInt, int] = 0, dim2: Union[TorchInt, int] = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenDiagonalCopyOp(self_, offset, dim1, dim2))
    
def diagonal_scatter(self_: Tensor, src: Tensor, offset: Union[TorchInt, int] = 0, dim1: Union[TorchInt, int] = 0, dim2: Union[TorchInt, int] = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(src), f'`src` should be a Tensor but is {type(src)}'
    return Tensor(torch_dialect.AtenDiagonalScatterOp(self_, src, offset, dim1, dim2))
    
def dim(self_: Tensor) -> TorchInt:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return TorchInt(torch_dialect.AtenDimOp(self_).result)
    
# overload Tensor
@dispatch
def div(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenDivTensorOp(self_, other))
    
# overload Scalar
@dispatch
def div(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenDivScalarOp(self_, other))
    
# overload Tensor_mode
@dispatch
def div(self_: Tensor, other: Tensor, rounding_mode: Optional[Union[TorchString, str]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenDivTensorModeOp(self_, other, rounding_mode))
    
# overload int
@dispatch
def div(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenDivIntOp(a, b).result)
    
# overload float
@dispatch
def div(a: Union[TorchFloat, float], b: Union[TorchFloat, float]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenDivFloatOp(a, b).result)
    
@dispatch
def div(a: TorchNumber, b: TorchNumber) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenDivOp(a, b).result)
    
# overload Tensor
@dispatch
def div_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenDiv_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def div_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenDiv_ScalarOp(self_, other))
    
# overload Tensor_mode
@dispatch
def div_(self_: Tensor, other: Tensor, rounding_mode: Optional[Union[TorchString, str]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenDiv_TensorModeOp(self_, other, rounding_mode))
    
def dropout(input: Tensor, p: Union[TorchFloat, float], train: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    return Tensor(torch_dialect.AtenDropoutOp(input, p, train))
    
def dropout_(self_: Tensor, p: Union[TorchFloat, float], train: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenDropout_Op(self_, p, train))
    
def embedding(weight: Tensor, indices: Tensor, padding_idx: Union[TorchInt, int] = -1, scale_grad_by_freq: Union[TorchBool, bool] = False, sparse: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    assert is_a_torch_tensor(indices), f'`indices` should be a Tensor but is {type(indices)}'
    return Tensor(torch_dialect.AtenEmbeddingOp(weight, indices, padding_idx, scale_grad_by_freq, sparse))
    
# overload padding_idx
def embedding_bag(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: Union[TorchBool, bool], mode: Union[TorchInt, int], sparse: Union[TorchBool, bool], per_sample_weights: Optional[Tensor], include_last_offset: Union[TorchBool, bool], padding_idx: Optional[Union[TorchInt, int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    assert is_a_torch_tensor(indices), f'`indices` should be a Tensor but is {type(indices)}'
    assert is_a_torch_tensor(offsets), f'`offsets` should be a Tensor but is {type(offsets)}'
    if per_sample_weights is not None:
        assert is_a_torch_tensor(per_sample_weights), f'`per_sample_weights` should be a Tensor but is {type(per_sample_weights)}'
    op_results = get_op_results_or_values(torch_dialect.AtenEmbeddingBagPaddingIdxOp(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def embedding_dense_backward(grad_output: Tensor, indices: Tensor, num_weights: Union[TorchInt, int], padding_idx: Union[TorchInt, int], scale_grad_by_freq: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(indices), f'`indices` should be a Tensor but is {type(indices)}'
    return Tensor(torch_dialect.AtenEmbeddingDenseBackwardOp(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq))
    
# overload memory_format
def empty(size: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenEmptyMemoryFormatOp(size, dtype, layout, device, pin_memory, memory_format))
    
def empty_like(self_: Tensor, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenEmptyLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    
# overload Tensor
@dispatch
def eq(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenEqTensorOp(self_, other))
    
# overload Scalar
@dispatch
def eq(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenEqScalarOp(self_, other))
    
# overload int_list
@dispatch
def eq(a: List[Union[TorchInt, int]], b: List[Union[TorchInt, int]]) -> TorchBool:
    if not isinstance(a, (tuple, builtins.list)):
        a = [a]
    if not isinstance(b, (tuple, builtins.list)):
        b = [b]
    return TorchBool(torch_dialect.AtenEqIntListOp(a, b).result)
    
# overload str
@dispatch
def eq(a: Union[TorchString, str], b: Union[TorchString, str]) -> TorchBool:
    return TorchBool(torch_dialect.AtenEqStrOp(a, b).result)
    
# overload int
@dispatch
def eq(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenEqIntOp(a, b).result)
    
# overload float
@dispatch
def eq(a: Union[TorchFloat, float], b: Union[TorchFloat, float]) -> TorchBool:
    return TorchBool(torch_dialect.AtenEqFloatOp(a, b).result)
    
# overload device
@dispatch
def eq(a: Device, b: Device) -> TorchBool:
    return TorchBool(torch_dialect.AtenEqDeviceOp(a, b).result)
    
# overload Tensor
@dispatch
def eq_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenEq_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def eq_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenEq_ScalarOp(self_, other))
    
def erf(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenErfOp(self_))
    
def erf_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenErf_Op(self_))
    
def exp(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenExpOp(self_))
    
def exp_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenExp_Op(self_))
    
def expand(self_: Tensor, size: List[Union[TorchInt, int]], implicit: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    return Tensor(torch_dialect.AtenExpandOp(self_, size, implicit))
    
def expand_as(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenExpandAsOp(self_, other))
    
def expand_copy(self_: Tensor, size: List[Union[TorchInt, int]], implicit: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    return Tensor(torch_dialect.AtenExpandCopyOp(self_, size, implicit))
    
def expm1(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenExpm1Op(self_))
    
def expm1_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenExpm1_Op(self_))
    
def fft_fft(self_: Tensor, n: Optional[Union[TorchInt, int]] = None, dim: Union[TorchInt, int] = -1, norm: Optional[Union[TorchString, str]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFftFftOp(self_, n, dim, norm))
    
# overload Scalar
@dispatch
def fill(self_: Tensor, value: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFillScalarOp(self_, value))
    
# overload Tensor
@dispatch
def fill(self_: Tensor, value: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(value), f'`value` should be a Tensor but is {type(value)}'
    return Tensor(torch_dialect.AtenFillTensorOp(self_, value))
    
# overload Scalar
@dispatch
def fill_(self_: Tensor, value: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFill_ScalarOp(self_, value))
    
# overload Tensor
@dispatch
def fill_(self_: Tensor, value: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(value), f'`value` should be a Tensor but is {type(value)}'
    return Tensor(torch_dialect.AtenFill_TensorOp(self_, value))
    
# overload using_ints
def flatten(self_: Tensor, start_dim: Union[TorchInt, int] = 0, end_dim: Union[TorchInt, int] = -1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFlattenUsingIntsOp(self_, start_dim, end_dim))
    
def flip(self_: Tensor, dims: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(dims, (tuple, builtins.list)):
        dims = [dims]
    return Tensor(torch_dialect.AtenFlipOp(self_, dims))
    
def floor(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFloorOp(self_))
    
def floor_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFloor_Op(self_))
    
@dispatch
def floor_divide(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenFloorDivideOp(self_, other))
    
# overload Scalar
@dispatch
def floor_divide(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFloorDivideScalarOp(self_, other))
    
# overload int
def floordiv(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.AtenFloordivIntOp(a, b).result)
    
# overload Scalar
def fmod(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFmodScalarOp(self_, other))
    
# overload Scalar
def fmod_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenFmod_ScalarOp(self_, other))
    
# overload dim
def frobenius_norm(self_: Tensor, dim: List[Union[TorchInt, int]], keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    return Tensor(torch_dialect.AtenFrobeniusNormDimOp(self_, dim, keepdim))
    
def full(size: List[Union[TorchInt, int]], fill_value: TorchNumber, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenFullOp(size, fill_value, dtype, layout, device, pin_memory))
    
def full_like(self_: Tensor, fill_value: TorchNumber, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenFullLikeOp(self_, fill_value, dtype, layout, device, pin_memory, memory_format))
    
def gather(self_: Tensor, dim: Union[TorchInt, int], index: Tensor, sparse_grad: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(index), f'`index` should be a Tensor but is {type(index)}'
    return Tensor(torch_dialect.AtenGatherOp(self_, dim, index, sparse_grad))
    
# overload Tensor
@dispatch
def ge(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenGeTensorOp(self_, other))
    
# overload Scalar
@dispatch
def ge(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenGeScalarOp(self_, other))
    
# overload int
@dispatch
def ge(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenGeIntOp(a, b).result)
    
# overload float
@dispatch
def ge(a: Union[TorchFloat, float], b: Union[TorchFloat, float]) -> TorchBool:
    return TorchBool(torch_dialect.AtenGeFloatOp(a, b).result)
    
# overload float_int
@dispatch
def ge(a: Union[TorchFloat, float], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenGeFloatIntOp(a, b).result)
    
# overload Tensor
@dispatch
def ge_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenGe_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def ge_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenGe_ScalarOp(self_, other))
    
def gelu(self_: Tensor, approximate: Union[TorchString, str] = "none") -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenGeluOp(self_, approximate))
    
def gelu_backward(grad_output: Tensor, self_: Tensor, approximate: Union[TorchString, str] = "none") -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenGeluBackwardOp(grad_output, self_, approximate))
    
# overload default_str
def get(self_: Dict[str, Tensor], key: Union[TorchString, str], default_value: Tensor) -> Tensor:
    return Tensor(torch_dialect.AtenGetDefaultStrOp(self_, key, default_value).result)
    
# overload Tensor
@dispatch
def gt(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenGtTensorOp(self_, other))
    
# overload Scalar
@dispatch
def gt(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenGtScalarOp(self_, other))
    
# overload int
@dispatch
def gt(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenGtIntOp(a, b).result)
    
# overload float
@dispatch
def gt(a: Union[TorchFloat, float], b: Union[TorchFloat, float]) -> TorchBool:
    return TorchBool(torch_dialect.AtenGtFloatOp(a, b).result)
    
# overload float_int
@dispatch
def gt(a: Union[TorchFloat, float], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenGtFloatIntOp(a, b).result)
    
# overload Tensor
@dispatch
def gt_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenGt_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def gt_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenGt_ScalarOp(self_, other))
    
def hardsigmoid(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenHardsigmoidOp(self_))
    
def hardsigmoid_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenHardsigmoid_Op(self_))
    
def hardswish(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenHardswishOp(self_))
    
def hardswish_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenHardswish_Op(self_))
    
def hardtanh(self_: Tensor, min_val: TorchNumber = -1, max_val: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenHardtanhOp(self_, min_val, max_val))
    
def hardtanh_(self_: Tensor, min_val: TorchNumber = -1, max_val: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenHardtanh_Op(self_, min_val, max_val))
    
# overload Tensor
@dispatch
def index(self_: Tensor, indices: List[Optional[Tensor]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in indices)
    return Tensor(torch_dialect.AtenIndexTensorOp(self_, indices))
    
# overload Tensor_hacked_twin
@dispatch
def index(self_: Tensor, indices: List[Tensor]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in indices)
    return Tensor(torch_dialect.AtenIndexTensorHackedTwinOp(self_, indices))
    
@dispatch
def index_put(self_: Tensor, indices: List[Optional[Tensor]], values: Tensor, accumulate: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in indices)
    assert is_a_torch_tensor(values), f'`values` should be a Tensor but is {type(values)}'
    return Tensor(torch_dialect.AtenIndexPutOp(self_, indices, values, accumulate))
    
# overload hacked_twin
@dispatch
def index_put(self_: Tensor, indices: List[Tensor], values: Tensor, accumulate: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in indices)
    assert is_a_torch_tensor(values), f'`values` should be a Tensor but is {type(values)}'
    return Tensor(torch_dialect.AtenIndexPutHackedTwinOp(self_, indices, values, accumulate))
    
@dispatch
def index_put_(self_: Tensor, indices: List[Optional[Tensor]], values: Tensor, accumulate: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in indices)
    assert is_a_torch_tensor(values), f'`values` should be a Tensor but is {type(values)}'
    return Tensor(torch_dialect.AtenIndexPut_Op(self_, indices, values, accumulate))
    
# overload hacked_twin
@dispatch
def index_put_(self_: Tensor, indices: List[Tensor], values: Tensor, accumulate: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in indices)
    assert is_a_torch_tensor(values), f'`values` should be a Tensor but is {type(values)}'
    return Tensor(torch_dialect.AtenIndexPut_HackedTwinOp(self_, indices, values, accumulate))
    
def index_select(self_: Tensor, dim: Union[TorchInt, int], index: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(index), f'`index` should be a Tensor but is {type(index)}'
    return Tensor(torch_dialect.AtenIndexSelectOp(self_, dim, index))
    
# overload t
def insert(self_: List[Tensor], idx: Union[TorchInt, int], el: Tensor) -> None:
    torch_dialect.AtenInsertTOp(self_, idx, el)
    
def is_floating_point(self_: Tensor) -> TorchBool:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return TorchBool(torch_dialect.AtenIsFloatingPointOp(self_).result)
    
def item(self_: Tensor) -> TorchNumber:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return TorchNumber(torch_dialect.AtenItemOp(self_).result)
    
def join(self_: Union[TorchString, str], values: List[Union[TorchString, str]]) -> TorchString:
    return TorchString(torch_dialect.AtenJoinOp(self_, values).result)
    
# overload str
def keys(self_: Dict[str, Tensor]) -> List[TorchString]:
    return List[TorchString](torch_dialect.AtenKeysStrOp(self_).result)
    
def layer_norm(input: Tensor, normalized_shape: List[Union[TorchInt, int]], weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: Union[TorchFloat, float] = 1.0000000000000001e-05, cudnn_enable: Union[TorchBool, bool] = True) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    if not isinstance(normalized_shape, (tuple, builtins.list)):
        normalized_shape = [normalized_shape]
    if weight is not None:
        assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    return Tensor(torch_dialect.AtenLayerNormOp(input, normalized_shape, weight, bias, eps, cudnn_enable))
    
def layout(a: Tensor) -> TorchInt:
    assert is_a_torch_tensor(a), f'`a` should be a Tensor but is {type(a)}'
    return TorchInt(torch_dialect.PrimLayoutOp(a).result)
    
# overload Tensor
@dispatch
def le(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLeTensorOp(self_, other))
    
# overload Scalar
@dispatch
def le(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLeScalarOp(self_, other))
    
# overload int
@dispatch
def le(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenLeIntOp(a, b).result)
    
# overload Tensor
@dispatch
def le_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLe_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def le_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLe_ScalarOp(self_, other))
    
def leaky_relu(self_: Tensor, negative_slope: TorchNumber = 0.01) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLeakyReluOp(self_, negative_slope))
    
def leaky_relu_(self_: Tensor, negative_slope: TorchNumber = 0.01) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLeakyRelu_Op(self_, negative_slope))
    
def leaky_relu_backward(grad_output: Tensor, self_: Tensor, negative_slope: TorchNumber, self_is_result: Union[TorchBool, bool]) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLeakyReluBackwardOp(grad_output, self_, negative_slope, self_is_result))
    
# overload Tensor
@dispatch
def len(t: Tensor) -> TorchInt:
    assert is_a_torch_tensor(t), f'`t` should be a Tensor but is {type(t)}'
    return TorchInt(torch_dialect.AtenLenTensorOp(t).result)
    
# overload str
@dispatch
def len(s: Union[TorchString, str]) -> TorchInt:
    return TorchInt(torch_dialect.AtenLenStrOp(s).result)
    
# overload t
@dispatch
def len(a: List[Tensor]) -> TorchInt:
    return TorchInt(torch_dialect.AtenLenTOp(a).result)
    
# overload Tensor
def lerp(self_: Tensor, end: Tensor, weight: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(end), f'`end` should be a Tensor but is {type(end)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    return Tensor(torch_dialect.AtenLerpTensorOp(self_, end, weight))
    
# overload Tensor
def lerp_(self_: Tensor, end: Tensor, weight: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(end), f'`end` should be a Tensor but is {type(end)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    return Tensor(torch_dialect.AtenLerp_TensorOp(self_, end, weight))
    
def lift_fresh_copy(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLiftFreshCopyOp(self_))
    
def vector_norm(self_: Tensor, ord: TorchNumber = 2, dim: Optional[List[Union[TorchInt, int]]] = None, keepdim: Union[TorchBool, bool] = False, dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dim is not None and not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenLinalgVectorNormOp(self_, ord, dim, keepdim, dtype))
    
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    return Tensor(torch_dialect.AtenLinearOp(input, weight, bias))
    
# overload t
def list(l: List[Tensor]) -> List[Tensor]:
    return List[Tensor](torch_dialect.AtenListTOp(l).result)
    
@dispatch
def log(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLogOp(self_))
    
# overload int
@dispatch
def log(a: Union[TorchInt, int]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenLogIntOp(a).result)
    
def log1p(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLog1pOp(self_))
    
def log1p_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLog1p_Op(self_))
    
def log2(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLog2Op(self_))
    
def log2_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLog2_Op(self_))
    
def log_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLog_Op(self_))
    
# overload int
def log_softmax(self_: Tensor, dim: Union[TorchInt, int], dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenLogSoftmaxIntOp(self_, dim, dtype))
    
def logical_and(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLogicalAndOp(self_, other))
    
def logical_and_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLogicalAnd_Op(self_, other))
    
def logical_not(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLogicalNotOp(self_))
    
def logical_not_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLogicalNot_Op(self_))
    
def logical_or(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLogicalOrOp(self_, other))
    
def logical_or_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLogicalOr_Op(self_, other))
    
def logical_xor(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLogicalXorOp(self_, other))
    
def logical_xor_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLogicalXor_Op(self_, other))
    
def logsumexp(self_: Tensor, dim: List[Union[TorchInt, int]], keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    return Tensor(torch_dialect.AtenLogsumexpOp(self_, dim, keepdim))
    
# overload Tensor
@dispatch
def lt(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLtTensorOp(self_, other))
    
# overload Scalar
@dispatch
def lt(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLtScalarOp(self_, other))
    
# overload int
@dispatch
def lt(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenLtIntOp(a, b).result)
    
# overload float
@dispatch
def lt(a: Union[TorchFloat, float], b: Union[TorchFloat, float]) -> TorchBool:
    return TorchBool(torch_dialect.AtenLtFloatOp(a, b).result)
    
# overload float_int
@dispatch
def lt(a: Union[TorchFloat, float], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenLtFloatIntOp(a, b).result)
    
# overload Tensor
@dispatch
def lt_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenLt_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def lt_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenLt_ScalarOp(self_, other))
    
# overload Scalar
@dispatch
def masked_fill(self_: Tensor, mask: Tensor, value: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(mask), f'`mask` should be a Tensor but is {type(mask)}'
    return Tensor(torch_dialect.AtenMaskedFillScalarOp(self_, mask, value))
    
# overload Tensor
@dispatch
def masked_fill(self_: Tensor, mask: Tensor, value: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(mask), f'`mask` should be a Tensor but is {type(mask)}'
    assert is_a_torch_tensor(value), f'`value` should be a Tensor but is {type(value)}'
    return Tensor(torch_dialect.AtenMaskedFillTensorOp(self_, mask, value))
    
# overload Scalar
@dispatch
def masked_fill_(self_: Tensor, mask: Tensor, value: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(mask), f'`mask` should be a Tensor but is {type(mask)}'
    return Tensor(torch_dialect.AtenMaskedFill_ScalarOp(self_, mask, value))
    
# overload Tensor
@dispatch
def masked_fill_(self_: Tensor, mask: Tensor, value: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(mask), f'`mask` should be a Tensor but is {type(mask)}'
    assert is_a_torch_tensor(value), f'`value` should be a Tensor but is {type(value)}'
    return Tensor(torch_dialect.AtenMaskedFill_TensorOp(self_, mask, value))
    
def masked_select(self_: Tensor, mask: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(mask), f'`mask` should be a Tensor but is {type(mask)}'
    return Tensor(torch_dialect.AtenMaskedSelectOp(self_, mask))
    
def matmul(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenMatmulOp(self_, other))
    
@dispatch
def max(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenMaxOp(self_))
    
# overload dim
@dispatch
def max(self_: Tensor, dim: Union[TorchInt, int], keepdim: Union[TorchBool, bool] = False) -> Tuple[Tensor, Tensor]:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    op_results = get_op_results_or_values(torch_dialect.AtenMaxDimOp(self_, dim, keepdim))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
# overload self_int
@dispatch
def max(self_: List[Union[TorchInt, int]]) -> TorchInt:
    if not isinstance(self_, (tuple, builtins.list)):
        self_ = [self_]
    return TorchInt(torch_dialect.PrimMaxSelfIntOp(self_).result)
    
# overload int
@dispatch
def max(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.PrimMaxIntOp(a, b).result)
    
def max_pool2d(self_: Tensor, kernel_size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]] = (), padding: List[Union[TorchInt, int]] = (0, 0), dilation: List[Union[TorchInt, int]] = (1, 1), ceil_mode: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(kernel_size, (tuple, builtins.list)):
        kernel_size = [kernel_size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    return Tensor(torch_dialect.AtenMaxPool2dOp(self_, kernel_size, stride, padding, dilation, ceil_mode))
    
def max_pool2d_with_indices(self_: Tensor, kernel_size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]] = (), padding: List[Union[TorchInt, int]] = (0, 0), dilation: List[Union[TorchInt, int]] = (1, 1), ceil_mode: Union[TorchBool, bool] = False) -> Tuple[Tensor, Tensor]:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(kernel_size, (tuple, builtins.list)):
        kernel_size = [kernel_size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    op_results = get_op_results_or_values(torch_dialect.AtenMaxPool2dWithIndicesOp(self_, kernel_size, stride, padding, dilation, ceil_mode))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def max_pool2d_with_indices_backward(grad_output: Tensor, self_: Tensor, kernel_size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]], padding: List[Union[TorchInt, int]], dilation: List[Union[TorchInt, int]], ceil_mode: Union[TorchBool, bool], indices: Tensor) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(kernel_size, (tuple, builtins.list)):
        kernel_size = [kernel_size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if not isinstance(padding, (tuple, builtins.list)):
        padding = [padding]
    if not isinstance(dilation, (tuple, builtins.list)):
        dilation = [dilation]
    assert is_a_torch_tensor(indices), f'`indices` should be a Tensor but is {type(indices)}'
    return Tensor(torch_dialect.AtenMaxPool2dWithIndicesBackwardOp(grad_output, self_, kernel_size, stride, padding, dilation, ceil_mode, indices))
    
def maximum(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenMaximumOp(self_, other))
    
# overload dim
@dispatch
def mean(self_: Tensor, dim: Optional[List[Union[TorchInt, int]]], keepdim: Union[TorchBool, bool] = False, dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dim is not None and not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenMeanDimOp(self_, dim, keepdim, dtype))
    
@dispatch
def mean(self_: Tensor, dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenMeanOp(self_, dtype))
    
# overload self_int
@dispatch
def min(self_: List[Union[TorchInt, int]]) -> TorchInt:
    if not isinstance(self_, (tuple, builtins.list)):
        self_ = [self_]
    return TorchInt(torch_dialect.PrimMinSelfIntOp(self_).result)
    
# overload int
@dispatch
def min(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.PrimMinIntOp(a, b).result)
    
def minimum(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenMinimumOp(self_, other))
    
def mish(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenMishOp(self_))
    
def mm(self_: Tensor, mat2: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(mat2), f'`mat2` should be a Tensor but is {type(mat2)}'
    return Tensor(torch_dialect.AtenMmOp(self_, mat2))
    
def mse_loss(self_: Tensor, target: Tensor, reduction: Union[TorchInt, int] = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(target), f'`target` should be a Tensor but is {type(target)}'
    return Tensor(torch_dialect.AtenMseLossOp(self_, target, reduction))
    
# overload Tensor
@dispatch
def mul(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenMulTensorOp(self_, other))
    
# overload Scalar
@dispatch
def mul(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenMulScalarOp(self_, other))
    
# overload int
@dispatch
def mul(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.AtenMulIntOp(a, b).result)
    
# overload float
@dispatch
def mul(a: Union[TorchFloat, float], b: Union[TorchFloat, float]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenMulFloatOp(a, b).result)
    
# overload Tensor
@dispatch
def mul_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenMul_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def mul_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenMul_ScalarOp(self_, other))
    
def mv(self_: Tensor, vec: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(vec), f'`vec` should be a Tensor but is {type(vec)}'
    return Tensor(torch_dialect.AtenMvOp(self_, vec))
    
def narrow(self_: Tensor, dim: Union[TorchInt, int], start: Union[TorchInt, int], length: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenNarrowOp(self_, dim, start, length))
    
def native_batch_norm(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: Union[TorchBool, bool], momentum: Union[TorchFloat, float], eps: Union[TorchFloat, float]) -> Tuple[Tensor, Tensor, Tensor]:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    if weight is not None:
        assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    if running_mean is not None:
        assert is_a_torch_tensor(running_mean), f'`running_mean` should be a Tensor but is {type(running_mean)}'
    if running_var is not None:
        assert is_a_torch_tensor(running_var), f'`running_var` should be a Tensor but is {type(running_var)}'
    op_results = get_op_results_or_values(torch_dialect.AtenNativeBatchNormOp(input, weight, bias, running_mean, running_var, training, momentum, eps))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def native_batch_norm_backward(grad_out: Tensor, input: Tensor, weight: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], save_mean: Optional[Tensor], save_invstd: Optional[Tensor], train: Union[TorchBool, bool], eps: Union[TorchFloat, float], output_mask: List[Union[TorchBool, bool]]) -> Tuple[Tensor, Tensor, Tensor]:
    assert is_a_torch_tensor(grad_out), f'`grad_out` should be a Tensor but is {type(grad_out)}'
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    if weight is not None:
        assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if running_mean is not None:
        assert is_a_torch_tensor(running_mean), f'`running_mean` should be a Tensor but is {type(running_mean)}'
    if running_var is not None:
        assert is_a_torch_tensor(running_var), f'`running_var` should be a Tensor but is {type(running_var)}'
    if save_mean is not None:
        assert is_a_torch_tensor(save_mean), f'`save_mean` should be a Tensor but is {type(save_mean)}'
    if save_invstd is not None:
        assert is_a_torch_tensor(save_invstd), f'`save_invstd` should be a Tensor but is {type(save_invstd)}'
    op_results = get_op_results_or_values(torch_dialect.AtenNativeBatchNormBackwardOp(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def native_dropout(input: Tensor, p: Union[TorchFloat, float], train: Optional[Union[TorchBool, bool]]) -> Tuple[Tensor, Tensor]:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    op_results = get_op_results_or_values(torch_dialect.AtenNativeDropoutOp(input, p, train))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: Union[TorchFloat, float]) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(mask), f'`mask` should be a Tensor but is {type(mask)}'
    return Tensor(torch_dialect.AtenNativeDropoutBackwardOp(grad_output, mask, scale))
    
def native_layer_norm(input: Tensor, normalized_shape: List[Union[TorchInt, int]], weight: Optional[Tensor], bias: Optional[Tensor], eps: Union[TorchFloat, float]) -> Tuple[Tensor, Tensor, Tensor]:
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    if not isinstance(normalized_shape, (tuple, builtins.list)):
        normalized_shape = [normalized_shape]
    if weight is not None:
        assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    op_results = get_op_results_or_values(torch_dialect.AtenNativeLayerNormOp(input, normalized_shape, weight, bias, eps))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def native_layer_norm_backward(grad_out: Tensor, input: Tensor, normalized_shape: List[Union[TorchInt, int]], mean: Tensor, rstd: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], output_mask: List[Union[TorchBool, bool]]) -> Tuple[Tensor, Tensor, Tensor]:
    assert is_a_torch_tensor(grad_out), f'`grad_out` should be a Tensor but is {type(grad_out)}'
    assert is_a_torch_tensor(input), f'`input` should be a Tensor but is {type(input)}'
    if not isinstance(normalized_shape, (tuple, builtins.list)):
        normalized_shape = [normalized_shape]
    assert is_a_torch_tensor(mean), f'`mean` should be a Tensor but is {type(mean)}'
    assert is_a_torch_tensor(rstd), f'`rstd` should be a Tensor but is {type(rstd)}'
    if weight is not None:
        assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    if bias is not None:
        assert is_a_torch_tensor(bias), f'`bias` should be a Tensor but is {type(bias)}'
    op_results = get_op_results_or_values(torch_dialect.AtenNativeLayerNormBackwardOp(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
# overload Tensor
@dispatch
def ne(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenNeTensorOp(self_, other))
    
# overload Scalar
@dispatch
def ne(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenNeScalarOp(self_, other))
    
# overload int_list
@dispatch
def ne(a: List[Union[TorchInt, int]], b: List[Union[TorchInt, int]]) -> TorchBool:
    if not isinstance(a, (tuple, builtins.list)):
        a = [a]
    if not isinstance(b, (tuple, builtins.list)):
        b = [b]
    return TorchBool(torch_dialect.AtenNeIntListOp(a, b).result)
    
# overload int
@dispatch
def ne(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenNeIntOp(a, b).result)
    
# overload float_int
@dispatch
def ne(a: Union[TorchFloat, float], b: Union[TorchInt, int]) -> TorchBool:
    return TorchBool(torch_dialect.AtenNeFloatIntOp(a, b).result)
    
# overload bool
@dispatch
def ne(a: Union[TorchBool, bool], b: Union[TorchBool, bool]) -> TorchBool:
    return TorchBool(torch_dialect.AtenNeBoolOp(a, b).result)
    
# overload Tensor
@dispatch
def ne_(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenNe_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def ne_(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenNe_ScalarOp(self_, other))
    
@dispatch
def neg(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenNegOp(self_))
    
# overload int
@dispatch
def neg(a: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.AtenNegIntOp(a).result)
    
# overload float
@dispatch
def neg(a: Union[TorchFloat, float]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenNegFloatOp(a).result)
    
def neg_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenNeg_Op(self_))
    
def new_empty(self_: Tensor, size: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenNewEmptyOp(self_, size, dtype, layout, device, pin_memory))
    
def new_empty_strided(self_: Tensor, size: List[Union[TorchInt, int]], stride: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if not isinstance(stride, (tuple, builtins.list)):
        stride = [stride]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenNewEmptyStridedOp(self_, size, stride, dtype, layout, device, pin_memory))
    
def new_ones(self_: Tensor, size: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenNewOnesOp(self_, size, dtype, layout, device, pin_memory))
    
def new_zeros(self_: Tensor, size: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenNewZerosOp(self_, size, dtype, layout, device, pin_memory))
    
def nll_loss_backward(grad_output: Tensor, self_: Tensor, target: Tensor, weight: Optional[Tensor], reduction: Union[TorchInt, int], ignore_index: Union[TorchInt, int], total_weight: Tensor) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(target), f'`target` should be a Tensor but is {type(target)}'
    if weight is not None:
        assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    assert is_a_torch_tensor(total_weight), f'`total_weight` should be a Tensor but is {type(total_weight)}'
    return Tensor(torch_dialect.AtenNllLossBackwardOp(grad_output, self_, target, weight, reduction, ignore_index, total_weight))
    
def nll_loss_forward(self_: Tensor, target: Tensor, weight: Optional[Tensor], reduction: Union[TorchInt, int], ignore_index: Union[TorchInt, int]) -> Tuple[Tensor, Tensor]:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(target), f'`target` should be a Tensor but is {type(target)}'
    if weight is not None:
        assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    op_results = get_op_results_or_values(torch_dialect.AtenNllLossForwardOp(self_, target, weight, reduction, ignore_index))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
# overload ScalarOpt_dim
def norm(self_: Tensor, p: Optional[TorchNumber], dim: List[Union[TorchInt, int]], keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    return Tensor(torch_dialect.AtenNormScalarOptDimOp(self_, p, dim, keepdim))
    
def numel(self_: Tensor) -> TorchInt:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return TorchInt(torch_dialect.AtenNumelOp(self_).result)
    
def numpy_T(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenNumpyTOp(self_))
    
def ones(size: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenOnesOp(size, dtype, layout, device, pin_memory))
    
def ones_like(self_: Tensor, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenOnesLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    
def pad(self_: Tensor, pad: List[Union[TorchInt, int]], mode: Union[TorchString, str] = "constant", value: Optional[Union[TorchFloat, float]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(pad, (tuple, builtins.list)):
        pad = [pad]
    return Tensor(torch_dialect.AtenPadOp(self_, pad, mode, value))
    
def permute(self_: Tensor, dims: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(dims, (tuple, builtins.list)):
        dims = [dims]
    return Tensor(torch_dialect.AtenPermuteOp(self_, dims))
    
def permute_copy(self_: Tensor, dims: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(dims, (tuple, builtins.list)):
        dims = [dims]
    return Tensor(torch_dialect.AtenPermuteCopyOp(self_, dims))
    
# overload Tensor_Scalar
@dispatch
def pow(self_: Tensor, exponent: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenPowTensorScalarOp(self_, exponent))
    
# overload Tensor_Tensor
@dispatch
def pow(self_: Tensor, exponent: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(exponent), f'`exponent` should be a Tensor but is {type(exponent)}'
    return Tensor(torch_dialect.AtenPowTensorTensorOp(self_, exponent))
    
def prelu(self_: Tensor, weight: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(weight), f'`weight` should be a Tensor but is {type(weight)}'
    return Tensor(torch_dialect.AtenPreluOp(self_, weight))
    
def rand_like(self_: Tensor, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenRandLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    
# overload low
def randint(low: Union[TorchInt, int], high: Union[TorchInt, int], size: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenRandintLowOp(low, high, size, dtype, layout, device, pin_memory))
    
@dispatch
def randn(size: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenRandnOp(size, dtype, layout, device, pin_memory))
    
# overload generator
@dispatch
def randn(size: List[Union[TorchInt, int]], generator: Optional[Generator], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenRandnGeneratorOp(size, generator, dtype, layout, device, pin_memory))
    
def randn_like(self_: Tensor, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenRandnLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    
def reciprocal(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenReciprocalOp(self_))
    
def reciprocal_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenReciprocal_Op(self_))
    
def relu(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenReluOp(self_))
    
def relu6(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRelu6Op(self_))
    
def relu6_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRelu6_Op(self_))
    
def relu_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRelu_Op(self_))
    
# overload int
@dispatch
def remainder(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.AtenRemainderIntOp(a, b).result)
    
# overload Scalar
@dispatch
def remainder(self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRemainderScalarOp(self_, other))
    
def repeat(self_: Tensor, repeats: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(repeats, (tuple, builtins.list)):
        repeats = [repeats]
    return Tensor(torch_dialect.AtenRepeatOp(self_, repeats))
    
def reshape(self_: Tensor, shape: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(shape, (tuple, builtins.list)):
        shape = [shape]
    return Tensor(torch_dialect.AtenReshapeOp(self_, shape))
    
def resize_(self_: Tensor, size: List[Union[TorchInt, int]], memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenResize_Op(self_, size, memory_format))
    
def roll(self_: Tensor, shifts: List[Union[TorchInt, int]], dims: List[Union[TorchInt, int]] = ()) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(shifts, (tuple, builtins.list)):
        shifts = [shifts]
    if not isinstance(dims, (tuple, builtins.list)):
        dims = [dims]
    return Tensor(torch_dialect.AtenRollOp(self_, shifts, dims))
    
def round(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRoundOp(self_))
    
def round_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRound_Op(self_))
    
def rsqrt(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRsqrtOp(self_))
    
def rsqrt_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRsqrt_Op(self_))
    
# overload Scalar
def rsub(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenRsubScalarOp(self_, other, alpha))
    
def scatter_add(self_: Tensor, dim: Union[TorchInt, int], index: Tensor, src: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(index), f'`index` should be a Tensor but is {type(index)}'
    assert is_a_torch_tensor(src), f'`src` should be a Tensor but is {type(src)}'
    return Tensor(torch_dialect.AtenScatterAddOp(self_, dim, index, src))
    
# overload int
def select(self_: Tensor, dim: Union[TorchInt, int], index: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSelectIntOp(self_, dim, index))
    
# overload int
def select_copy(self_: Tensor, dim: Union[TorchInt, int], index: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSelectCopyIntOp(self_, dim, index))
    
def select_scatter(self_: Tensor, src: Tensor, dim: Union[TorchInt, int], index: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(src), f'`src` should be a Tensor but is {type(src)}'
    return Tensor(torch_dialect.AtenSelectScatterOp(self_, src, dim, index))
    
def sigmoid(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSigmoidOp(self_))
    
def sigmoid_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSigmoid_Op(self_))
    
def silu(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSiluOp(self_))
    
def silu_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSilu_Op(self_))
    
def sin(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSinOp(self_))
    
def sin_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSin_Op(self_))
    
@dispatch
def size(self_: Tensor) -> List[TorchInt]:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return List[TorchInt](torch_dialect.AtenSizeOp(self_).result)
    
# overload int
@dispatch
def size(self_: Tensor, dim: Union[TorchInt, int]) -> TorchInt:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return TorchInt(torch_dialect.AtenSizeIntOp(self_, dim).result)
    
# overload Tensor
@dispatch
def slice(self_: Tensor, dim: Union[TorchInt, int] = 0, start: Optional[Union[TorchInt, int]] = None, end: Optional[Union[TorchInt, int]] = None, step: Union[TorchInt, int] = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSliceTensorOp(self_, dim, start, end, step))
    
# overload t
@dispatch
def slice(l: List[Tensor], start: Optional[Union[TorchInt, int]] = None, end: Optional[Union[TorchInt, int]] = None, step: Union[TorchInt, int] = 1) -> List[Tensor]:
    return List[Tensor](torch_dialect.AtenSliceTOp(l, start, end, step).result)
    
# overload Tensor
def slice_copy(self_: Tensor, dim: Union[TorchInt, int] = 0, start: Optional[Union[TorchInt, int]] = None, end: Optional[Union[TorchInt, int]] = None, step: Union[TorchInt, int] = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSliceCopyTensorOp(self_, dim, start, end, step))
    
def slice_scatter(self_: Tensor, src: Tensor, dim: Union[TorchInt, int] = 0, start: Optional[Union[TorchInt, int]] = None, end: Optional[Union[TorchInt, int]] = None, step: Union[TorchInt, int] = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(src), f'`src` should be a Tensor but is {type(src)}'
    return Tensor(torch_dialect.AtenSliceScatterOp(self_, src, dim, start, end, step))
    
# overload int
def softmax(self_: Tensor, dim: Union[TorchInt, int], dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenSoftmaxIntOp(self_, dim, dtype))
    
def softplus(self_: Tensor, beta: TorchNumber = 1, threshold: TorchNumber = 20) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSoftplusOp(self_, beta, threshold))
    
# overload int
def sort(self_: List[Union[TorchInt, int]], reverse: Union[TorchBool, bool] = False) -> None:
    if not isinstance(self_, (tuple, builtins.list)):
        self_ = [self_]
    torch_dialect.AtenSortIntOp(self_, reverse)
    
@dispatch
def sqrt(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSqrtOp(self_))
    
# overload int
@dispatch
def sqrt(a: Union[TorchInt, int]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenSqrtIntOp(a).result)
    
def sqrt_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSqrt_Op(self_))
    
def square(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSquareOp(self_))
    
def square_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSquare_Op(self_))
    
# overload dim
@dispatch
def squeeze(self_: Tensor, dim: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSqueezeDimOp(self_, dim))
    
@dispatch
def squeeze(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSqueezeOp(self_))
    
@dispatch
def squeeze_copy(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSqueezeCopyOp(self_))
    
# overload dim
@dispatch
def squeeze_copy(self_: Tensor, dim: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSqueezeCopyDimOp(self_, dim))
    
def stack(tensors: List[Tensor], dim: Union[TorchInt, int] = 0) -> Tensor:
    assert builtins.all(is_a_torch_tensor(t) or t is None for t in tensors)
    return Tensor(torch_dialect.AtenStackOp(tensors, dim))
    
@dispatch
def std(self_: Tensor, unbiased: Union[TorchBool, bool] = True) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenStdOp(self_, unbiased))
    
# overload dim
@dispatch
def std(self_: Tensor, dim: Optional[List[Union[TorchInt, int]]], unbiased: Union[TorchBool, bool] = True, keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dim is not None and not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    return Tensor(torch_dialect.AtenStdDimOp(self_, dim, unbiased, keepdim))
    
# overload correction
@dispatch
def std(self_: Tensor, dim: Optional[List[Union[TorchInt, int]]] = None, correction: Optional[Union[TorchInt, int]] = None, keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dim is not None and not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    return Tensor(torch_dialect.AtenStdCorrectionOp(self_, dim, correction, keepdim))
    
def str(elem: Tensor) -> TorchString:
    return TorchString(torch_dialect.AtenStrOp(elem).result)
    
# overload Tensor
@dispatch
def sub(self_: Tensor, other: Tensor, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenSubTensorOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def sub(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSubScalarOp(self_, other, alpha))
    
# overload int
@dispatch
def sub(a: Union[TorchInt, int], b: Union[TorchInt, int]) -> TorchInt:
    return TorchInt(torch_dialect.AtenSubIntOp(a, b).result)
    
# overload float
@dispatch
def sub(a: Union[TorchFloat, float], b: Union[TorchFloat, float]) -> TorchFloat:
    return TorchFloat(torch_dialect.AtenSubFloatOp(a, b).result)
    
@dispatch
def sub(a: TorchNumber, b: TorchNumber) -> TorchNumber:
    return TorchNumber(torch_dialect.AtenSubOp(a, b).result)
    
# overload Tensor
@dispatch
def sub_(self_: Tensor, other: Tensor, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenSub_TensorOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def sub_(self_: Tensor, other: TorchNumber, alpha: TorchNumber = 1) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenSub_ScalarOp(self_, other, alpha))
    
@dispatch
def sum(self_: Tensor, dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenSumOp(self_, dtype))
    
# overload dim_IntList
@dispatch
def sum(self_: Tensor, dim: Optional[List[Union[TorchInt, int]]], keepdim: Union[TorchBool, bool] = False, dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dim is not None and not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenSumDimIntListOp(self_, dim, keepdim, dtype))
    
def t(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenTOp(self_))
    
def t_copy(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenTCopyOp(self_))
    
def tanh(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenTanhOp(self_))
    
def tanh_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenTanh_Op(self_))
    
def tanh_backward(grad_output: Tensor, output: Tensor) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(output), f'`output` should be a Tensor but is {type(output)}'
    return Tensor(torch_dialect.AtenTanhBackwardOp(grad_output, output))
    
@dispatch
def tensor(data: List[Tensor], dtype: Optional[pi_dtype] = None, device: Optional[Device] = None, requires_grad: Union[TorchBool, bool] = False) -> Tensor:
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenTensorOp(data, dtype, device, requires_grad))
    
# overload bool
@dispatch
def tensor(t: Union[TorchBool, bool], dtype: Optional[pi_dtype] = None, device: Optional[Device] = None, requires_grad: Union[TorchBool, bool] = False) -> Tensor:
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenTensorBoolOp(t, dtype, device, requires_grad))
    
# overload int
@dispatch
def tensor(t: Union[TorchInt, int], dtype: Optional[pi_dtype] = None, device: Optional[Device] = None, requires_grad: Union[TorchBool, bool] = False) -> Tensor:
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenTensorIntOp(t, dtype, device, requires_grad))
    
# overload float
@dispatch
def tensor(t: Union[TorchFloat, float], dtype: Optional[pi_dtype] = None, device: Optional[Device] = None, requires_grad: Union[TorchBool, bool] = False) -> Tensor:
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenTensorFloatOp(t, dtype, device, requires_grad))
    
def threshold(self_: Tensor, threshold: TorchNumber, value: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenThresholdOp(self_, threshold, value))
    
def threshold_(self_: Tensor, threshold: TorchNumber, value: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenThreshold_Op(self_, threshold, value))
    
def threshold_backward(grad_output: Tensor, self_: Tensor, threshold: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenThresholdBackwardOp(grad_output, self_, threshold))
    
# overload dtype
@dispatch
def to(self_: Tensor, dtype: pi_dtype, non_blocking: Union[TorchBool, bool] = False, copy: Union[TorchBool, bool] = False, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenToDtypeOp(self_, dtype, non_blocking, copy, memory_format))
    
# overload dtype_layout
@dispatch
def to(self_: Tensor, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, non_blocking: Union[TorchBool, bool] = False, copy: Union[TorchBool, bool] = False, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenToDtypeLayoutOp(self_, dtype, layout, device, pin_memory, non_blocking, copy, memory_format))
    
# overload other
@dispatch
def to(self_: Tensor, other: Tensor, non_blocking: Union[TorchBool, bool] = False, copy: Union[TorchBool, bool] = False, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenToOtherOp(self_, other, non_blocking, copy, memory_format))
    
# overload prim_Device
@dispatch
def to(self_: Tensor, device: Optional[Device], dtype: Optional[pi_dtype] = None, non_blocking: Union[TorchBool, bool] = False, copy: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenToPrimDeviceOp(self_, device, dtype, non_blocking, copy))
    
# overload device
@dispatch
def to(self_: Tensor, device: Device, dtype: pi_dtype, non_blocking: Union[TorchBool, bool] = False, copy: Union[TorchBool, bool] = False, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenToDeviceOp(self_, device, dtype, non_blocking, copy, memory_format))
    
def topk(self_: Tensor, k: Union[TorchInt, int], dim: Union[TorchInt, int] = -1, largest: Union[TorchBool, bool] = True, sorted: Union[TorchBool, bool] = True) -> Tuple[Tensor, Tensor]:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    op_results = get_op_results_or_values(torch_dialect.AtenTopkOp(self_, k, dim, largest, sorted))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
# overload int
def transpose(self_: Tensor, dim0: Union[TorchInt, int], dim1: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenTransposeIntOp(self_, dim0, dim1))
    
# overload int
def transpose_copy(self_: Tensor, dim0: Union[TorchInt, int], dim1: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenTransposeCopyIntOp(self_, dim0, dim1))
    
def triu(self_: Tensor, diagonal: Union[TorchInt, int] = 0) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenTriuOp(self_, diagonal))
    
def triu_(self_: Tensor, diagonal: Union[TorchInt, int] = 0) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenTriu_Op(self_, diagonal))
    
def type_as(self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenTypeAsOp(self_, other))
    
def unfold_copy(self_: Tensor, dimension: Union[TorchInt, int], size: Union[TorchInt, int], step: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenUnfoldCopyOp(self_, dimension, size, step))
    
def uniform(self_: Tensor, from_: Union[TorchFloat, float] = 0., to: Union[TorchFloat, float] = 1., generator: Optional[Generator] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenUniformOp(self_, from_, to, generator))
    
def uniform_(self_: Tensor, from_: Union[TorchFloat, float] = 0., to: Union[TorchFloat, float] = 1., generator: Optional[Generator] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenUniform_Op(self_, from_, to, generator))
    
def unsqueeze(self_: Tensor, dim: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenUnsqueezeOp(self_, dim))
    
def unsqueeze_(self_: Tensor, dim: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenUnsqueeze_Op(self_, dim))
    
def unsqueeze_copy(self_: Tensor, dim: Union[TorchInt, int]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenUnsqueezeCopyOp(self_, dim))
    
def upsample_nearest2d(self_: Tensor, output_size: List[Union[TorchInt, int]], scales_h: Optional[Union[TorchFloat, float]] = None, scales_w: Optional[Union[TorchFloat, float]] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(output_size, (tuple, builtins.list)):
        output_size = [output_size]
    return Tensor(torch_dialect.AtenUpsampleNearest2dOp(self_, output_size, scales_h, scales_w))
    
def upsample_nearest2d_backward(grad_output: Tensor, output_size: List[Union[TorchInt, int]], input_size: List[Union[TorchInt, int]], scales_h: Optional[Union[TorchFloat, float]] = None, scales_w: Optional[Union[TorchFloat, float]] = None) -> Tensor:
    assert is_a_torch_tensor(grad_output), f'`grad_output` should be a Tensor but is {type(grad_output)}'
    if not isinstance(output_size, (tuple, builtins.list)):
        output_size = [output_size]
    if not isinstance(input_size, (tuple, builtins.list)):
        input_size = [input_size]
    return Tensor(torch_dialect.AtenUpsampleNearest2dBackwardOp(grad_output, output_size, input_size, scales_h, scales_w))
    
@dispatch
def var(self_: Tensor, unbiased: Union[TorchBool, bool] = True) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenVarOp(self_, unbiased))
    
# overload dim
@dispatch
def var(self_: Tensor, dim: Optional[List[Union[TorchInt, int]]], unbiased: Union[TorchBool, bool] = True, keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dim is not None and not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    return Tensor(torch_dialect.AtenVarDimOp(self_, dim, unbiased, keepdim))
    
# overload correction
@dispatch
def var(self_: Tensor, dim: Optional[List[Union[TorchInt, int]]] = None, correction: Optional[Union[TorchInt, int]] = None, keepdim: Union[TorchBool, bool] = False) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dim is not None and not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    return Tensor(torch_dialect.AtenVarCorrectionOp(self_, dim, correction, keepdim))
    
@dispatch
def var(inp: Tensor, dims: Optional[List[Union[TorchInt, int]]], correction: Union[TorchInt, int], output_dtype: Optional[pi_dtype] = None) -> Tensor:
    assert is_a_torch_tensor(inp), f'`inp` should be a Tensor but is {type(inp)}'
    if dims is not None and not isinstance(dims, (tuple, builtins.list)):
        dims = [dims]
    if output_dtype is not None:
        assert isinstance(output_dtype, pi_dtype), f'expected pi_dtype, got {type(output_dtype)}'
        output_dtype = output_dtype.value
    return Tensor(torch_dialect.PrimsVarOp(inp, dims, correction, output_dtype))
    
# overload correction
@dispatch
def var_mean(self_: Tensor, dim: Optional[List[Union[TorchInt, int]]] = None, correction: Optional[Union[TorchInt, int]] = None, keepdim: Union[TorchBool, bool] = False) -> Tuple[Tensor, Tensor]:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dim is not None and not isinstance(dim, (tuple, builtins.list)):
        dim = [dim]
    op_results = get_op_results_or_values(torch_dialect.AtenVarMeanCorrectionOp(self_, dim, correction, keepdim))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
@dispatch
def var_mean(self_: Tensor, unbiased: Union[TorchBool, bool] = True) -> Tuple[Tensor, Tensor]:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    op_results = get_op_results_or_values(torch_dialect.AtenVarMeanOp(self_, unbiased))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def view(self_: Tensor, size: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    return Tensor(torch_dialect.AtenViewOp(self_, size))
    
@dispatch
def view_copy(self_: Tensor, size: List[Union[TorchInt, int]]) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    return Tensor(torch_dialect.AtenViewCopyOp(self_, size))
    
# overload dtype
@dispatch
def view_copy(self_: Tensor, dtype: pi_dtype) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    return Tensor(torch_dialect.AtenViewCopyDtypeOp(self_, dtype))
    
# overload self
@dispatch
def where(condition: Tensor, self_: Tensor, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(condition), f'`condition` should be a Tensor but is {type(condition)}'
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenWhereSelfOp(condition, self_, other))
    
# overload Scalar
@dispatch
def where(condition: Tensor, self_: TorchNumber, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(condition), f'`condition` should be a Tensor but is {type(condition)}'
    return Tensor(torch_dialect.AtenWhereScalarOp(condition, self_, other))
    
# overload ScalarOther
@dispatch
def where(condition: Tensor, self_: Tensor, other: TorchNumber) -> Tensor:
    assert is_a_torch_tensor(condition), f'`condition` should be a Tensor but is {type(condition)}'
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenWhereScalarOtherOp(condition, self_, other))
    
# overload ScalarSelf
@dispatch
def where(condition: Tensor, self_: TorchNumber, other: Tensor) -> Tensor:
    assert is_a_torch_tensor(condition), f'`condition` should be a Tensor but is {type(condition)}'
    assert is_a_torch_tensor(other), f'`other` should be a Tensor but is {type(other)}'
    return Tensor(torch_dialect.AtenWhereScalarSelfOp(condition, self_, other))
    
def zero(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenZeroOp(self_))
    
def zero_(self_: Tensor) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    return Tensor(torch_dialect.AtenZero_Op(self_))
    
def zeros(size: List[Union[TorchInt, int]], dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None) -> Tensor:
    if not isinstance(size, (tuple, builtins.list)):
        size = [size]
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    return Tensor(torch_dialect.AtenZerosOp(size, dtype, layout, device, pin_memory))
    
def zeros_like(self_: Tensor, dtype: Optional[pi_dtype] = None, layout: Optional[pi_layout] = None, device: Optional[Device] = None, pin_memory: Optional[Union[TorchBool, bool]] = None, memory_format: Optional[pi_memory_format] = None) -> Tensor:
    assert is_a_torch_tensor(self_), f'`self_` should be a Tensor but is {type(self_)}'
    if dtype is not None:
        assert isinstance(dtype, pi_dtype), f'expected pi_dtype, got {type(dtype)}'
        dtype = dtype.value
    if layout is not None:
        assert isinstance(layout, pi_layout), f'expected pi_layout, got {type(layout)}'
        layout = layout.value
    if memory_format is not None:
        assert isinstance(memory_format, pi_memory_format), f'expected pi_memory_format, got {type(memory_format)}'
        memory_format = memory_format.value
    return Tensor(torch_dialect.AtenZerosLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    


__all__ = ['ScalarImplicit', "Bool", "Delete", "Float", "FloatImplicit", "Int", "IntImplicit", "NumToTensor", "RaiseException", "TupleIndex", "Uninitialized", "__and__", "__contains__", "__derive_index", "__getitem__", "__is__", "__isnot__", "__not__", "__range_length", "_convolution", "_embedding_bag", "_index_put_impl", "_index_put_impl_", "_log_softmax", "_log_softmax_backward_data", "_reshape_alias", "_reshape_alias_copy", "_set_item", "_shape_as_tensor", "_softmax", "_softmax_backward_data", "_to_copy", "_unsafe_view", "abs", "abs_", "adaptive_avg_pool2d", "add", "add_", "addcdiv", "addcdiv_", "addcmul", "addcmul_", "addmm", "alias_copy", "all", "amax", "any", "append", "arange", "argmax", "as_strided_copy", "as_strided_scatter", "atan2", "atan2_", "avg_pool2d", "baddbmm", "baddbmm_", "batch_norm", "bernoulli", "bernoulli_", "bincount", "bitwise_and", "bitwise_and_", "bitwise_not", "bitwise_not_", "bitwise_or", "bitwise_or_", "bitwise_xor", "bitwise_xor_", "bmm", "broadcast_to", "bucketize", "cat", "ceil", "ceil_", "clamp", "clamp_", "clamp_max", "clamp_max_", "clamp_min", "clamp_min_", "clone", "constant_pad_nd", "contiguous", "conv2d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "convert_element_type", "convolution", "convolution_backward", "convolution_backward_overrideable", "convolution_overrideable", "copy", "copy_", "cos", "cos_", "cpu", "cumsum", "detach", "detach_copy", "diagonal_copy", "diagonal_scatter", "dim", "div", "div_", "dropout", "dropout_", "embedding", "embedding_bag", "embedding_dense_backward", "empty", "empty_like", "eq", "eq_", "erf", "erf_", "exp", "exp_", "expand", "expand_as", "expand_copy", "expm1", "expm1_", "fft_fft", "fill", "fill_", "flatten", "flip", "floor", "floor_", "floor_divide", "floordiv", "fmod", "fmod_", "frobenius_norm", "full", "full_like", "gather", "ge", "ge_", "gelu", "gelu_backward", "get", "gt", "gt_", "hardsigmoid", "hardsigmoid_", "hardswish", "hardswish_", "hardtanh", "hardtanh_", "index", "index_put", "index_put_", "index_select", "insert", "is_floating_point", "item", "join", "keys", "layer_norm", "layout", "le", "le_", "leaky_relu", "leaky_relu_", "leaky_relu_backward", "len", "lerp", "lerp_", "lift_fresh_copy", "linear", "list", "log", "log1p", "log1p_", "log2", "log2_", "log_", "log_softmax", "logical_and", "logical_and_", "logical_not", "logical_not_", "logical_or", "logical_or_", "logical_xor", "logical_xor_", "logsumexp", "lt", "lt_", "masked_fill", "masked_fill_", "masked_select", "matmul", "max", "max_pool2d", "max_pool2d_with_indices", "max_pool2d_with_indices_backward", "maximum", "mean", "min", "minimum", "mish", "mm", "mse_loss", "mul", "mul_", "mv", "narrow", "native_batch_norm", "native_batch_norm_backward", "native_dropout", "native_dropout_backward", "native_layer_norm", "native_layer_norm_backward", "ne", "ne_", "neg", "neg_", "new_empty", "new_empty_strided", "new_ones", "new_zeros", "nll_loss_backward", "nll_loss_forward", "norm", "numel", "numpy_T", "ones", "ones_like", "pad", "permute", "permute_copy", "pow", "prelu", "rand_like", "randint", "randn", "randn_like", "reciprocal", "reciprocal_", "relu", "relu6", "relu6_", "relu_", "remainder", "repeat", "reshape", "resize_", "roll", "round", "round_", "rsqrt", "rsqrt_", "rsub", "scatter_add", "select", "select_copy", "select_scatter", "sigmoid", "sigmoid_", "silu", "silu_", "sin", "sin_", "size", "slice", "slice_copy", "slice_scatter", "softmax", "softplus", "sort", "sqrt", "sqrt_", "square", "square_", "squeeze", "squeeze_copy", "stack", "std", "str", "sub", "sub_", "sum", "t", "t_copy", "tanh", "tanh_", "tanh_backward", "tensor", "threshold", "threshold_", "threshold_backward", "to", "topk", "transpose", "transpose_copy", "triu", "triu_", "type_as", "unfold_copy", "uniform", "uniform_", "unsqueeze", "unsqueeze_", "unsqueeze_copy", "upsample_nearest2d", "upsample_nearest2d_backward", "var", "var_mean", "vector_norm", "view", "view_copy", "where", "zero", "zero_", "zeros", "zeros_like"]
