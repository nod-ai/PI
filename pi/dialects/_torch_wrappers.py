from plum import dispatch
from enum import Enum
import builtins
from numbers import Number

from .._tensor import Tensor
from ..types_ import is_a_torch_tensor, Device, Generator
from typing import List, Optional, Any, Tuple, Dict

from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.dialects._ods_common import (
    get_default_loc_context,
    get_op_result_or_value,
    get_op_results_or_values,
)

def tanh(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTanhOp(self_))
    
def tanh_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTanh_Op(self_))
    
def hardtanh(self_: Tensor, min_val: Number = -1, max_val: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardtanhOp(self_, min_val, max_val))
    
def hardtanh_(self_: Tensor, min_val: Number = -1, max_val: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardtanh_Op(self_, min_val, max_val))
    
def relu(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenReluOp(self_))
    
def relu_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRelu_Op(self_))
    
def relu6(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRelu6Op(self_))
    
def relu6_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRelu6_Op(self_))
    
def leaky_relu(self_: Tensor, negative_slope: Number = 0.01):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLeakyReluOp(self_, negative_slope))
    
def leaky_relu_(self_: Tensor, negative_slope: Number = 0.01):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLeakyRelu_Op(self_, negative_slope))
    
@dispatch
def log(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLogOp(self_))
    
def log_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog_Op(self_))
    
def sigmoid(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSigmoidOp(self_))
    
def sigmoid_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSigmoid_Op(self_))
    
def hardsigmoid(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardsigmoidOp(self_))
    
def hardsigmoid_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardsigmoid_Op(self_))
    
def hardswish(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardswishOp(self_))
    
def hardswish_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardswish_Op(self_))
    
def erf(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenErfOp(self_))
    
def erf_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenErf_Op(self_))
    
def silu(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSiluOp(self_))
    
def silu_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSilu_Op(self_))
    
def sin(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSinOp(self_))
    
def sin_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSin_Op(self_))
    
def exp(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpOp(self_))
    
def exp_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExp_Op(self_))
    
def expm1(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpm1Op(self_))
    
def expm1_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpm1_Op(self_))
    
def cos(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCosOp(self_))
    
def cos_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCos_Op(self_))
    
def atan2(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenAtan2Op(self_, other))
    
def atan2_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenAtan2_Op(self_, other))
    
@dispatch
def neg(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNegOp(self_))
    
def neg_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNeg_Op(self_))
    
def floor(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFloorOp(self_))
    
def floor_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFloor_Op(self_))
    
@dispatch
def ceil(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCeilOp(self_))
    
def ceil_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCeil_Op(self_))
    
def bitwise_not(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenBitwiseNotOp(self_))
    
def bitwise_not_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenBitwiseNot_Op(self_))
    
# overload Tensor
@dispatch
def div(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenDivTensorOp(self_, other))
    
# overload Tensor
@dispatch
def div_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenDiv_TensorOp(self_, other))
    
def logical_or(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalOrOp(self_, other))
    
def logical_or_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalOr_Op(self_, other))
    
def logical_and(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalAndOp(self_, other))
    
def logical_and_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalAnd_Op(self_, other))
    
def logical_xor(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalXorOp(self_, other))
    
def logical_xor_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalXor_Op(self_, other))
    
def logical_not(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLogicalNotOp(self_))
    
def logical_not_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLogicalNot_Op(self_))
    
# overload Tensor
def lerp(self_: Tensor, end: Tensor, weight: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(end, Tensor), f'`end` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(end).__module__}.{type(end).__name__}'
    end = end.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    return Tensor(torch_dialect.AtenLerpTensorOp(self_, end, weight))
    
# overload Tensor
def lerp_(self_: Tensor, end: Tensor, weight: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(end, Tensor), f'`end` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(end).__module__}.{type(end).__name__}'
    end = end.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    return Tensor(torch_dialect.AtenLerp_TensorOp(self_, end, weight))
    
# overload Tensor
@dispatch
def eq(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenEqTensorOp(self_, other))
    
# overload Tensor
@dispatch
def eq_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenEq_TensorOp(self_, other))
    
# overload Tensor
@dispatch
def gt(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenGtTensorOp(self_, other))
    
# overload Tensor
@dispatch
def gt_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenGt_TensorOp(self_, other))
    
# overload Tensor
@dispatch
def ge(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenGeTensorOp(self_, other))
    
# overload Tensor
@dispatch
def ge_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenGe_TensorOp(self_, other))
    
# overload Tensor
@dispatch
def lt(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLtTensorOp(self_, other))
    
# overload Tensor
@dispatch
def lt_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLt_TensorOp(self_, other))
    
# overload Tensor
@dispatch
def le(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLeTensorOp(self_, other))
    
# overload Tensor
@dispatch
def le_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLe_TensorOp(self_, other))
    
# overload Tensor
@dispatch
def ne(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenNeTensorOp(self_, other))
    
# overload Tensor
@dispatch
def ne_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenNe_TensorOp(self_, other))
    
# overload Scalar
@dispatch
def div(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDivScalarOp(self_, other))
    
# overload Scalar
@dispatch
def div_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDiv_ScalarOp(self_, other))
    
# overload Scalar
@dispatch
def ne(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNeScalarOp(self_, other))
    
# overload Scalar
@dispatch
def ne_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNe_ScalarOp(self_, other))
    
# overload Scalar
@dispatch
def eq(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenEqScalarOp(self_, other))
    
# overload Scalar
@dispatch
def eq_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenEq_ScalarOp(self_, other))
    
# overload Scalar
@dispatch
def gt(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGtScalarOp(self_, other))
    
# overload Scalar
@dispatch
def gt_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGt_ScalarOp(self_, other))
    
# overload Scalar
@dispatch
def ge(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGeScalarOp(self_, other))
    
# overload Scalar
@dispatch
def ge_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGe_ScalarOp(self_, other))
    
# overload Scalar
@dispatch
def lt(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLtScalarOp(self_, other))
    
# overload Scalar
@dispatch
def lt_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLt_ScalarOp(self_, other))
    
# overload Scalar
@dispatch
def le(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLeScalarOp(self_, other))
    
# overload Scalar
@dispatch
def le_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLe_ScalarOp(self_, other))
    
# overload Scalar
def fmod(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFmodScalarOp(self_, other))
    
# overload Scalar
def fmod_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFmod_ScalarOp(self_, other))
    
# overload Scalar
@dispatch
def masked_fill(self_: Tensor, mask: Tensor, value: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    return Tensor(torch_dialect.AtenMaskedFillScalarOp(self_, mask, value))
    
# overload Scalar
@dispatch
def masked_fill_(self_: Tensor, mask: Tensor, value: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    return Tensor(torch_dialect.AtenMaskedFill_ScalarOp(self_, mask, value))
    
# overload Tensor
@dispatch
def masked_fill(self_: Tensor, mask: Tensor, value: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    assert isinstance(value, Tensor), f'`value` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(value).__module__}.{type(value).__name__}'
    value = value.value
    return Tensor(torch_dialect.AtenMaskedFillTensorOp(self_, mask, value))
    
# overload Tensor
@dispatch
def masked_fill_(self_: Tensor, mask: Tensor, value: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    assert isinstance(value, Tensor), f'`value` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(value).__module__}.{type(value).__name__}'
    value = value.value
    return Tensor(torch_dialect.AtenMaskedFill_TensorOp(self_, mask, value))
    
def clamp(self_: Tensor, min: Optional[Number] = None, max: Optional[Number] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampOp(self_, min, max))
    
def clamp_(self_: Tensor, min: Optional[Number] = None, max: Optional[Number] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClamp_Op(self_, min, max))
    
def clamp_min(self_: Tensor, min: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampMinOp(self_, min))
    
def clamp_min_(self_: Tensor, min: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampMin_Op(self_, min))
    
def clamp_max(self_: Tensor, max: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampMaxOp(self_, max))
    
def clamp_max_(self_: Tensor, max: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampMax_Op(self_, max))
    
def log2(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog2Op(self_))
    
def log2_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog2_Op(self_))
    
@dispatch
def sqrt(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqrtOp(self_))
    
def sqrt_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqrt_Op(self_))
    
def log1p(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog1pOp(self_))
    
def log1p_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog1p_Op(self_))
    
def rsqrt(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRsqrtOp(self_))
    
def rsqrt_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRsqrt_Op(self_))
    
@dispatch
def abs(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAbsOp(self_))
    
def abs_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAbs_Op(self_))
    
def reciprocal(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenReciprocalOp(self_))
    
def reciprocal_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenReciprocal_Op(self_))
    
# overload Tensor
def bitwise_and(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenBitwiseAndTensorOp(self_, other))
    
# overload Tensor
def bitwise_and_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenBitwiseAnd_TensorOp(self_, other))
    
# overload Tensor
def bitwise_or(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenBitwiseOrTensorOp(self_, other))
    
# overload Tensor
def bitwise_or_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenBitwiseOr_TensorOp(self_, other))
    
def threshold(self_: Tensor, threshold: Number, value: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenThresholdOp(self_, threshold, value))
    
def threshold_(self_: Tensor, threshold: Number, value: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenThreshold_Op(self_, threshold, value))
    
def square(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSquareOp(self_))
    
def square_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSquare_Op(self_))
    
def unsqueeze(self_: Tensor, dim: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUnsqueezeOp(self_, dim))
    
def unsqueeze_(self_: Tensor, dim: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUnsqueeze_Op(self_, dim))
    
def zero(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenZeroOp(self_))
    
def zero_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenZero_Op(self_))
    
# overload Scalar
@dispatch
def fill(self_: Tensor, value: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFillScalarOp(self_, value))
    
# overload Scalar
@dispatch
def fill_(self_: Tensor, value: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFill_ScalarOp(self_, value))
    
# overload Tensor
@dispatch
def fill(self_: Tensor, value: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(value, Tensor), f'`value` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(value).__module__}.{type(value).__name__}'
    value = value.value
    return Tensor(torch_dialect.AtenFillTensorOp(self_, value))
    
# overload Tensor
@dispatch
def fill_(self_: Tensor, value: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(value, Tensor), f'`value` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(value).__module__}.{type(value).__name__}'
    value = value.value
    return Tensor(torch_dialect.AtenFill_TensorOp(self_, value))
    
# overload Tensor_mode
@dispatch
def div(self_: Tensor, other: Tensor, rounding_mode: Optional[str]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenDivTensorModeOp(self_, other, rounding_mode))
    
# overload Tensor_mode
@dispatch
def div_(self_: Tensor, other: Tensor, rounding_mode: Optional[str]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenDiv_TensorModeOp(self_, other, rounding_mode))
    
# overload Tensor
@dispatch
def mul(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMulTensorOp(self_, other))
    
# overload Tensor
@dispatch
def mul_(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMul_TensorOp(self_, other))
    
# overload Tensor
@dispatch
def add(self_: Tensor, other: Tensor, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenAddTensorOp(self_, other, alpha))
    
# overload Tensor
@dispatch
def add_(self_: Tensor, other: Tensor, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenAdd_TensorOp(self_, other, alpha))
    
# overload Tensor
@dispatch
def sub(self_: Tensor, other: Tensor, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenSubTensorOp(self_, other, alpha))
    
# overload Tensor
@dispatch
def sub_(self_: Tensor, other: Tensor, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenSub_TensorOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def add(self_: Tensor, other: Number, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAddScalarOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def add_(self_: Tensor, other: Number, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAdd_ScalarOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def sub(self_: Tensor, other: Number, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSubScalarOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def sub_(self_: Tensor, other: Number, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSub_ScalarOp(self_, other, alpha))
    
# overload Scalar
@dispatch
def mul(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMulScalarOp(self_, other))
    
# overload Scalar
@dispatch
def mul_(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMul_ScalarOp(self_, other))
    
def addcmul(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(tensor1, Tensor), f'`tensor1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor1).__module__}.{type(tensor1).__name__}'
    tensor1 = tensor1.value
    assert isinstance(tensor2, Tensor), f'`tensor2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor2).__module__}.{type(tensor2).__name__}'
    tensor2 = tensor2.value
    return Tensor(torch_dialect.AtenAddcmulOp(self_, tensor1, tensor2, value))
    
def addcmul_(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(tensor1, Tensor), f'`tensor1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor1).__module__}.{type(tensor1).__name__}'
    tensor1 = tensor1.value
    assert isinstance(tensor2, Tensor), f'`tensor2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor2).__module__}.{type(tensor2).__name__}'
    tensor2 = tensor2.value
    return Tensor(torch_dialect.AtenAddcmul_Op(self_, tensor1, tensor2, value))
    
def addcdiv(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(tensor1, Tensor), f'`tensor1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor1).__module__}.{type(tensor1).__name__}'
    tensor1 = tensor1.value
    assert isinstance(tensor2, Tensor), f'`tensor2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor2).__module__}.{type(tensor2).__name__}'
    tensor2 = tensor2.value
    return Tensor(torch_dialect.AtenAddcdivOp(self_, tensor1, tensor2, value))
    
def addcdiv_(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(tensor1, Tensor), f'`tensor1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor1).__module__}.{type(tensor1).__name__}'
    tensor1 = tensor1.value
    assert isinstance(tensor2, Tensor), f'`tensor2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor2).__module__}.{type(tensor2).__name__}'
    tensor2 = tensor2.value
    return Tensor(torch_dialect.AtenAddcdiv_Op(self_, tensor1, tensor2, value))
    
def maximum(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMaximumOp(self_, other))
    
def minimum(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMinimumOp(self_, other))
    
def mish(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMishOp(self_))
    
# overload Scalar
def rsub(self_: Tensor, other: Number, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRsubScalarOp(self_, other, alpha))
    
def gelu(self_: Tensor, approximate: str = "none"):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGeluOp(self_, approximate))
    
# overload Tensor_Scalar
@dispatch
def pow(self_: Tensor, exponent: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenPowTensorScalarOp(self_, exponent))
    
# overload Tensor_Tensor
@dispatch
def pow(self_: Tensor, exponent: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(exponent, Tensor), f'`exponent` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(exponent).__module__}.{type(exponent).__name__}'
    exponent = exponent.value
    return Tensor(torch_dialect.AtenPowTensorTensorOp(self_, exponent))
    
def threshold_backward(grad_output: Tensor, self_: Tensor, threshold: Number):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenThresholdBackwardOp(grad_output, self_, threshold))
    
@dispatch
def floor_divide(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenFloorDivideOp(self_, other))
    
def softplus(self_: Tensor, beta: Number = 1, threshold: Number = 20):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSoftplusOp(self_, beta, threshold))
    
def prelu(self_: Tensor, weight: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    return Tensor(torch_dialect.AtenPreluOp(self_, weight))
    
def uniform(self_: Tensor, from_: float = 0., to: float = 1., generator: Optional[Generator] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUniformOp(self_, from_, to, generator))
    
def uniform_(self_: Tensor, from_: float = 0., to: float = 1., generator: Optional[Generator] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUniform_Op(self_, from_, to, generator))
    
def rand_like(self_: Tensor, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenRandLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    
@dispatch
def bernoulli(self_: Tensor, generator: Optional[Generator] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenBernoulliOp(self_, generator))
    
# overload float
@dispatch
def bernoulli_(self_: Tensor, p: float = 0.5, generator: Optional[Generator] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenBernoulli_FloatOp(self_, p, generator))
    
# overload low
def randint(low: int, high: int, size: List[int], dtype: Optional[int] = 4, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenRandintLowOp(low, high, size, dtype, layout, device, pin_memory))
    
# overload Tensor
@dispatch
def bernoulli(self_: Tensor, p: Tensor, generator: Optional[Generator] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(p, Tensor), f'`p` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(p).__module__}.{type(p).__name__}'
    p = p.value
    return Tensor(torch_dialect.AtenBernoulliTensorOp(self_, p, generator))
    
# overload Tensor
@dispatch
def bernoulli_(self_: Tensor, p: Tensor, generator: Optional[Generator] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(p, Tensor), f'`p` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(p).__module__}.{type(p).__name__}'
    p = p.value
    return Tensor(torch_dialect.AtenBernoulli_TensorOp(self_, p, generator))
    
def triu(self_: Tensor, diagonal: int = 0):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTriuOp(self_, diagonal))
    
def triu_(self_: Tensor, diagonal: int = 0):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTriu_Op(self_, diagonal))
    
def round(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRoundOp(self_))
    
def round_(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRound_Op(self_))
    
@dispatch
def index_put(self_: Tensor, indices: List[Optional[Tensor]], values: Tensor, accumulate: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) or t is None for t in indices)
    indices = [(t.value if t is not None else None) for t in indices]
    assert isinstance(values, Tensor), f'`values` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(values).__module__}.{type(values).__name__}'
    values = values.value
    return Tensor(torch_dialect.AtenIndexPutOp(self_, indices, values, accumulate))
    
@dispatch
def index_put_(self_: Tensor, indices: List[Optional[Tensor]], values: Tensor, accumulate: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) or t is None for t in indices)
    indices = [(t.value if t is not None else None) for t in indices]
    assert isinstance(values, Tensor), f'`values` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(values).__module__}.{type(values).__name__}'
    values = values.value
    return Tensor(torch_dialect.AtenIndexPut_Op(self_, indices, values, accumulate))
    
# overload hacked_twin
@dispatch
def index_put(self_: Tensor, indices: List[Tensor], values: Tensor, accumulate: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) or t is None for t in indices)
    indices = [(t.value if t is not None else None) for t in indices]
    assert isinstance(values, Tensor), f'`values` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(values).__module__}.{type(values).__name__}'
    values = values.value
    return Tensor(torch_dialect.AtenIndexPutHackedTwinOp(self_, indices, values, accumulate))
    
# overload hacked_twin
@dispatch
def index_put_(self_: Tensor, indices: List[Tensor], values: Tensor, accumulate: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) or t is None for t in indices)
    indices = [(t.value if t is not None else None) for t in indices]
    assert isinstance(values, Tensor), f'`values` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(values).__module__}.{type(values).__name__}'
    values = values.value
    return Tensor(torch_dialect.AtenIndexPut_HackedTwinOp(self_, indices, values, accumulate))
    
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenLinearOp(input, weight, bias))
    
def mm(self_: Tensor, mat2: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mat2, Tensor), f'`mat2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mat2).__module__}.{type(mat2).__name__}'
    mat2 = mat2.value
    return Tensor(torch_dialect.AtenMmOp(self_, mat2))
    
def addmm(self_: Tensor, mat1: Tensor, mat2: Tensor, beta: Number = 1, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mat1, Tensor), f'`mat1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mat1).__module__}.{type(mat1).__name__}'
    mat1 = mat1.value
    assert isinstance(mat2, Tensor), f'`mat2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mat2).__module__}.{type(mat2).__name__}'
    mat2 = mat2.value
    return Tensor(torch_dialect.AtenAddmmOp(self_, mat1, mat2, beta, alpha))
    
def matmul(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMatmulOp(self_, other))
    
def mv(self_: Tensor, vec: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(vec, Tensor), f'`vec` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(vec).__module__}.{type(vec).__name__}'
    vec = vec.value
    return Tensor(torch_dialect.AtenMvOp(self_, vec))
    
def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), groups: int = 1):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConv2dOp(input, weight, bias, stride, padding, dilation, groups))
    
def conv_transpose1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[int] = (1), padding: List[int] = (0), output_padding: List[int] = (0), groups: int = 1, dilation: List[int] = (1)):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvTranspose1dOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
# overload input
def conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), output_padding: List[int] = (0, 0), groups: int = 1, dilation: List[int] = (1, 1)):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvTranspose2dInputOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
# overload input
def conv_transpose3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[int] = (1, 1, 1), padding: List[int] = (0, 0, 0), output_padding: List[int] = (0, 0, 0), groups: int = 1, dilation: List[int] = (1, 1, 1)):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvTranspose3dInputOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
def convolution(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvolutionOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups))
    
def convolution_overrideable(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvolutionOverrideableOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups))
    
@dispatch
def _convolution(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, allow_tf32: bool):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.Aten_ConvolutionOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32))
    
# overload deprecated
@dispatch
def _convolution(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.Aten_ConvolutionDeprecatedOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled))
    
def roll(self_: Tensor, shifts: List[int], dims: List[int] = ()):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRollOp(self_, shifts, dims))
    
def convolution_backward(grad_output: Tensor, input: Tensor, weight: Tensor, bias_sizes: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    op_results = get_op_results_or_values(torch_dialect.AtenConvolutionBackwardOp(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def convolution_backward_overrideable(grad_output: Tensor, input: Tensor, weight: Tensor, stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    op_results = get_op_results_or_values(torch_dialect.AtenConvolutionBackwardOverrideableOp(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def flip(self_: Tensor, dims: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFlipOp(self_, dims))
    
def native_batch_norm(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, momentum: float, eps: float):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    if running_mean is not None:
        assert isinstance(running_mean, Tensor), f'`running_mean` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(running_mean).__module__}.{type(running_mean).__name__}'
        running_mean = running_mean.value
    if running_var is not None:
        assert isinstance(running_var, Tensor), f'`running_var` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(running_var).__module__}.{type(running_var).__name__}'
        running_var = running_var.value
    op_results = get_op_results_or_values(torch_dialect.AtenNativeBatchNormOp(input, weight, bias, running_mean, running_var, training, momentum, eps))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def batch_norm(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, momentum: float, eps: float, cudnn_enabled: bool):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    if running_mean is not None:
        assert isinstance(running_mean, Tensor), f'`running_mean` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(running_mean).__module__}.{type(running_mean).__name__}'
        running_mean = running_mean.value
    if running_var is not None:
        assert isinstance(running_var, Tensor), f'`running_var` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(running_var).__module__}.{type(running_var).__name__}'
        running_var = running_var.value
    return Tensor(torch_dialect.AtenBatchNormOp(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled))
    
def layer_norm(input: Tensor, normalized_shape: List[int], weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1.0000000000000001e-05, cudnn_enable: bool = True):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenLayerNormOp(input, normalized_shape, weight, bias, eps, cudnn_enable))
    
def native_layer_norm(input: Tensor, normalized_shape: List[int], weight: Optional[Tensor], bias: Optional[Tensor], eps: float):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    op_results = get_op_results_or_values(torch_dialect.AtenNativeLayerNormOp(input, normalized_shape, weight, bias, eps))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def max_pool2d(self_: Tensor, kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMaxPool2dOp(self_, kernel_size, stride, padding, dilation, ceil_mode))
    
def max_pool2d_with_indices(self_: Tensor, kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenMaxPool2dWithIndicesOp(self_, kernel_size, stride, padding, dilation, ceil_mode))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def max_pool2d_with_indices_backward(grad_output: Tensor, self_: Tensor, kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, indices: Tensor):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(indices, Tensor), f'`indices` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(indices).__module__}.{type(indices).__name__}'
    indices = indices.value
    return Tensor(torch_dialect.AtenMaxPool2dWithIndicesBackwardOp(grad_output, self_, kernel_size, stride, padding, dilation, ceil_mode, indices))
    
def avg_pool2d(self_: Tensor, kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAvgPool2dOp(self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override))
    
# overload int
def softmax(self_: Tensor, dim: int, dtype: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenSoftmaxIntOp(self_, dim, dtype))
    
# overload int
def log_softmax(self_: Tensor, dim: int, dtype: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenLogSoftmaxIntOp(self_, dim, dtype))
    
def _log_softmax(self_: Tensor, dim: int, half_to_float: bool):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_LogSoftmaxOp(self_, dim, half_to_float))
    
def adaptive_avg_pool2d(self_: Tensor, output_size: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAdaptiveAvgPool2dOp(self_, output_size))
    
def topk(self_: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenTopkOp(self_, k, dim, largest, sorted))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
# overload int
def transpose(self_: Tensor, dim0: int, dim1: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTransposeIntOp(self_, dim0, dim1))
    
def permute(self_: Tensor, dims: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenPermuteOp(self_, dims))
    
def bmm(self_: Tensor, mat2: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mat2, Tensor), f'`mat2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mat2).__module__}.{type(mat2).__name__}'
    mat2 = mat2.value
    return Tensor(torch_dialect.AtenBmmOp(self_, mat2))
    
def cumsum(self_: Tensor, dim: int, dtype: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenCumsumOp(self_, dim, dtype))
    
# overload Scalar
@dispatch
def floor_divide(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFloorDivideScalarOp(self_, other))
    
def logsumexp(self_: Tensor, dim: List[int], keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLogsumexpOp(self_, dim, keepdim))
    
# overload dim
@dispatch
def mean(self_: Tensor, dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenMeanDimOp(self_, dim, keepdim, dtype))
    
# overload Tensor
@dispatch
def __and__(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.Aten__And__TensorOp(self_, other))
    
def _softmax(self_: Tensor, dim: int, half_to_float: bool):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_SoftmaxOp(self_, dim, half_to_float))
    
@dispatch
def mean(self_: Tensor, dtype: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenMeanOp(self_, dtype))
    
@dispatch
def std(self_: Tensor, unbiased: bool = True):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenStdOp(self_, unbiased))
    
# overload dim
@dispatch
def std(self_: Tensor, dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenStdDimOp(self_, dim, unbiased, keepdim))
    
# overload correction
@dispatch
def std(self_: Tensor, dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenStdCorrectionOp(self_, dim, correction, keepdim))
    
@dispatch
def var(self_: Tensor, unbiased: bool = True):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenVarOp(self_, unbiased))
    
# overload dim
@dispatch
def var(self_: Tensor, dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenVarDimOp(self_, dim, unbiased, keepdim))
    
# overload correction
@dispatch
def var(self_: Tensor, dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenVarCorrectionOp(self_, dim, correction, keepdim))
    
# overload correction
@dispatch
def var_mean(self_: Tensor, dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenVarMeanCorrectionOp(self_, dim, correction, keepdim))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
@dispatch
def var_mean(self_: Tensor, unbiased: bool = True):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenVarMeanOp(self_, unbiased))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def nll_loss_forward(self_: Tensor, target: Tensor, weight: Optional[Tensor], reduction: int, ignore_index: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(target, Tensor), f'`target` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(target).__module__}.{type(target).__name__}'
    target = target.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    op_results = get_op_results_or_values(torch_dialect.AtenNllLossForwardOp(self_, target, weight, reduction, ignore_index))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def nll_loss_backward(grad_output: Tensor, self_: Tensor, target: Tensor, weight: Optional[Tensor], reduction: int, ignore_index: int, total_weight: Tensor):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(target, Tensor), f'`target` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(target).__module__}.{type(target).__name__}'
    target = target.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    assert isinstance(total_weight, Tensor), f'`total_weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(total_weight).__module__}.{type(total_weight).__name__}'
    total_weight = total_weight.value
    return Tensor(torch_dialect.AtenNllLossBackwardOp(grad_output, self_, target, weight, reduction, ignore_index, total_weight))
    
def bincount(self_: Tensor, weights: Optional[Tensor] = None, minlength: int = 0):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if weights is not None:
        assert isinstance(weights, Tensor), f'`weights` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weights).__module__}.{type(weights).__name__}'
        weights = weights.value
    return Tensor(torch_dialect.AtenBincountOp(self_, weights, minlength))
    
def linalg_vector_norm(self_: Tensor, ord: Number = 2, dim: Optional[List[int]] = None, keepdim: bool = False, dtype: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenLinalgVectorNormOp(self_, ord, dim, keepdim, dtype))
    
# overload dim
def frobenius_norm(self_: Tensor, dim: List[int], keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFrobeniusNormDimOp(self_, dim, keepdim))
    
def mse_loss(self_: Tensor, target: Tensor, reduction: int = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(target, Tensor), f'`target` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(target).__module__}.{type(target).__name__}'
    target = target.value
    return Tensor(torch_dialect.AtenMseLossOp(self_, target, reduction))
    
def upsample_nearest2d_backward(grad_output: Tensor, output_size: List[int], input_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    return Tensor(torch_dialect.AtenUpsampleNearest2dBackwardOp(grad_output, output_size, input_size, scales_h, scales_w))
    
def constant_pad_nd(self_: Tensor, pad: List[int], value: Number = 0):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenConstantPadNdOp(self_, pad, value))
    
def pad(self_: Tensor, pad: List[int], mode: str = "constant", value: Optional[float] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenPadOp(self_, pad, mode, value))
    
# overload dim
@dispatch
def squeeze(self_: Tensor, dim: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqueezeDimOp(self_, dim))
    
@dispatch
def squeeze(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqueezeOp(self_))
    
# overload using_ints
def flatten(self_: Tensor, start_dim: int = 0, end_dim: int = -1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFlattenUsingIntsOp(self_, start_dim, end_dim))
    
def dim(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return torch_dialect.AtenDimOp(self_).result
    
@dispatch
def size(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return torch_dialect.AtenSizeOp(self_).result
    
# overload Tensor
@dispatch
def Bool(a: Tensor):
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return torch_dialect.AtenBoolTensorOp(a).result
    
def is_floating_point(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return torch_dialect.AtenIsFloatingPointOp(self_).result
    
def new_ones(self_: Tensor, size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenNewOnesOp(self_, size, dtype, layout, device, pin_memory))
    
def new_zeros(self_: Tensor, size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenNewZerosOp(self_, size, dtype, layout, device, pin_memory))
    
@dispatch
def tensor(data: List[Tensor], dtype: Optional[int] = None, device: Optional[Device] = None, requires_grad: bool = False):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenTensorOp(data, dtype, device, requires_grad))
    
# overload bool
@dispatch
def tensor(t: bool, dtype: Optional[int] = None, device: Optional[Device] = None, requires_grad: bool = False):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenTensorBoolOp(t, dtype, device, requires_grad))
    
# overload int
@dispatch
def tensor(t: int, dtype: Optional[int] = None, device: Optional[Device] = None, requires_grad: bool = False):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenTensorIntOp(t, dtype, device, requires_grad))
    
def _shape_as_tensor(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_ShapeAsTensorOp(self_))
    
@dispatch
def all(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAllOp(self_))
    
# overload bool
@dispatch
def all(self_: List[bool]):
    return torch_dialect.AtenAllBoolOp(self_).result
    
@dispatch
def any(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAnyOp(self_))
    
# overload dim
@dispatch
def any(self_: Tensor, dim: int, keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAnyDimOp(self_, dim, keepdim))
    
@dispatch
def arange(end: Number, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenArangeOp(end, dtype, layout, device, pin_memory))
    
# overload start
@dispatch
def arange(start: Number, end: Number, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenArangeStartOp(start, end, dtype, layout, device, pin_memory))
    
# overload start_step
@dispatch
def arange(start: Number, end: Number, step: Number = 1, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenArangeStartStepOp(start, end, step, dtype, layout, device, pin_memory))
    
# overload start_out
@dispatch
def arange(start: Number, end: Number, step: Number = 1, out: Optional[Tensor] = None):
    assert isinstance(out, Tensor), f'`out` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(out).__module__}.{type(out).__name__}'
    out = out.value
    return Tensor(torch_dialect.AtenArangeStartOutOp(start, end, step, out))
    
def argmax(self_: Tensor, dim: Optional[int] = None, keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenArgmaxOp(self_, dim, keepdim))
    
# overload Tensor
def bucketize(self_: Tensor, boundaries: Tensor, out_int32: bool = False, right: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(boundaries, Tensor), f'`boundaries` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(boundaries).__module__}.{type(boundaries).__name__}'
    boundaries = boundaries.value
    return Tensor(torch_dialect.AtenBucketizeTensorOp(self_, boundaries, out_int32, right))
    
def clone(self_: Tensor, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCloneOp(self_, memory_format))
    
def lift_fresh_copy(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLiftFreshCopyOp(self_))
    
def contiguous(self_: Tensor, memory_format: int = 0):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenContiguousOp(self_, memory_format))
    
def copy(self_: Tensor, src: Tensor, non_blocking: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenCopyOp(self_, src, non_blocking))
    
def copy_(self_: Tensor, src: Tensor, non_blocking: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenCopy_Op(self_, src, non_blocking))
    
def _to_copy(self_: Tensor, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.Aten_ToCopyOp(self_, dtype, layout, device, pin_memory, non_blocking, memory_format))
    
def detach(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDetachOp(self_))
    
def embedding(weight: Tensor, indices: Tensor, padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False):
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    assert isinstance(indices, Tensor), f'`indices` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(indices).__module__}.{type(indices).__name__}'
    indices = indices.value
    return Tensor(torch_dialect.AtenEmbeddingOp(weight, indices, padding_idx, scale_grad_by_freq, sparse))
    
# overload padding_idx
def embedding_bag(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: bool, mode: int, sparse: bool, per_sample_weights: Optional[Tensor], include_last_offset: bool, padding_idx: Optional[int]):
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    assert isinstance(indices, Tensor), f'`indices` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(indices).__module__}.{type(indices).__name__}'
    indices = indices.value
    assert isinstance(offsets, Tensor), f'`offsets` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(offsets).__module__}.{type(offsets).__name__}'
    offsets = offsets.value
    if per_sample_weights is not None:
        assert isinstance(per_sample_weights, Tensor), f'`per_sample_weights` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(per_sample_weights).__module__}.{type(per_sample_weights).__name__}'
        per_sample_weights = per_sample_weights.value
    op_results = get_op_results_or_values(torch_dialect.AtenEmbeddingBagPaddingIdxOp(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def _embedding_bag(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: bool = False, mode: int = 0, sparse: bool = False, per_sample_weights: Optional[Tensor] = None, include_last_offset: bool = False, padding_idx: int = -1):
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    assert isinstance(indices, Tensor), f'`indices` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(indices).__module__}.{type(indices).__name__}'
    indices = indices.value
    assert isinstance(offsets, Tensor), f'`offsets` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(offsets).__module__}.{type(offsets).__name__}'
    offsets = offsets.value
    if per_sample_weights is not None:
        assert isinstance(per_sample_weights, Tensor), f'`per_sample_weights` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(per_sample_weights).__module__}.{type(per_sample_weights).__name__}'
        per_sample_weights = per_sample_weights.value
    op_results = get_op_results_or_values(torch_dialect.Aten_EmbeddingBagOp(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def empty_like(self_: Tensor, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenEmptyLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    
def new_empty(self_: Tensor, size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenNewEmptyOp(self_, size, dtype, layout, device, pin_memory))
    
def zeros_like(self_: Tensor, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenZerosLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    
def ones_like(self_: Tensor, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenOnesLikeOp(self_, dtype, layout, device, pin_memory, memory_format))
    
# overload memory_format
def empty(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenEmptyMemoryFormatOp(size, dtype, layout, device, pin_memory, memory_format))
    
def expand(self_: Tensor, size: List[int], implicit: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpandOp(self_, size, implicit))
    
def expand_as(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenExpandAsOp(self_, other))
    
def broadcast_to(self_: Tensor, size: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenBroadcastToOp(self_, size))
    
# overload Tensor
@dispatch
def index(self_: Tensor, indices: List[Optional[Tensor]]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) or t is None for t in indices)
    indices = [(t.value if t is not None else None) for t in indices]
    return Tensor(torch_dialect.AtenIndexTensorOp(self_, indices))
    
# overload Tensor_hacked_twin
@dispatch
def index(self_: Tensor, indices: List[Tensor]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) or t is None for t in indices)
    indices = [(t.value if t is not None else None) for t in indices]
    return Tensor(torch_dialect.AtenIndexTensorHackedTwinOp(self_, indices))
    
def index_select(self_: Tensor, dim: int, index: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(index, Tensor), f'`index` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(index).__module__}.{type(index).__name__}'
    index = index.value
    return Tensor(torch_dialect.AtenIndexSelectOp(self_, dim, index))
    
def _index_put_impl(self_: Tensor, indices: List[Optional[Tensor]], values: Tensor, accumulate: bool = False, unsafe: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) or t is None for t in indices)
    indices = [(t.value if t is not None else None) for t in indices]
    assert isinstance(values, Tensor), f'`values` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(values).__module__}.{type(values).__name__}'
    values = values.value
    return Tensor(torch_dialect.Aten_IndexPutImplOp(self_, indices, values, accumulate, unsafe))
    
def _index_put_impl_(self_: Tensor, indices: List[Optional[Tensor]], values: Tensor, accumulate: bool = False, unsafe: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) or t is None for t in indices)
    indices = [(t.value if t is not None else None) for t in indices]
    assert isinstance(values, Tensor), f'`values` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(values).__module__}.{type(values).__name__}'
    values = values.value
    return Tensor(torch_dialect.Aten_IndexPutImpl_Op(self_, indices, values, accumulate, unsafe))
    
def item(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return torch_dialect.AtenItemOp(self_).result
    
def masked_select(self_: Tensor, mask: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    return Tensor(torch_dialect.AtenMaskedSelectOp(self_, mask))
    
def numel(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return torch_dialect.AtenNumelOp(self_).result
    
def repeat(self_: Tensor, repeats: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRepeatOp(self_, repeats))
    
def reshape(self_: Tensor, shape: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenReshapeOp(self_, shape))
    
def _reshape_alias(self_: Tensor, size: List[int], stride: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_ReshapeAliasOp(self_, size, stride))
    
def resize_(self_: Tensor, size: List[int], memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenResize_Op(self_, size, memory_format))
    
# overload int
def select(self_: Tensor, dim: int, index: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSelectIntOp(self_, dim, index))
    
# overload int
@dispatch
def size(self_: Tensor, dim: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return torch_dialect.AtenSizeIntOp(self_, dim).result
    
def stack(tensors: List[Tensor], dim: int = 0):
    assert builtins.all(isinstance(t, Tensor) or t is None for t in tensors)
    tensors = [(t.value if t is not None else None) for t in tensors]
    return Tensor(torch_dialect.AtenStackOp(tensors, dim))
    
@dispatch
def sum(self_: Tensor, dtype: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenSumOp(self_, dtype))
    
# overload dim_IntList
@dispatch
def sum(self_: Tensor, dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenSumDimIntListOp(self_, dim, keepdim, dtype))
    
@dispatch
def max(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMaxOp(self_))
    
# overload dim
@dispatch
def max(self_: Tensor, dim: int, keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenMaxDimOp(self_, dim, keepdim))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def amax(self_: Tensor, dim: List[int] = (), keepdim: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAmaxOp(self_, dim, keepdim))
    
# overload dtype
@dispatch
def to(self_: Tensor, dtype: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenToDtypeOp(self_, dtype, non_blocking, copy, memory_format))
    
# overload dtype_layout
@dispatch
def to(self_: Tensor, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenToDtypeLayoutOp(self_, dtype, layout, device, pin_memory, non_blocking, copy, memory_format))
    
# overload other
@dispatch
def to(self_: Tensor, other: Tensor, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenToOtherOp(self_, other, non_blocking, copy, memory_format))
    
# overload prim_Device
@dispatch
def to(self_: Tensor, device: Optional[Device], dtype: Optional[int] = None, non_blocking: bool = False, copy: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenToPrimDeviceOp(self_, device, dtype, non_blocking, copy))
    
# overload device
@dispatch
def to(self_: Tensor, device: Device, dtype: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenToDeviceOp(self_, device, dtype, non_blocking, copy, memory_format))
    
def type_as(self_: Tensor, other: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenTypeAsOp(self_, other))
    
def view(self_: Tensor, size: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenViewOp(self_, size))
    
def _unsafe_view(self_: Tensor, size: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_UnsafeViewOp(self_, size))
    
# overload self
@dispatch
def where(condition: Tensor, self_: Tensor, other: Tensor):
    assert isinstance(condition, Tensor), f'`condition` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(condition).__module__}.{type(condition).__name__}'
    condition = condition.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenWhereSelfOp(condition, self_, other))
    
# overload Scalar
@dispatch
def where(condition: Tensor, self_: Number, other: Number):
    assert isinstance(condition, Tensor), f'`condition` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(condition).__module__}.{type(condition).__name__}'
    condition = condition.value
    return Tensor(torch_dialect.AtenWhereScalarOp(condition, self_, other))
    
# overload ScalarOther
@dispatch
def where(condition: Tensor, self_: Tensor, other: Number):
    assert isinstance(condition, Tensor), f'`condition` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(condition).__module__}.{type(condition).__name__}'
    condition = condition.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenWhereScalarOtherOp(condition, self_, other))
    
# overload ScalarSelf
@dispatch
def where(condition: Tensor, self_: Number, other: Tensor):
    assert isinstance(condition, Tensor), f'`condition` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(condition).__module__}.{type(condition).__name__}'
    condition = condition.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenWhereScalarSelfOp(condition, self_, other))
    
# overload Tensor
@dispatch
def slice(self_: Tensor, dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSliceTensorOp(self_, dim, start, end, step))
    
# overload Tensor
@dispatch
def len(t: Tensor):
    assert isinstance(t, Tensor), f'`t` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(t).__module__}.{type(t).__name__}'
    t = t.value
    return torch_dialect.AtenLenTensorOp(t).result
    
def cpu(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCpuOp(self_))
    
def gather(self_: Tensor, dim: int, index: Tensor, sparse_grad: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(index, Tensor), f'`index` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(index).__module__}.{type(index).__name__}'
    index = index.value
    return Tensor(torch_dialect.AtenGatherOp(self_, dim, index, sparse_grad))
    
def scatter_add(self_: Tensor, dim: int, index: Tensor, src: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(index, Tensor), f'`index` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(index).__module__}.{type(index).__name__}'
    index = index.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenScatterAddOp(self_, dim, index, src))
    
def IntImplicit(a: Tensor):
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return torch_dialect.AtenIntImplicitOp(a).result
    
def FloatImplicit(a: Tensor):
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return torch_dialect.AtenFloatImplicitOp(a).result
    
# overload float
@dispatch
def tensor(t: float, dtype: Optional[int] = None, device: Optional[Device] = None, requires_grad: bool = False):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenTensorFloatOp(t, dtype, device, requires_grad))
    
# overload Tensor
@dispatch
def Int(a: Tensor):
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return torch_dialect.AtenIntTensorOp(a).result
    
# overload Tensor
@dispatch
def Float(a: Tensor):
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return torch_dialect.AtenFloatTensorOp(a).result
    
def dropout(input: Tensor, p: float, train: bool):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    return Tensor(torch_dialect.AtenDropoutOp(input, p, train))
    
def dropout_(self_: Tensor, p: float, train: bool):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDropout_Op(self_, p, train))
    
def native_dropout(input: Tensor, p: float, train: Optional[bool]):
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    op_results = get_op_results_or_values(torch_dialect.AtenNativeDropoutOp(input, p, train))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def t(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTOp(self_))
    
def numpy_T(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNumpyTOp(self_))
    
def full(size: List[int], fill_value: Number, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None):
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenFullOp(size, fill_value, dtype, layout, device, pin_memory))
    
def full_like(self_: Tensor, fill_value: Number, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[Device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenFullLikeOp(self_, fill_value, dtype, layout, device, pin_memory, memory_format))
    
def baddbmm(self_: Tensor, batch1: Tensor, batch2: Tensor, beta: Number = 1, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(batch1, Tensor), f'`batch1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(batch1).__module__}.{type(batch1).__name__}'
    batch1 = batch1.value
    assert isinstance(batch2, Tensor), f'`batch2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(batch2).__module__}.{type(batch2).__name__}'
    batch2 = batch2.value
    return Tensor(torch_dialect.AtenBaddbmmOp(self_, batch1, batch2, beta, alpha))
    
def baddbmm_(self_: Tensor, batch1: Tensor, batch2: Tensor, beta: Number = 1, alpha: Number = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(batch1, Tensor), f'`batch1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(batch1).__module__}.{type(batch1).__name__}'
    batch1 = batch1.value
    assert isinstance(batch2, Tensor), f'`batch2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(batch2).__module__}.{type(batch2).__name__}'
    batch2 = batch2.value
    return Tensor(torch_dialect.AtenBaddbmm_Op(self_, batch1, batch2, beta, alpha))
    
def fft_fft(self_: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFftFftOp(self_, n, dim, norm))
    
def alias_copy(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAliasCopyOp(self_))
    
def as_strided_copy(self_: Tensor, size: List[int], stride: List[int], storage_offset: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAsStridedCopyOp(self_, size, stride, storage_offset))
    
def diagonal_copy(self_: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDiagonalCopyOp(self_, offset, dim1, dim2))
    
def expand_copy(self_: Tensor, size: List[int], implicit: bool = False):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpandCopyOp(self_, size, implicit))
    
def permute_copy(self_: Tensor, dims: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenPermuteCopyOp(self_, dims))
    
def _reshape_alias_copy(self_: Tensor, size: List[int], stride: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_ReshapeAliasCopyOp(self_, size, stride))
    
# overload int
def select_copy(self_: Tensor, dim: int, index: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSelectCopyIntOp(self_, dim, index))
    
def detach_copy(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDetachCopyOp(self_))
    
# overload Tensor
def slice_copy(self_: Tensor, dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSliceCopyTensorOp(self_, dim, start, end, step))
    
@dispatch
def squeeze_copy(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqueezeCopyOp(self_))
    
# overload dim
@dispatch
def squeeze_copy(self_: Tensor, dim: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqueezeCopyDimOp(self_, dim))
    
def t_copy(self_: Tensor):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTCopyOp(self_))
    
# overload int
def transpose_copy(self_: Tensor, dim0: int, dim1: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTransposeCopyIntOp(self_, dim0, dim1))
    
def unsqueeze_copy(self_: Tensor, dim: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUnsqueezeCopyOp(self_, dim))
    
@dispatch
def view_copy(self_: Tensor, size: List[int]):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenViewCopyOp(self_, size))
    
# overload dtype
@dispatch
def view_copy(self_: Tensor, dtype: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenViewCopyDtypeOp(self_, dtype))
    
def unfold_copy(self_: Tensor, dimension: int, size: int, step: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUnfoldCopyOp(self_, dimension, size, step))
    
def select_scatter(self_: Tensor, src: Tensor, dim: int, index: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenSelectScatterOp(self_, src, dim, index))
    
def slice_scatter(self_: Tensor, src: Tensor, dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenSliceScatterOp(self_, src, dim, start, end, step))
    
def diagonal_scatter(self_: Tensor, src: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenDiagonalScatterOp(self_, src, offset, dim1, dim2))
    
def as_strided_scatter(self_: Tensor, src: Tensor, size: List[int], stride: List[int], storage_offset: Optional[int] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenAsStridedScatterOp(self_, src, size, stride, storage_offset))
    
def upsample_nearest2d(self_: Tensor, output_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUpsampleNearest2dOp(self_, output_size, scales_h, scales_w))
    
# overload str
@dispatch
def __contains__(dict: Dict[str, Tensor], key: str):
    return torch_dialect.Aten__Contains__StrOp(dict, key).result
    
# overload int_list
@dispatch
def __contains__(l: List[int], item: int):
    return torch_dialect.Aten__Contains__IntListOp(l, item).result
    
# overload Dict_str
@dispatch
def __getitem__(self_: Dict[str, Tensor], key: str):
    return torch_dialect.Aten__Getitem__DictStrOp(self_, key).result
    
# overload str
@dispatch
def _set_item(l: Dict[str, Tensor], idx: str, v: Tensor):
    torch_dialect.Aten_SetItemStrOp(l, idx, v)
    
# overload str
def keys(self_: Dict[str, Tensor]):
    return torch_dialect.AtenKeysStrOp(self_).result
    
# overload default_str
def get(self_: Dict[str, Tensor], key: str, default_value: Tensor):
    return torch_dialect.AtenGetDefaultStrOp(self_, key, default_value).result
    
# overload Dict_str
def Delete(self_: Dict[str, Tensor], key: str):
    torch_dialect.AtenDeleteDictStrOp(self_, key)
    
def cat(tensors: List[Tensor], dim: int = 0):
    assert builtins.all(isinstance(t, Tensor) or t is None for t in tensors)
    tensors = [(t.value if t is not None else None) for t in tensors]
    return Tensor(torch_dialect.AtenCatOp(tensors, dim))
    
# overload t
def append(self_: List[Tensor], el: Tensor):
    return torch_dialect.AtenAppendTOp(self_, el).result
    
# overload t
@dispatch
def add(a: List[Tensor], b: List[Tensor]):
    return torch_dialect.AtenAddTOp(a, b).result
    
# overload int_list
@dispatch
def eq(a: List[int], b: List[int]):
    return torch_dialect.AtenEqIntListOp(a, b).result
    
# overload t
def list(l: List[Tensor]):
    return torch_dialect.AtenListTOp(l).result
    
# overload t
@dispatch
def slice(l: List[Tensor], start: Optional[int] = None, end: Optional[int] = None, step: int = 1):
    return torch_dialect.AtenSliceTOp(l, start, end, step).result
    
# overload t
def insert(self_: List[Tensor], idx: int, el: Tensor):
    torch_dialect.AtenInsertTOp(self_, idx, el)
    
# overload int_list
@dispatch
def ne(a: List[int], b: List[int]):
    return torch_dialect.AtenNeIntListOp(a, b).result
    
# overload bool
@dispatch
def any(self_: List[bool]):
    return torch_dialect.AtenAnyBoolOp(self_).result
    
# overload int
def sort(self_: List[int], reverse: bool = False):
    torch_dialect.AtenSortIntOp(self_, reverse)
    
# overload str
@dispatch
def add(a: str, b: str):
    return torch_dialect.AtenAddStrOp(a, b).result
    
# overload str
@dispatch
def eq(a: str, b: str):
    return torch_dialect.AtenEqStrOp(a, b).result
    
# overload str
@dispatch
def len(s: str):
    return torch_dialect.AtenLenStrOp(s).result
    
def format(self_: str):
    return torch_dialect.AtenFormatOp(self_).result
    
def join(self_: str, values: List[str]):
    return torch_dialect.AtenJoinOp(self_, values).result
    
# overload Scalar
@dispatch
def Float(a: Number):
    return torch_dialect.AtenFloatScalarOp(a).result
    
# overload str
@dispatch
def Float(a: str):
    return torch_dialect.AtenFloatStrOp(a).result
    
# overload float
@dispatch
def Int(a: float):
    return torch_dialect.AtenIntFloatOp(a).result
    
# overload Scalar
@dispatch
def Int(a: Number):
    return torch_dialect.AtenIntScalarOp(a).result
    
def __range_length(lo: int, hi: int, step: int):
    return torch_dialect.Aten__RangeLengthOp(lo, hi, step).result
    
def __derive_index(index: int, start: int, step: int):
    return torch_dialect.Aten__DeriveIndexOp(index, start, step).result
    
# overload int
@dispatch
def gt(a: int, b: int):
    return torch_dialect.AtenGtIntOp(a, b).result
    
# overload int
@dispatch
def ge(a: int, b: int):
    return torch_dialect.AtenGeIntOp(a, b).result
    
# overload int
@dispatch
def lt(a: int, b: int):
    return torch_dialect.AtenLtIntOp(a, b).result
    
# overload int
@dispatch
def le(a: int, b: int):
    return torch_dialect.AtenLeIntOp(a, b).result
    
# overload int
@dispatch
def ne(a: int, b: int):
    return torch_dialect.AtenNeIntOp(a, b).result
    
# overload int
@dispatch
def eq(a: int, b: int):
    return torch_dialect.AtenEqIntOp(a, b).result
    
# overload int
def floordiv(a: int, b: int):
    return torch_dialect.AtenFloordivIntOp(a, b).result
    
# overload int
@dispatch
def remainder(a: int, b: int):
    return torch_dialect.AtenRemainderIntOp(a, b).result
    
# overload Scalar
@dispatch
def remainder(self_: Tensor, other: Number):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRemainderScalarOp(self_, other))
    
# overload int
@dispatch
def add(a: int, b: int):
    return torch_dialect.AtenAddIntOp(a, b).result
    
# overload int
@dispatch
def sub(a: int, b: int):
    return torch_dialect.AtenSubIntOp(a, b).result
    
# overload int
@dispatch
def mul(a: int, b: int):
    return torch_dialect.AtenMulIntOp(a, b).result
    
# overload int
@dispatch
def div(a: int, b: int):
    return torch_dialect.AtenDivIntOp(a, b).result
    
# overload int
@dispatch
def neg(a: int):
    return torch_dialect.AtenNegIntOp(a).result
    
# overload int
@dispatch
def log(a: int):
    return torch_dialect.AtenLogIntOp(a).result
    
# overload float_int
@dispatch
def add(a: float, b: int):
    return torch_dialect.AtenAddFloatIntOp(a, b).result
    
# overload float
@dispatch
def sub(a: float, b: float):
    return torch_dialect.AtenSubFloatOp(a, b).result
    
# overload float
@dispatch
def mul(a: float, b: float):
    return torch_dialect.AtenMulFloatOp(a, b).result
    
# overload float
@dispatch
def div(a: float, b: float):
    return torch_dialect.AtenDivFloatOp(a, b).result
    
# overload float
@dispatch
def neg(a: float):
    return torch_dialect.AtenNegFloatOp(a).result
    
# overload float
@dispatch
def eq(a: float, b: float):
    return torch_dialect.AtenEqFloatOp(a, b).result
    
# overload float
@dispatch
def gt(a: float, b: float):
    return torch_dialect.AtenGtFloatOp(a, b).result
    
# overload float
@dispatch
def ge(a: float, b: float):
    return torch_dialect.AtenGeFloatOp(a, b).result
    
# overload float
@dispatch
def lt(a: float, b: float):
    return torch_dialect.AtenLtFloatOp(a, b).result
    
# overload float_int
@dispatch
def lt(a: float, b: int):
    return torch_dialect.AtenLtFloatIntOp(a, b).result
    
# overload float_int
@dispatch
def ge(a: float, b: int):
    return torch_dialect.AtenGeFloatIntOp(a, b).result
    
# overload float_int
@dispatch
def ne(a: float, b: int):
    return torch_dialect.AtenNeFloatIntOp(a, b).result
    
# overload float_int
@dispatch
def gt(a: float, b: int):
    return torch_dialect.AtenGtFloatIntOp(a, b).result
    
# overload bool
@dispatch
def __and__(a: bool, b: bool):
    return torch_dialect.Aten__And__BoolOp(a, b).result
    
# overload bool
@dispatch
def ne(a: bool, b: bool):
    return torch_dialect.AtenNeBoolOp(a, b).result
    
def __is__(self_: Tensor, obj: Tensor):
    return torch_dialect.Aten__Is__Op(self_, obj).result
    
def __isnot__(self_: Tensor, obj: Tensor):
    return torch_dialect.Aten__Isnot__Op(self_, obj).result
    
def __not__(self_: bool):
    return torch_dialect.Aten__Not__Op(self_).result
    
# overload t
@dispatch
def len(a: List[Tensor]):
    return torch_dialect.AtenLenTOp(a).result
    
# overload t
@dispatch
def __getitem__(list_: List[Tensor], idx: int):
    return torch_dialect.Aten__Getitem__TOp(list_, idx).result
    
# overload t
@dispatch
def _set_item(l: List[Tensor], idx: int, el: Tensor):
    return torch_dialect.Aten_SetItemTOp(l, idx, el).result
    
@dispatch
def div(a: Number, b: Number):
    return torch_dialect.AtenDivOp(a, b).result
    
@dispatch
def add(a: Number, b: Number):
    return torch_dialect.AtenAddOp(a, b).result
    
@dispatch
def sub(a: Number, b: Number):
    return torch_dialect.AtenSubOp(a, b).result
    
# overload Scalar
@dispatch
def ceil(a: Number):
    return torch_dialect.AtenCeilScalarOp(a).result
    
# overload int
@dispatch
def sqrt(a: int):
    return torch_dialect.AtenSqrtIntOp(a).result
    
# overload float
@dispatch
def Bool(a: float):
    return torch_dialect.AtenBoolFloatOp(a).result
    
# overload int
@dispatch
def Bool(a: int):
    return torch_dialect.AtenBoolIntOp(a).result
    
# overload device
@dispatch
def eq(a: Device, b: Device):
    return torch_dialect.AtenEqDeviceOp(a, b).result
    
# overload float
@dispatch
def ceil(a: float):
    return torch_dialect.AtenCeilFloatOp(a).result
    
def narrow(self_: Tensor, dim: int, start: int, length: int):
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNarrowOp(self_, dim, start, length))
    
def ScalarImplicit(a: Tensor):
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return torch_dialect.AtenScalarImplicitOp(a).result
    
def _softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: int):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(output, Tensor), f'`output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(output).__module__}.{type(output).__name__}'
    output = output.value
    return Tensor(torch_dialect.Aten_SoftmaxBackwardDataOp(grad_output, output, dim, input_dtype))
    
def tanh_backward(grad_output: Tensor, output: Tensor):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(output, Tensor), f'`output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(output).__module__}.{type(output).__name__}'
    output = output.value
    return Tensor(torch_dialect.AtenTanhBackwardOp(grad_output, output))
    
def gelu_backward(grad_output: Tensor, self_: Tensor, approximate: str = "none"):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGeluBackwardOp(grad_output, self_, approximate))
    
def _log_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: int):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(output, Tensor), f'`output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(output).__module__}.{type(output).__name__}'
    output = output.value
    return Tensor(torch_dialect.Aten_LogSoftmaxBackwardDataOp(grad_output, output, dim, input_dtype))
    
def native_layer_norm_backward(grad_out: Tensor, input: Tensor, normalized_shape: List[int], mean: Tensor, rstd: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], output_mask: List[bool]):
    assert isinstance(grad_out, Tensor), f'`grad_out` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_out).__module__}.{type(grad_out).__name__}'
    grad_out = grad_out.value
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(mean, Tensor), f'`mean` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mean).__module__}.{type(mean).__name__}'
    mean = mean.value
    assert isinstance(rstd, Tensor), f'`rstd` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(rstd).__module__}.{type(rstd).__name__}'
    rstd = rstd.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    op_results = get_op_results_or_values(torch_dialect.AtenNativeLayerNormBackwardOp(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def embedding_dense_backward(grad_output: Tensor, indices: Tensor, num_weights: int, padding_idx: int, scale_grad_by_freq: bool):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(indices, Tensor), f'`indices` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(indices).__module__}.{type(indices).__name__}'
    indices = indices.value
    return Tensor(torch_dialect.AtenEmbeddingDenseBackwardOp(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq))
    
def native_batch_norm_backward(grad_out: Tensor, input: Tensor, weight: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], save_mean: Optional[Tensor], save_invstd: Optional[Tensor], train: bool, eps: float, output_mask: List[bool]):
    assert isinstance(grad_out, Tensor), f'`grad_out` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_out).__module__}.{type(grad_out).__name__}'
    grad_out = grad_out.value
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    if running_mean is not None:
        assert isinstance(running_mean, Tensor), f'`running_mean` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(running_mean).__module__}.{type(running_mean).__name__}'
        running_mean = running_mean.value
    if running_var is not None:
        assert isinstance(running_var, Tensor), f'`running_var` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(running_var).__module__}.{type(running_var).__name__}'
        running_var = running_var.value
    if save_mean is not None:
        assert isinstance(save_mean, Tensor), f'`save_mean` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(save_mean).__module__}.{type(save_mean).__name__}'
        save_mean = save_mean.value
    if save_invstd is not None:
        assert isinstance(save_invstd, Tensor), f'`save_invstd` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(save_invstd).__module__}.{type(save_invstd).__name__}'
        save_invstd = save_invstd.value
    op_results = get_op_results_or_values(torch_dialect.AtenNativeBatchNormBackwardOp(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    return Tensor(torch_dialect.AtenNativeDropoutBackwardOp(grad_output, mask, scale))
    
def layout(a: Tensor):
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return torch_dialect.PrimLayoutOp(a).result
    
def TupleIndex(tup: Any, i: int):
    return torch_dialect.PrimTupleIndexOp(tup, i).result
    
def TupleUnpack(tup: Any):
    torch_dialect.PrimTupleUnpackOp(tup)
    
# overload Scalar
def NumToTensor(a: Number):
    return Tensor(torch_dialect.PrimNumToTensorScalarOp(a))
    
# overload self_int
@dispatch
def min(self_: List[int]):
    return torch_dialect.PrimMinSelfIntOp(self_).result
    
# overload int
@dispatch
def min(a: int, b: int):
    return torch_dialect.PrimMinIntOp(a, b).result
    
# overload self_int
@dispatch
def max(self_: List[int]):
    return torch_dialect.PrimMaxSelfIntOp(self_).result
    
# overload int
@dispatch
def max(a: int, b: int):
    return torch_dialect.PrimMaxIntOp(a, b).result
    
def RaiseException(msg: str, cls: Optional[str] = None):
    torch_dialect.PrimRaiseExceptionOp(msg, cls)
    
def Uninitialized():
    return torch_dialect.PrimUninitializedOp().result
    
def unchecked_cast(x: Tensor):
    return torch_dialect.PrimUncheckedCastOp(x).result
    
def Print():
    torch_dialect.PrimPrintOp()
    
def tolist():
    torch_dialect.PrimTolistOp()
    
# overload Scalar
@dispatch
def abs(a: Number):
    return torch_dialect.PrimAbsScalarOp(a).result
    
def convert_element_type(a: Tensor, dtype: int):
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.PrimsConvertElementTypeOp(a, dtype))
    


__all__ = ['expand_copy', 'round_', 'mm', 'threshold_backward', 'logsumexp', 'unsqueeze_copy', 'lt_', 'floordiv', '_shape_as_tensor', 'clone', 'full_like', 'minimum', 'ne', 'logical_xor', 'sub', 'squeeze', 'cpu', 'embedding_dense_backward', 'floor', 'relu', 'max', 'tanh_backward', 'alias_copy', 'uniform', '_log_softmax', '_reshape_alias', 'roll', 'leaky_relu_', 'reciprocal', 'adaptive_avg_pool2d', 'atan2', 'index_put_', 'permute', 'layer_norm', 'avg_pool2d', 'nll_loss_backward', 'ge_', 'bitwise_or', '__range_length', 'var_mean', 'tanh', 'index_put', 'matmul', 'addmm', 'hardsigmoid_', 'abs_', 'tensor', 'convolution', 'add_', 'logical_not', '_index_put_impl_', 'convolution_overrideable', 'convert_element_type', 'fill_', 'addcdiv', 'layout', 'mish', 'relu_', 'conv2d', 'slice_scatter', 'triu_', 'sort', '__isnot__', 'rsub', 'stack', 'randint', 'TupleIndex', 'expand', 'logical_and_', '__contains__', '__not__', 'bincount', 'empty', 'new_ones', 'select_copy', 'uniform_', 'as_strided_copy', 'mul_', 't', 'pad', 'log1p', 'sigmoid', '_embedding_bag', 'remainder', 'fmod', 'logical_or_', 'hardtanh', 'log2_', 'ceil_', 'tolist', 'addcdiv_', 'erf', 'clamp_max_', 'ceil', 'ScalarImplicit', 'masked_fill', 'slice', 'softplus', 'fft_fft', 'transpose', 'clamp', 'index', 'squeeze_copy', 'neg_', 'lift_fresh_copy', 'hardswish_', 'bitwise_or_', 'any', 'fmod_', 'append', 'upsample_nearest2d', 'upsample_nearest2d_backward', 'silu_', 'rand_like', 'join', 'unsqueeze_', 'Print', 'index_select', '__getitem__', 'frobenius_norm', 'expand_as', 'exp_', 'native_batch_norm_backward', '_convolution', 'argmax', 'native_dropout', '_to_copy', '__and__', 'linear', 'unchecked_cast', 'triu', 'unfold_copy', 'topk', 'type_as', 'bmm', 'bitwise_and_', 'dropout_', '_reshape_alias_copy', 'diagonal_copy', 'sin_', 'hardswish', 'slice_copy', 'NumToTensor', 'lerp', 'embedding', 'floor_', 'transpose_copy', 'maximum', 'gt', 'min', 'dim', 'broadcast_to', 'gt_', 'narrow', 'logical_xor_', 'masked_select', 'gelu', 'div', 'log', 'masked_fill_', 'as_strided_scatter', 'view', 'sin', 'bitwise_and', 'batch_norm', 'full', 'ne_', 'new_zeros', 'diagonal_scatter', 'sub_', 'zeros_like', 'threshold', 'Bool', 'new_empty', 'arange', 'var', 'bernoulli', 'format', 'numel', 'item', 'relu6_', 'logical_not_', 'gelu_backward', 'rsqrt_', 'sqrt', 'erf_', 'neg', 'relu6', 'baddbmm_', 'insert', 'select_scatter', '_set_item', '_softmax', 'lt', 'square', 'size', 'RaiseException', 'linalg_vector_norm', 'log_', 'cumsum', 'threshold_', 'zero_', 'len', 'nll_loss_forward', 'eq_', 'prelu', 'native_batch_norm', 'sigmoid_', 'bitwise_not', 'keys', 'bitwise_not_', 'sqrt_', 'view_copy', 'log1p_', 'conv_transpose2d', 'empty_like', 'exp', 'constant_pad_nd', 'le', 'native_dropout_backward', '__derive_index', 'list', 'permute_copy', 'max_pool2d_with_indices', 'sum', 'log_softmax', 'add', '_unsafe_view', 'silu', 'get', 'detach_copy', 'tanh_', 'all', 'clamp_min', 'TupleUnpack', '_index_put_impl', 'convolution_backward_overrideable', 'hardsigmoid', 'logical_or', 'leaky_relu', 'select', 'resize_', 'round', '_softmax_backward_data', 'bucketize', 'fill', 'std', 'amax', 'Float', 'dropout', 'convolution_backward', 'reshape', 'Uninitialized', 'where', 'max_pool2d_with_indices_backward', 'gather', 'copy', 'scatter_add', 'numpy_T', 'mse_loss', 'zero', 'bernoulli_', 'contiguous', 'cat', 'max_pool2d', 'native_layer_norm', 'cos', 'repeat', '__is__', 'conv_transpose1d', 'Delete', 'lerp_', 'log2', 'mean', 'native_layer_norm_backward', 'clamp_min_', 'div_', 'Int', 'clamp_max', 'embedding_bag', 'to', 'copy_', 'addcmul_', 'expm1_', 'softmax', 'mul', 'ones_like', 'rsqrt', 'le_', 'unsqueeze', 'reciprocal_', 'flatten', 'square_', 'is_floating_point', 'hardtanh_', 'cos_', 'eq', 'atan2_', 'clamp_', 'abs', 'detach', 'pow', 'conv_transpose3d', 'flip', 'ge', 't_copy', 'logical_and', 'baddbmm', 'FloatImplicit', 'expm1', 'floor_divide', 'addcmul', 'IntImplicit', 'mv', '_log_softmax_backward_data']
