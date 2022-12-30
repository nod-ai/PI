from enum import Enum
import builtins

from .._tensor import Tensor
from ..types_ import Number, is_a_torch_tensor
from typing import List, Optional, Any, Tuple

from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.dialects._ods_common import (
    get_default_loc_context,
    get_op_result_or_value,
    get_op_results_or_values,
)

def tanh(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTanhOp(self_))
    
def tanh_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTanh_Op(self_))
    
def hardtanh(self_: Tensor, min_val: Number = -1, max_val: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardtanhOp(self_, min_val, max_val))
    
def hardtanh_(self_: Tensor, min_val: Number = -1, max_val: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardtanh_Op(self_, min_val, max_val))
    
def relu(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenReluOp(self_))
    
def relu_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRelu_Op(self_))
    
def relu6(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRelu6Op(self_))
    
def relu6_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRelu6_Op(self_))
    
def leaky_relu(self_: Tensor, negative_slope: Number = 0.01) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLeakyReluOp(self_, negative_slope))
    
def leaky_relu_(self_: Tensor, negative_slope: Number = 0.01) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLeakyRelu_Op(self_, negative_slope))
    
def log(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLogOp(self_))
    
def log_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog_Op(self_))
    
def sigmoid(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSigmoidOp(self_))
    
def sigmoid_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSigmoid_Op(self_))
    
def hardsigmoid(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardsigmoidOp(self_))
    
def hardsigmoid_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardsigmoid_Op(self_))
    
def hardswish(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardswishOp(self_))
    
def hardswish_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenHardswish_Op(self_))
    
def erf(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenErfOp(self_))
    
def erf_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenErf_Op(self_))
    
def silu(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSiluOp(self_))
    
def silu_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSilu_Op(self_))
    
def sin(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSinOp(self_))
    
def sin_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSin_Op(self_))
    
def exp(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpOp(self_))
    
def exp_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExp_Op(self_))
    
def expm1(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpm1Op(self_))
    
def expm1_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpm1_Op(self_))
    
def cos(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCosOp(self_))
    
def cos_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCos_Op(self_))
    
def atan2(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenAtan2Op(self_, other))
    
def atan2_(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenAtan2_Op(self_, other))
    
def neg(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNegOp(self_))
    
def neg_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNeg_Op(self_))
    
def floor(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFloorOp(self_))
    
def floor_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFloor_Op(self_))
    
def ceil(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCeilOp(self_))
    
def ceil_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCeil_Op(self_))
    
def bitwise_not(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenBitwiseNotOp(self_))
    
def bitwise_not_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenBitwiseNot_Op(self_))
    
def div_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenDivTensorOp(self_, other))
    
def div__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenDiv_TensorOp(self_, other))
    
def logical_or(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalOrOp(self_, other))
    
def logical_or_(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalOr_Op(self_, other))
    
def logical_and(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalAndOp(self_, other))
    
def logical_and_(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalAnd_Op(self_, other))
    
def logical_xor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalXorOp(self_, other))
    
def logical_xor_(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLogicalXor_Op(self_, other))
    
def logical_not(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLogicalNotOp(self_))
    
def logical_not_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLogicalNot_Op(self_))
    
def lerp_Tensor(self_: Tensor, end: Tensor, weight: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(end, Tensor), f'`end` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(end).__module__}.{type(end).__name__}'
    end = end.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    return Tensor(torch_dialect.AtenLerpTensorOp(self_, end, weight))
    
def lerp__Tensor(self_: Tensor, end: Tensor, weight: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(end, Tensor), f'`end` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(end).__module__}.{type(end).__name__}'
    end = end.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    return Tensor(torch_dialect.AtenLerp_TensorOp(self_, end, weight))
    
def eq_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenEqTensorOp(self_, other))
    
def eq__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenEq_TensorOp(self_, other))
    
def gt_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenGtTensorOp(self_, other))
    
def gt__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenGt_TensorOp(self_, other))
    
def ge_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenGeTensorOp(self_, other))
    
def ge__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenGe_TensorOp(self_, other))
    
def lt_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLtTensorOp(self_, other))
    
def lt__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLt_TensorOp(self_, other))
    
def le_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLeTensorOp(self_, other))
    
def le__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenLe_TensorOp(self_, other))
    
def ne_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenNeTensorOp(self_, other))
    
def ne__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenNe_TensorOp(self_, other))
    
def div_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDivScalarOp(self_, other))
    
def div__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDiv_ScalarOp(self_, other))
    
def ne_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNeScalarOp(self_, other))
    
def ne__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNe_ScalarOp(self_, other))
    
def eq_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenEqScalarOp(self_, other))
    
def eq__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenEq_ScalarOp(self_, other))
    
def gt_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGtScalarOp(self_, other))
    
def gt__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGt_ScalarOp(self_, other))
    
def ge_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGeScalarOp(self_, other))
    
def ge__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGe_ScalarOp(self_, other))
    
def lt_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLtScalarOp(self_, other))
    
def lt__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLt_ScalarOp(self_, other))
    
def le_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLeScalarOp(self_, other))
    
def le__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLe_ScalarOp(self_, other))
    
def fmod_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFmodScalarOp(self_, other))
    
def fmod__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFmod_ScalarOp(self_, other))
    
def masked_fill_Scalar(self_: Tensor, mask: Tensor, value: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    return Tensor(torch_dialect.AtenMaskedFillScalarOp(self_, mask, value))
    
def masked_fill__Scalar(self_: Tensor, mask: Tensor, value: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    return Tensor(torch_dialect.AtenMaskedFill_ScalarOp(self_, mask, value))
    
def masked_fill_Tensor(self_: Tensor, mask: Tensor, value: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    assert isinstance(value, Tensor), f'`value` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(value).__module__}.{type(value).__name__}'
    value = value.value
    return Tensor(torch_dialect.AtenMaskedFillTensorOp(self_, mask, value))
    
def masked_fill__Tensor(self_: Tensor, mask: Tensor, value: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    assert isinstance(value, Tensor), f'`value` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(value).__module__}.{type(value).__name__}'
    value = value.value
    return Tensor(torch_dialect.AtenMaskedFill_TensorOp(self_, mask, value))
    
def clamp(self_: Tensor, min: Optional[Number] = None, max: Optional[Number] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampOp(self_, min, max))
    
def clamp_(self_: Tensor, min: Optional[Number] = None, max: Optional[Number] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClamp_Op(self_, min, max))
    
def clamp_min(self_: Tensor, min: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampMinOp(self_, min))
    
def clamp_min_(self_: Tensor, min: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampMin_Op(self_, min))
    
def clamp_max(self_: Tensor, max: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampMaxOp(self_, max))
    
def clamp_max_(self_: Tensor, max: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenClampMax_Op(self_, max))
    
def log2(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog2Op(self_))
    
def log2_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog2_Op(self_))
    
def sqrt(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqrtOp(self_))
    
def sqrt_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqrt_Op(self_))
    
def log1p(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog1pOp(self_))
    
def log1p_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLog1p_Op(self_))
    
def rsqrt(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRsqrtOp(self_))
    
def rsqrt_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRsqrt_Op(self_))
    
def abs(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAbsOp(self_))
    
def abs_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAbs_Op(self_))
    
def reciprocal(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenReciprocalOp(self_))
    
def reciprocal_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenReciprocal_Op(self_))
    
def bitwise_and_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenBitwiseAndTensorOp(self_, other))
    
def bitwise_and__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenBitwiseAnd_TensorOp(self_, other))
    
def bitwise_or_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenBitwiseOrTensorOp(self_, other))
    
def bitwise_or__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenBitwiseOr_TensorOp(self_, other))
    
def threshold(self_: Tensor, threshold: Number, value: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenThresholdOp(self_, threshold, value))
    
def threshold_(self_: Tensor, threshold: Number, value: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenThreshold_Op(self_, threshold, value))
    
def square(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSquareOp(self_))
    
def square_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSquare_Op(self_))
    
def unsqueeze(self_: Tensor, dim: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUnsqueezeOp(self_, dim))
    
def unsqueeze_(self_: Tensor, dim: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUnsqueeze_Op(self_, dim))
    
def zero(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenZeroOp(self_))
    
def zero_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenZero_Op(self_))
    
def fill_Scalar(self_: Tensor, value: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFillScalarOp(self_, value))
    
def fill__Scalar(self_: Tensor, value: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFill_ScalarOp(self_, value))
    
def fill_Tensor(self_: Tensor, value: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(value, Tensor), f'`value` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(value).__module__}.{type(value).__name__}'
    value = value.value
    return Tensor(torch_dialect.AtenFillTensorOp(self_, value))
    
def fill__Tensor(self_: Tensor, value: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(value, Tensor), f'`value` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(value).__module__}.{type(value).__name__}'
    value = value.value
    return Tensor(torch_dialect.AtenFill_TensorOp(self_, value))
    
def div_Tensor_mode(self_: Tensor, other: Tensor, rounding_mode: Optional[str]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenDivTensorModeOp(self_, other, rounding_mode))
    
def div__Tensor_mode(self_: Tensor, other: Tensor, rounding_mode: Optional[str]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenDiv_TensorModeOp(self_, other, rounding_mode))
    
def mul_Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMulTensorOp(self_, other))
    
def mul__Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMul_TensorOp(self_, other))
    
def add_Tensor(self_: Tensor, other: Tensor, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenAddTensorOp(self_, other, alpha))
    
def add__Tensor(self_: Tensor, other: Tensor, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenAdd_TensorOp(self_, other, alpha))
    
def sub_Tensor(self_: Tensor, other: Tensor, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenSubTensorOp(self_, other, alpha))
    
def sub__Tensor(self_: Tensor, other: Tensor, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenSub_TensorOp(self_, other, alpha))
    
def add_Scalar(self_: Tensor, other: Number, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAddScalarOp(self_, other, alpha))
    
def add__Scalar(self_: Tensor, other: Number, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAdd_ScalarOp(self_, other, alpha))
    
def sub_Scalar(self_: Tensor, other: Number, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSubScalarOp(self_, other, alpha))
    
def sub__Scalar(self_: Tensor, other: Number, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSub_ScalarOp(self_, other, alpha))
    
def mul_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMulScalarOp(self_, other))
    
def mul__Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMul_ScalarOp(self_, other))
    
def addcmul(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(tensor1, Tensor), f'`tensor1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor1).__module__}.{type(tensor1).__name__}'
    tensor1 = tensor1.value
    assert isinstance(tensor2, Tensor), f'`tensor2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor2).__module__}.{type(tensor2).__name__}'
    tensor2 = tensor2.value
    return Tensor(torch_dialect.AtenAddcmulOp(self_, tensor1, tensor2, value))
    
def addcmul_(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(tensor1, Tensor), f'`tensor1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor1).__module__}.{type(tensor1).__name__}'
    tensor1 = tensor1.value
    assert isinstance(tensor2, Tensor), f'`tensor2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor2).__module__}.{type(tensor2).__name__}'
    tensor2 = tensor2.value
    return Tensor(torch_dialect.AtenAddcmul_Op(self_, tensor1, tensor2, value))
    
def addcdiv(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(tensor1, Tensor), f'`tensor1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor1).__module__}.{type(tensor1).__name__}'
    tensor1 = tensor1.value
    assert isinstance(tensor2, Tensor), f'`tensor2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor2).__module__}.{type(tensor2).__name__}'
    tensor2 = tensor2.value
    return Tensor(torch_dialect.AtenAddcdivOp(self_, tensor1, tensor2, value))
    
def addcdiv_(self_: Tensor, tensor1: Tensor, tensor2: Tensor, value: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(tensor1, Tensor), f'`tensor1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor1).__module__}.{type(tensor1).__name__}'
    tensor1 = tensor1.value
    assert isinstance(tensor2, Tensor), f'`tensor2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(tensor2).__module__}.{type(tensor2).__name__}'
    tensor2 = tensor2.value
    return Tensor(torch_dialect.AtenAddcdiv_Op(self_, tensor1, tensor2, value))
    
def maximum(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMaximumOp(self_, other))
    
def minimum(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMinimumOp(self_, other))
    
def mish(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMishOp(self_))
    
def rsub_Scalar(self_: Tensor, other: Number, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRsubScalarOp(self_, other, alpha))
    
def gelu(self_: Tensor, approximate: str = "none") -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGeluOp(self_, approximate))
    
def pow_Tensor_Scalar(self_: Tensor, exponent: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenPowTensorScalarOp(self_, exponent))
    
def pow_Tensor_Tensor(self_: Tensor, exponent: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(exponent, Tensor), f'`exponent` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(exponent).__module__}.{type(exponent).__name__}'
    exponent = exponent.value
    return Tensor(torch_dialect.AtenPowTensorTensorOp(self_, exponent))
    
def threshold_backward(grad_output: Tensor, self_: Tensor, threshold: Number) -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenThresholdBackwardOp(grad_output, self_, threshold))
    
def floor_divide(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenFloorDivideOp(self_, other))
    
def softplus(self_: Tensor, beta: Number = 1, threshold: Number = 20) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSoftplusOp(self_, beta, threshold))
    
def prelu(self_: Tensor, weight: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    return Tensor(torch_dialect.AtenPreluOp(self_, weight))
    
def triu(self_: Tensor, diagonal: int = 0) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTriuOp(self_, diagonal))
    
def triu_(self_: Tensor, diagonal: int = 0) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTriu_Op(self_, diagonal))
    
def round(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRoundOp(self_))
    
def round_(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRound_Op(self_))
    
def index_put_hacked_twin(self_: Tensor, indices: List[Tensor], values: Tensor, accumulate: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) for t in indices)
    indices = [t.value for t in indices]
    assert isinstance(values, Tensor), f'`values` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(values).__module__}.{type(values).__name__}'
    values = values.value
    return Tensor(torch_dialect.AtenIndexPutHackedTwinOp(self_, indices, values, accumulate))
    
def index_put__hacked_twin(self_: Tensor, indices: List[Tensor], values: Tensor, accumulate: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) for t in indices)
    indices = [t.value for t in indices]
    assert isinstance(values, Tensor), f'`values` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(values).__module__}.{type(values).__name__}'
    values = values.value
    return Tensor(torch_dialect.AtenIndexPut_HackedTwinOp(self_, indices, values, accumulate))
    
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenLinearOp(input, weight, bias))
    
def mm(self_: Tensor, mat2: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mat2, Tensor), f'`mat2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mat2).__module__}.{type(mat2).__name__}'
    mat2 = mat2.value
    return Tensor(torch_dialect.AtenMmOp(self_, mat2))
    
def addmm(self_: Tensor, mat1: Tensor, mat2: Tensor, beta: Number = 1, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mat1, Tensor), f'`mat1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mat1).__module__}.{type(mat1).__name__}'
    mat1 = mat1.value
    assert isinstance(mat2, Tensor), f'`mat2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mat2).__module__}.{type(mat2).__name__}'
    mat2 = mat2.value
    return Tensor(torch_dialect.AtenAddmmOp(self_, mat1, mat2, beta, alpha))
    
def matmul(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenMatmulOp(self_, other))
    
def mv(self_: Tensor, vec: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(vec, Tensor), f'`vec` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(vec).__module__}.{type(vec).__name__}'
    vec = vec.value
    return Tensor(torch_dialect.AtenMvOp(self_, vec))
    
def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), groups: int = 1) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConv2dOp(input, weight, bias, stride, padding, dilation, groups))
    
def conv_transpose1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[int] = (1), padding: List[int] = (0), output_padding: List[int] = (0), groups: int = 1, dilation: List[int] = (1)) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvTranspose1dOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
def conv_transpose2d_input(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), output_padding: List[int] = (0, 0), groups: int = 1, dilation: List[int] = (1, 1)) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvTranspose2dInputOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
def conv_transpose3d_input(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: List[int] = (1, 1, 1), padding: List[int] = (0, 0, 0), output_padding: List[int] = (0, 0, 0), groups: int = 1, dilation: List[int] = (1, 1, 1)) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvTranspose3dInputOp(input, weight, bias, stride, padding, output_padding, groups, dilation))
    
def convolution(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvolutionOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups))
    
def convolution_overrideable(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenConvolutionOverrideableOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups))
    
def _convolution(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, allow_tf32: bool) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.Aten_ConvolutionOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32))
    
def _convolution_deprecated(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.Aten_ConvolutionDeprecatedOp(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled))
    
def roll(self_: Tensor, shifts: List[int], dims: List[int] = ()) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRollOp(self_, shifts, dims))
    
def convolution_backward_overrideable(grad_output: Tensor, input: Tensor, weight: Tensor, stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]) -> Tuple[Tensor, Tensor, Tensor]:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    op_results = get_op_results_or_values(torch_dialect.AtenConvolutionBackwardOverrideableOp(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def flip(self_: Tensor, dims: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFlipOp(self_, dims))
    
def native_batch_norm(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, momentum: float, eps: float) -> Tuple[Tensor, Tensor, Tensor]:
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
    
def batch_norm(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, momentum: float, eps: float, cudnn_enabled: bool) -> Tensor:
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
    
def layer_norm(input: Tensor, normalized_shape: List[int], weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1.0000000000000001e-05, cudnn_enable: bool = True) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    if bias is not None:
        assert isinstance(bias, Tensor), f'`bias` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(bias).__module__}.{type(bias).__name__}'
        bias = bias.value
    return Tensor(torch_dialect.AtenLayerNormOp(input, normalized_shape, weight, bias, eps, cudnn_enable))
    
def native_layer_norm(input: Tensor, normalized_shape: List[int], weight: Optional[Tensor], bias: Optional[Tensor], eps: float) -> Tuple[Tensor, Tensor, Tensor]:
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
    
def max_pool2d(self_: Tensor, kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMaxPool2dOp(self_, kernel_size, stride, padding, dilation, ceil_mode))
    
def max_pool2d_with_indices(self_: Tensor, kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False) -> Tuple[Tensor, Tensor]:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenMaxPool2dWithIndicesOp(self_, kernel_size, stride, padding, dilation, ceil_mode))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def max_pool2d_with_indices_backward(grad_output: Tensor, self_: Tensor, kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, indices: Tensor) -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(indices, Tensor), f'`indices` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(indices).__module__}.{type(indices).__name__}'
    indices = indices.value
    return Tensor(torch_dialect.AtenMaxPool2dWithIndicesBackwardOp(grad_output, self_, kernel_size, stride, padding, dilation, ceil_mode, indices))
    
def avg_pool2d(self_: Tensor, kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAvgPool2dOp(self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override))
    
def softmax_int(self_: Tensor, dim: int, dtype: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenSoftmaxIntOp(self_, dim, dtype))
    
def log_softmax_int(self_: Tensor, dim: int, dtype: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenLogSoftmaxIntOp(self_, dim, dtype))
    
def _log_softmax(self_: Tensor, dim: int, half_to_float: bool) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_LogSoftmaxOp(self_, dim, half_to_float))
    
def adaptive_avg_pool2d(self_: Tensor, output_size: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAdaptiveAvgPool2dOp(self_, output_size))
    
def topk(self_: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple[Tensor, Tensor]:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenTopkOp(self_, k, dim, largest, sorted))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def transpose_int(self_: Tensor, dim0: int, dim1: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTransposeIntOp(self_, dim0, dim1))
    
def permute(self_: Tensor, dims: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenPermuteOp(self_, dims))
    
def bmm(self_: Tensor, mat2: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mat2, Tensor), f'`mat2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mat2).__module__}.{type(mat2).__name__}'
    mat2 = mat2.value
    return Tensor(torch_dialect.AtenBmmOp(self_, mat2))
    
def cumsum(self_: Tensor, dim: int, dtype: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenCumsumOp(self_, dim, dtype))
    
def floor_divide_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFloorDivideScalarOp(self_, other))
    
def logsumexp(self_: Tensor, dim: List[int], keepdim: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLogsumexpOp(self_, dim, keepdim))
    
def __and___Tensor(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.Aten__And__TensorOp(self_, other))
    
def _softmax(self_: Tensor, dim: int, half_to_float: bool) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_SoftmaxOp(self_, dim, half_to_float))
    
def mean(self_: Tensor, dtype: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenMeanOp(self_, dtype))
    
def std(self_: Tensor, unbiased: bool = True) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenStdOp(self_, unbiased))
    
def var(self_: Tensor, unbiased: bool = True) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenVarOp(self_, unbiased))
    
def var_mean(self_: Tensor, unbiased: bool = True) -> Tuple[Tensor, Tensor]:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenVarMeanOp(self_, unbiased))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def nll_loss_forward(self_: Tensor, target: Tensor, weight: Optional[Tensor], reduction: int, ignore_index: int) -> Tuple[Tensor, Tensor]:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(target, Tensor), f'`target` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(target).__module__}.{type(target).__name__}'
    target = target.value
    if weight is not None:
        assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
        weight = weight.value
    op_results = get_op_results_or_values(torch_dialect.AtenNllLossForwardOp(self_, target, weight, reduction, ignore_index))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def nll_loss_backward(grad_output: Tensor, self_: Tensor, target: Tensor, weight: Optional[Tensor], reduction: int, ignore_index: int, total_weight: Tensor) -> Tensor:
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
    
def bincount(self_: Tensor, weights: Optional[Tensor] = None, minlength: int = 0) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if weights is not None:
        assert isinstance(weights, Tensor), f'`weights` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weights).__module__}.{type(weights).__name__}'
        weights = weights.value
    return Tensor(torch_dialect.AtenBincountOp(self_, weights, minlength))
    
def frobenius_norm_dim(self_: Tensor, dim: List[int], keepdim: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFrobeniusNormDimOp(self_, dim, keepdim))
    
def mse_loss(self_: Tensor, target: Tensor, reduction: int = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(target, Tensor), f'`target` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(target).__module__}.{type(target).__name__}'
    target = target.value
    return Tensor(torch_dialect.AtenMseLossOp(self_, target, reduction))
    
def upsample_nearest2d_backward(grad_output: Tensor, output_size: List[int], input_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None) -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    return Tensor(torch_dialect.AtenUpsampleNearest2dBackwardOp(grad_output, output_size, input_size, scales_h, scales_w))
    
def constant_pad_nd(self_: Tensor, pad: List[int], value: Number = 0) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenConstantPadNdOp(self_, pad, value))
    
def pad(self_: Tensor, pad: List[int], mode: str = "constant", value: Optional[float] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenPadOp(self_, pad, mode, value))
    
def squeeze_dim(self_: Tensor, dim: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqueezeDimOp(self_, dim))
    
def squeeze(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqueezeOp(self_))
    
def flatten_using_ints(self_: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFlattenUsingIntsOp(self_, start_dim, end_dim))
    
def dim(self_: Tensor) -> int:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDimOp(self_))
    
def size(self_: Tensor) -> List[int]:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSizeOp(self_))
    
def Bool_Tensor(a: Tensor) -> bool:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return Tensor(torch_dialect.AtenBoolTensorOp(a))
    
def is_floating_point(self_: Tensor) -> bool:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenIsFloatingPointOp(self_))
    
def _shape_as_tensor(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_ShapeAsTensorOp(self_))
    
def all(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAllOp(self_))
    
def all_bool(self_: List[bool]) -> bool:
    return Tensor(torch_dialect.AtenAllBoolOp(self_))
    
def any(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAnyOp(self_))
    
def any_dim(self_: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAnyDimOp(self_, dim, keepdim))
    
def arange_start_out(start: Number, end: Number, step: Number = 1, out: Tensor= None) -> Tensor:
    assert isinstance(out, Tensor), f'`out` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(out).__module__}.{type(out).__name__}'
    out = out.value
    return Tensor(torch_dialect.AtenArangeStartOutOp(start, end, step, out))
    
def argmax(self_: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenArgmaxOp(self_, dim, keepdim))
    
def bucketize_Tensor(self_: Tensor, boundaries: Tensor, out_int32: bool = False, right: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(boundaries, Tensor), f'`boundaries` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(boundaries).__module__}.{type(boundaries).__name__}'
    boundaries = boundaries.value
    return Tensor(torch_dialect.AtenBucketizeTensorOp(self_, boundaries, out_int32, right))
    
def clone(self_: Tensor, memory_format: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCloneOp(self_, memory_format))
    
def lift_fresh_copy(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenLiftFreshCopyOp(self_))
    
def contiguous(self_: Tensor, memory_format: int = 0) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenContiguousOp(self_, memory_format))
    
def copy(self_: Tensor, src: Tensor, non_blocking: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenCopyOp(self_, src, non_blocking))
    
def copy_(self_: Tensor, src: Tensor, non_blocking: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenCopy_Op(self_, src, non_blocking))
    
def detach(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDetachOp(self_))
    
def embedding(weight: Tensor, indices: Tensor, padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> Tensor:
    assert isinstance(weight, Tensor), f'`weight` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(weight).__module__}.{type(weight).__name__}'
    weight = weight.value
    assert isinstance(indices, Tensor), f'`indices` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(indices).__module__}.{type(indices).__name__}'
    indices = indices.value
    return Tensor(torch_dialect.AtenEmbeddingOp(weight, indices, padding_idx, scale_grad_by_freq, sparse))
    
def embedding_bag_padding_idx(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: bool, mode: int, sparse: bool, per_sample_weights: Optional[Tensor], include_last_offset: bool, padding_idx: Optional[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    
def _embedding_bag(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: bool = False, mode: int = 0, sparse: bool = False, per_sample_weights: Optional[Tensor] = None, include_last_offset: bool = False, padding_idx: int = -1) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    
def expand(self_: Tensor, size: List[int], implicit: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpandOp(self_, size, implicit))
    
def expand_as(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenExpandAsOp(self_, other))
    
def broadcast_to(self_: Tensor, size: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenBroadcastToOp(self_, size))
    
def index_Tensor_hacked_twin(self_: Tensor, indices: List[Tensor]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert builtins.all(isinstance(t, Tensor) for t in indices)
    indices = [t.value for t in indices]
    return Tensor(torch_dialect.AtenIndexTensorHackedTwinOp(self_, indices))
    
def index_select(self_: Tensor, dim: int, index: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(index, Tensor), f'`index` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(index).__module__}.{type(index).__name__}'
    index = index.value
    return Tensor(torch_dialect.AtenIndexSelectOp(self_, dim, index))
    
def item(self_: Tensor) -> Number:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenItemOp(self_))
    
def masked_select(self_: Tensor, mask: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    return Tensor(torch_dialect.AtenMaskedSelectOp(self_, mask))
    
def numel(self_: Tensor) -> int:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNumelOp(self_))
    
def repeat(self_: Tensor, repeats: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRepeatOp(self_, repeats))
    
def reshape(self_: Tensor, shape: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenReshapeOp(self_, shape))
    
def _reshape_alias(self_: Tensor, size: List[int], stride: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_ReshapeAliasOp(self_, size, stride))
    
def resize_(self_: Tensor, size: List[int], memory_format: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenResize_Op(self_, size, memory_format))
    
def select_int(self_: Tensor, dim: int, index: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSelectIntOp(self_, dim, index))
    
def size_int(self_: Tensor, dim: int) -> int:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSizeIntOp(self_, dim))
    
def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    assert builtins.all(isinstance(t, Tensor) for t in tensors)
    tensors = [t.value for t in tensors]
    return Tensor(torch_dialect.AtenStackOp(tensors, dim))
    
def sum(self_: Tensor, dtype: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenSumOp(self_, dtype))
    
def max(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenMaxOp(self_))
    
def max_dim(self_: Tensor, dim: int, keepdim: bool = False) -> Tuple[Tensor, Tensor]:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    op_results = get_op_results_or_values(torch_dialect.AtenMaxDimOp(self_, dim, keepdim))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def amax(self_: Tensor, dim: List[int] = (), keepdim: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAmaxOp(self_, dim, keepdim))
    
def to_dtype(self_: Tensor, dtype: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenToDtypeOp(self_, dtype, non_blocking, copy, memory_format))
    
def to_other(self_: Tensor, other: Tensor, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenToOtherOp(self_, other, non_blocking, copy, memory_format))
    
def type_as(self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenTypeAsOp(self_, other))
    
def view(self_: Tensor, size: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenViewOp(self_, size))
    
def _unsafe_view(self_: Tensor, size: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_UnsafeViewOp(self_, size))
    
def where_self(condition: Tensor, self_: Tensor, other: Tensor) -> Tensor:
    assert isinstance(condition, Tensor), f'`condition` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(condition).__module__}.{type(condition).__name__}'
    condition = condition.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenWhereSelfOp(condition, self_, other))
    
def where_Scalar(condition: Tensor, self_: Number, other: Number) -> Tensor:
    assert isinstance(condition, Tensor), f'`condition` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(condition).__module__}.{type(condition).__name__}'
    condition = condition.value
    return Tensor(torch_dialect.AtenWhereScalarOp(condition, self_, other))
    
def where_ScalarOther(condition: Tensor, self_: Tensor, other: Number) -> Tensor:
    assert isinstance(condition, Tensor), f'`condition` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(condition).__module__}.{type(condition).__name__}'
    condition = condition.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenWhereScalarOtherOp(condition, self_, other))
    
def where_ScalarSelf(condition: Tensor, self_: Number, other: Tensor) -> Tensor:
    assert isinstance(condition, Tensor), f'`condition` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(condition).__module__}.{type(condition).__name__}'
    condition = condition.value
    assert isinstance(other, Tensor), f'`other` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(other).__module__}.{type(other).__name__}'
    other = other.value
    return Tensor(torch_dialect.AtenWhereScalarSelfOp(condition, self_, other))
    
def slice_Tensor(self_: Tensor, dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSliceTensorOp(self_, dim, start, end, step))
    
def len_Tensor(t: Tensor) -> int:
    assert isinstance(t, Tensor), f'`t` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(t).__module__}.{type(t).__name__}'
    t = t.value
    return Tensor(torch_dialect.AtenLenTensorOp(t))
    
def cpu(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenCpuOp(self_))
    
def gather(self_: Tensor, dim: int, index: Tensor, sparse_grad: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(index, Tensor), f'`index` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(index).__module__}.{type(index).__name__}'
    index = index.value
    return Tensor(torch_dialect.AtenGatherOp(self_, dim, index, sparse_grad))
    
def scatter_add(self_: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(index, Tensor), f'`index` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(index).__module__}.{type(index).__name__}'
    index = index.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenScatterAddOp(self_, dim, index, src))
    
def IntImplicit(a: Tensor) -> int:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return Tensor(torch_dialect.AtenIntImplicitOp(a))
    
def FloatImplicit(a: Tensor) -> float:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return Tensor(torch_dialect.AtenFloatImplicitOp(a))
    
def Int_Tensor(a: Tensor) -> int:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return Tensor(torch_dialect.AtenIntTensorOp(a))
    
def Float_Tensor(a: Tensor) -> float:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return Tensor(torch_dialect.AtenFloatTensorOp(a))
    
def dropout(input: Tensor, p: float, train: bool) -> Tensor:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    return Tensor(torch_dialect.AtenDropoutOp(input, p, train))
    
def dropout_(self_: Tensor, p: float, train: bool) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDropout_Op(self_, p, train))
    
def native_dropout(input: Tensor, p: float, train: Optional[bool]) -> Tuple[Tensor, Tensor]:
    assert isinstance(input, Tensor), f'`input` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(input).__module__}.{type(input).__name__}'
    input = input.value
    op_results = get_op_results_or_values(torch_dialect.AtenNativeDropoutOp(input, p, train))
    return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])
    
def t(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTOp(self_))
    
def numpy_T(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNumpyTOp(self_))
    
def baddbmm(self_: Tensor, batch1: Tensor, batch2: Tensor, beta: Number = 1, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(batch1, Tensor), f'`batch1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(batch1).__module__}.{type(batch1).__name__}'
    batch1 = batch1.value
    assert isinstance(batch2, Tensor), f'`batch2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(batch2).__module__}.{type(batch2).__name__}'
    batch2 = batch2.value
    return Tensor(torch_dialect.AtenBaddbmmOp(self_, batch1, batch2, beta, alpha))
    
def baddbmm_(self_: Tensor, batch1: Tensor, batch2: Tensor, beta: Number = 1, alpha: Number = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(batch1, Tensor), f'`batch1` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(batch1).__module__}.{type(batch1).__name__}'
    batch1 = batch1.value
    assert isinstance(batch2, Tensor), f'`batch2` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(batch2).__module__}.{type(batch2).__name__}'
    batch2 = batch2.value
    return Tensor(torch_dialect.AtenBaddbmm_Op(self_, batch1, batch2, beta, alpha))
    
def fft_fft(self_: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenFftFftOp(self_, n, dim, norm))
    
def alias_copy(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAliasCopyOp(self_))
    
def as_strided_copy(self_: Tensor, size: List[int], stride: List[int], storage_offset: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenAsStridedCopyOp(self_, size, stride, storage_offset))
    
def diagonal_copy(self_: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDiagonalCopyOp(self_, offset, dim1, dim2))
    
def expand_copy(self_: Tensor, size: List[int], implicit: bool = False) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenExpandCopyOp(self_, size, implicit))
    
def permute_copy(self_: Tensor, dims: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenPermuteCopyOp(self_, dims))
    
def _reshape_alias_copy(self_: Tensor, size: List[int], stride: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.Aten_ReshapeAliasCopyOp(self_, size, stride))
    
def select_copy_int(self_: Tensor, dim: int, index: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSelectCopyIntOp(self_, dim, index))
    
def detach_copy(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenDetachCopyOp(self_))
    
def slice_copy_Tensor(self_: Tensor, dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSliceCopyTensorOp(self_, dim, start, end, step))
    
def squeeze_copy(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqueezeCopyOp(self_))
    
def squeeze_copy_dim(self_: Tensor, dim: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenSqueezeCopyDimOp(self_, dim))
    
def t_copy(self_: Tensor) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTCopyOp(self_))
    
def transpose_copy_int(self_: Tensor, dim0: int, dim1: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenTransposeCopyIntOp(self_, dim0, dim1))
    
def unsqueeze_copy(self_: Tensor, dim: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUnsqueezeCopyOp(self_, dim))
    
def view_copy(self_: Tensor, size: List[int]) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenViewCopyOp(self_, size))
    
def view_copy_dtype(self_: Tensor, dtype: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.AtenViewCopyDtypeOp(self_, dtype))
    
def unfold_copy(self_: Tensor, dimension: int, size: int, step: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUnfoldCopyOp(self_, dimension, size, step))
    
def select_scatter(self_: Tensor, src: Tensor, dim: int, index: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenSelectScatterOp(self_, src, dim, index))
    
def slice_scatter(self_: Tensor, src: Tensor, dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenSliceScatterOp(self_, src, dim, start, end, step))
    
def diagonal_scatter(self_: Tensor, src: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenDiagonalScatterOp(self_, src, offset, dim1, dim2))
    
def as_strided_scatter(self_: Tensor, src: Tensor, size: List[int], stride: List[int], storage_offset: Optional[int] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    assert isinstance(src, Tensor), f'`src` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(src).__module__}.{type(src).__name__}'
    src = src.value
    return Tensor(torch_dialect.AtenAsStridedScatterOp(self_, src, size, stride, storage_offset))
    
def upsample_nearest2d(self_: Tensor, output_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenUpsampleNearest2dOp(self_, output_size, scales_h, scales_w))
    
def __contains___int_list(l: List[int], item: int) -> bool:
    return Tensor(torch_dialect.Aten__Contains__IntListOp(l, item))
    
def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    assert builtins.all(isinstance(t, Tensor) for t in tensors)
    tensors = [t.value for t in tensors]
    return Tensor(torch_dialect.AtenCatOp(tensors, dim))
    
def append_t(self_: List[Tensor], el: Tensor) -> List[Tensor]:
    return Tensor(torch_dialect.AtenAppendTOp(self_, el))
    
def add_t(a: List[Tensor], b: List[Tensor]) -> List[Tensor]:
    return Tensor(torch_dialect.AtenAddTOp(a, b))
    
def eq_int_list(a: List[int], b: List[int]) -> bool:
    return Tensor(torch_dialect.AtenEqIntListOp(a, b))
    
def list_t(l: List[Tensor]) -> List[Tensor]:
    return Tensor(torch_dialect.AtenListTOp(l))
    
def slice_t(l: List[Tensor], start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> List[Tensor]:
    return Tensor(torch_dialect.AtenSliceTOp(l, start, end, step))
    
def insert_t(self_: List[Tensor], idx: int, el: Tensor) -> None:
    torch_dialect.AtenInsertTOp(self_, idx, el)
    
def ne_int_list(a: List[int], b: List[int]) -> bool:
    return Tensor(torch_dialect.AtenNeIntListOp(a, b))
    
def any_bool(self_: List[bool]) -> bool:
    return Tensor(torch_dialect.AtenAnyBoolOp(self_))
    
def sort_int(self_: List[int], reverse: bool = False) -> None:
    torch_dialect.AtenSortIntOp(self_, reverse)
    
def add_str(a: str, b: str) -> str:
    return Tensor(torch_dialect.AtenAddStrOp(a, b))
    
def eq_str(a: str, b: str) -> bool:
    return Tensor(torch_dialect.AtenEqStrOp(a, b))
    
def len_str(s: str) -> int:
    return Tensor(torch_dialect.AtenLenStrOp(s))
    
def str(elem: Tensor) -> str:
    return Tensor(torch_dialect.AtenStrOp(elem))
    
def join(self_: str, values: List[str]) -> str:
    return Tensor(torch_dialect.AtenJoinOp(self_, values))
    
def Float_Scalar(a: Number) -> float:
    return Tensor(torch_dialect.AtenFloatScalarOp(a))
    
def Float_str(a: str) -> float:
    return Tensor(torch_dialect.AtenFloatStrOp(a))
    
def Int_float(a: float) -> int:
    return Tensor(torch_dialect.AtenIntFloatOp(a))
    
def Int_Scalar(a: Number) -> int:
    return Tensor(torch_dialect.AtenIntScalarOp(a))
    
def __range_length(lo: int, hi: int, step: int) -> int:
    return Tensor(torch_dialect.Aten__RangeLengthOp(lo, hi, step))
    
def __derive_index(index: int, start: int, step: int) -> int:
    return Tensor(torch_dialect.Aten__DeriveIndexOp(index, start, step))
    
def gt_int(a: int, b: int) -> bool:
    return Tensor(torch_dialect.AtenGtIntOp(a, b))
    
def ge_int(a: int, b: int) -> bool:
    return Tensor(torch_dialect.AtenGeIntOp(a, b))
    
def lt_int(a: int, b: int) -> bool:
    return Tensor(torch_dialect.AtenLtIntOp(a, b))
    
def le_int(a: int, b: int) -> bool:
    return Tensor(torch_dialect.AtenLeIntOp(a, b))
    
def ne_int(a: int, b: int) -> bool:
    return Tensor(torch_dialect.AtenNeIntOp(a, b))
    
def eq_int(a: int, b: int) -> bool:
    return Tensor(torch_dialect.AtenEqIntOp(a, b))
    
def floordiv_int(a: int, b: int) -> int:
    return Tensor(torch_dialect.AtenFloordivIntOp(a, b))
    
def remainder_int(a: int, b: int) -> int:
    return Tensor(torch_dialect.AtenRemainderIntOp(a, b))
    
def remainder_Scalar(self_: Tensor, other: Number) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenRemainderScalarOp(self_, other))
    
def add_int(a: int, b: int) -> int:
    return Tensor(torch_dialect.AtenAddIntOp(a, b))
    
def sub_int(a: int, b: int) -> int:
    return Tensor(torch_dialect.AtenSubIntOp(a, b))
    
def mul_int(a: int, b: int) -> int:
    return Tensor(torch_dialect.AtenMulIntOp(a, b))
    
def div_int(a: int, b: int) -> float:
    return Tensor(torch_dialect.AtenDivIntOp(a, b))
    
def neg_int(a: int) -> int:
    return Tensor(torch_dialect.AtenNegIntOp(a))
    
def log_int(a: int) -> float:
    return Tensor(torch_dialect.AtenLogIntOp(a))
    
def add_float_int(a: float, b: int) -> float:
    return Tensor(torch_dialect.AtenAddFloatIntOp(a, b))
    
def sub_float(a: float, b: float) -> float:
    return Tensor(torch_dialect.AtenSubFloatOp(a, b))
    
def mul_float(a: float, b: float) -> float:
    return Tensor(torch_dialect.AtenMulFloatOp(a, b))
    
def div_float(a: float, b: float) -> float:
    return Tensor(torch_dialect.AtenDivFloatOp(a, b))
    
def neg_float(a: float) -> float:
    return Tensor(torch_dialect.AtenNegFloatOp(a))
    
def eq_float(a: float, b: float) -> bool:
    return Tensor(torch_dialect.AtenEqFloatOp(a, b))
    
def gt_float(a: float, b: float) -> bool:
    return Tensor(torch_dialect.AtenGtFloatOp(a, b))
    
def ge_float(a: float, b: float) -> bool:
    return Tensor(torch_dialect.AtenGeFloatOp(a, b))
    
def lt_float(a: float, b: float) -> bool:
    return Tensor(torch_dialect.AtenLtFloatOp(a, b))
    
def lt_float_int(a: float, b: int) -> bool:
    return Tensor(torch_dialect.AtenLtFloatIntOp(a, b))
    
def ge_float_int(a: float, b: int) -> bool:
    return Tensor(torch_dialect.AtenGeFloatIntOp(a, b))
    
def ne_float_int(a: float, b: int) -> bool:
    return Tensor(torch_dialect.AtenNeFloatIntOp(a, b))
    
def gt_float_int(a: float, b: int) -> bool:
    return Tensor(torch_dialect.AtenGtFloatIntOp(a, b))
    
def __and___bool(a: bool, b: bool) -> bool:
    return Tensor(torch_dialect.Aten__And__BoolOp(a, b))
    
def ne_bool(a: bool, b: bool) -> bool:
    return Tensor(torch_dialect.AtenNeBoolOp(a, b))
    
def __is__(self_: Tensor, obj: Tensor) -> bool:
    return Tensor(torch_dialect.Aten__Is__Op(self_, obj))
    
def __isnot__(self_: Tensor, obj: Tensor) -> bool:
    return Tensor(torch_dialect.Aten__Isnot__Op(self_, obj))
    
def __not__(self_: bool) -> bool:
    return Tensor(torch_dialect.Aten__Not__Op(self_))
    
def len_t(a: List[Tensor]) -> int:
    return Tensor(torch_dialect.AtenLenTOp(a))
    
def __getitem___t(list_: List[Tensor], idx: int) -> Tensor:
    return Tensor(torch_dialect.Aten__Getitem__TOp(list_, idx))
    
def _set_item_t(l: List[Tensor], idx: int, el: Tensor) -> List[Tensor]:
    return Tensor(torch_dialect.Aten_SetItemTOp(l, idx, el))
    
def div(a: Number, b: Number) -> float:
    return Tensor(torch_dialect.AtenDivOp(a, b))
    
def add(a: Number, b: Number) -> Number:
    return Tensor(torch_dialect.AtenAddOp(a, b))
    
def sub(a: Number, b: Number) -> Number:
    return Tensor(torch_dialect.AtenSubOp(a, b))
    
def ceil_Scalar(a: Number) -> Number:
    return Tensor(torch_dialect.AtenCeilScalarOp(a))
    
def sqrt_int(a: int) -> float:
    return Tensor(torch_dialect.AtenSqrtIntOp(a))
    
def Bool_float(a: float) -> bool:
    return Tensor(torch_dialect.AtenBoolFloatOp(a))
    
def Bool_int(a: int) -> bool:
    return Tensor(torch_dialect.AtenBoolIntOp(a))
    
def ceil_float(a: float) -> int:
    return Tensor(torch_dialect.AtenCeilFloatOp(a))
    
def narrow(self_: Tensor, dim: int, start: int, length: int) -> Tensor:
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenNarrowOp(self_, dim, start, length))
    
def ScalarImplicit(a: Tensor) -> Number:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return Tensor(torch_dialect.AtenScalarImplicitOp(a))
    
def _softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: int) -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(output, Tensor), f'`output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(output).__module__}.{type(output).__name__}'
    output = output.value
    return Tensor(torch_dialect.Aten_SoftmaxBackwardDataOp(grad_output, output, dim, input_dtype))
    
def tanh_backward(grad_output: Tensor, output: Tensor) -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(output, Tensor), f'`output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(output).__module__}.{type(output).__name__}'
    output = output.value
    return Tensor(torch_dialect.AtenTanhBackwardOp(grad_output, output))
    
def gelu_backward(grad_output: Tensor, self_: Tensor, approximate: str = "none") -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(self_, Tensor), f'`self_` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(self_).__module__}.{type(self_).__name__}'
    self_ = self_.value
    return Tensor(torch_dialect.AtenGeluBackwardOp(grad_output, self_, approximate))
    
def _log_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: int) -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(output, Tensor), f'`output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(output).__module__}.{type(output).__name__}'
    output = output.value
    return Tensor(torch_dialect.Aten_LogSoftmaxBackwardDataOp(grad_output, output, dim, input_dtype))
    
def native_layer_norm_backward(grad_out: Tensor, input: Tensor, normalized_shape: List[int], mean: Tensor, rstd: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], output_mask: List[bool]) -> Tuple[Tensor, Tensor, Tensor]:
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
    
def embedding_dense_backward(grad_output: Tensor, indices: Tensor, num_weights: int, padding_idx: int, scale_grad_by_freq: bool) -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(indices, Tensor), f'`indices` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(indices).__module__}.{type(indices).__name__}'
    indices = indices.value
    return Tensor(torch_dialect.AtenEmbeddingDenseBackwardOp(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq))
    
def native_batch_norm_backward(grad_out: Tensor, input: Tensor, weight: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], save_mean: Optional[Tensor], save_invstd: Optional[Tensor], train: bool, eps: float, output_mask: List[bool]) -> Tuple[Tensor, Tensor, Tensor]:
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
    
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float) -> Tensor:
    assert isinstance(grad_output, Tensor), f'`grad_output` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(grad_output).__module__}.{type(grad_output).__name__}'
    grad_output = grad_output.value
    assert isinstance(mask, Tensor), f'`mask` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(mask).__module__}.{type(mask).__name__}'
    mask = mask.value
    return Tensor(torch_dialect.AtenNativeDropoutBackwardOp(grad_output, mask, scale))
    
def prim_layout(a: Tensor) -> int:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return Tensor(torch_dialect.PrimLayoutOp(a))
    
def prim_TupleIndex(tup: Any, i: int) -> Any:
    return Tensor(torch_dialect.PrimTupleIndexOp(tup, i))
    
def prim_dtype(a: Tensor) -> int:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    return Tensor(torch_dialect.PrimDtypeOp(a))
    
def prim_NumToTensor_Scalar(a: Number) -> Tensor:
    return Tensor(torch_dialect.PrimNumToTensorScalarOp(a))
    
def prim_min_self_int(self_: List[int]) -> int:
    return Tensor(torch_dialect.PrimMinSelfIntOp(self_))
    
def prim_min_int(a: int, b: int) -> int:
    return Tensor(torch_dialect.PrimMinIntOp(a, b))
    
def prim_max_self_int(self_: List[int]) -> int:
    return Tensor(torch_dialect.PrimMaxSelfIntOp(self_))
    
def prim_max_int(a: int, b: int) -> int:
    return Tensor(torch_dialect.PrimMaxIntOp(a, b))
    
def prim_RaiseException(msg: str, cls: Optional[str] = None) -> None:
    torch_dialect.PrimRaiseExceptionOp(msg, cls)
    
def prim_Uninitialized() -> Any:
    return Tensor(torch_dialect.PrimUninitializedOp())
    
def prim_unchecked_cast(x: Tensor) -> Tensor:
    return Tensor(torch_dialect.PrimUncheckedCastOp(x))
    
def prim_abs_Scalar(a: Number) -> Number:
    return Tensor(torch_dialect.PrimAbsScalarOp(a))
    
def prims_convert_element_type(a: Tensor, dtype: int) -> Tensor:
    assert isinstance(a, Tensor), f'`a` should be a {Tensor.__module__}.{Tensor.__name__} but is {type(a).__module__}.{type(a).__name__}'
    a = a.value
    if dtype is not None and isinstance(dtype, Enum):
        dtype = dtype.value
    return Tensor(torch_dialect.PrimsConvertElementTypeOp(a, dtype))
    


__all__ = ['tanh', 'tanh_', 'hardtanh', 'hardtanh_', 'relu', 'relu_', 'relu6', 'relu6_', 'leaky_relu', 'leaky_relu_', 'log', 'log_', 'sigmoid', 'sigmoid_', 'hardsigmoid', 'hardsigmoid_', 'hardswish', 'hardswish_', 'erf', 'erf_', 'silu', 'silu_', 'sin', 'sin_', 'exp', 'exp_', 'expm1', 'expm1_', 'cos', 'cos_', 'atan2', 'atan2_', 'neg', 'neg_', 'floor', 'floor_', 'ceil', 'ceil_', 'bitwise_not', 'bitwise_not_', 'div_Tensor', 'div__Tensor', 'logical_or', 'logical_or_', 'logical_and', 'logical_and_', 'logical_xor', 'logical_xor_', 'logical_not', 'logical_not_', 'lerp_Tensor', 'lerp__Tensor', 'eq_Tensor', 'eq__Tensor', 'gt_Tensor', 'gt__Tensor', 'ge_Tensor', 'ge__Tensor', 'lt_Tensor', 'lt__Tensor', 'le_Tensor', 'le__Tensor', 'ne_Tensor', 'ne__Tensor', 'div_Scalar', 'div__Scalar', 'ne_Scalar', 'ne__Scalar', 'eq_Scalar', 'eq__Scalar', 'gt_Scalar', 'gt__Scalar', 'ge_Scalar', 'ge__Scalar', 'lt_Scalar', 'lt__Scalar', 'le_Scalar', 'le__Scalar', 'fmod_Scalar', 'fmod__Scalar', 'masked_fill_Scalar', 'masked_fill__Scalar', 'masked_fill_Tensor', 'masked_fill__Tensor', 'clamp', 'clamp_', 'clamp_min', 'clamp_min_', 'clamp_max', 'clamp_max_', 'log2', 'log2_', 'sqrt', 'sqrt_', 'log1p', 'log1p_', 'rsqrt', 'rsqrt_', 'abs', 'abs_', 'reciprocal', 'reciprocal_', 'bitwise_and_Tensor', 'bitwise_and__Tensor', 'bitwise_or_Tensor', 'bitwise_or__Tensor', 'threshold', 'threshold_', 'square', 'square_', 'unsqueeze', 'unsqueeze_', 'zero', 'zero_', 'fill_Scalar', 'fill__Scalar', 'fill_Tensor', 'fill__Tensor', 'div_Tensor_mode', 'div__Tensor_mode', 'mul_Tensor', 'mul__Tensor', 'add_Tensor', 'add__Tensor', 'sub_Tensor', 'sub__Tensor', 'add_Scalar', 'add__Scalar', 'sub_Scalar', 'sub__Scalar', 'mul_Scalar', 'mul__Scalar', 'addcmul', 'addcmul_', 'addcdiv', 'addcdiv_', 'maximum', 'minimum', 'mish', 'rsub_Scalar', 'gelu', 'pow_Tensor_Scalar', 'pow_Tensor_Tensor', 'threshold_backward', 'floor_divide', 'softplus', 'prelu', 'triu', 'triu_', 'round', 'round_', 'index_put_hacked_twin', 'index_put__hacked_twin', 'linear', 'mm', 'addmm', 'matmul', 'mv', 'conv2d', 'conv_transpose1d', 'conv_transpose2d_input', 'conv_transpose3d_input', 'convolution', 'convolution_overrideable', '_convolution', '_convolution_deprecated', 'roll', 'convolution_backward_overrideable', 'flip', 'native_batch_norm', 'batch_norm', 'layer_norm', 'native_layer_norm', 'max_pool2d', 'max_pool2d_with_indices', 'max_pool2d_with_indices_backward', 'avg_pool2d', 'softmax_int', 'log_softmax_int', '_log_softmax', 'adaptive_avg_pool2d', 'topk', 'transpose_int', 'permute', 'bmm', 'cumsum', 'floor_divide_Scalar', 'logsumexp', '__and___Tensor', '_softmax', 'mean', 'std', 'var', 'var_mean', 'nll_loss_forward', 'nll_loss_backward', 'bincount', 'frobenius_norm_dim', 'mse_loss', 'upsample_nearest2d_backward', 'constant_pad_nd', 'pad', 'squeeze_dim', 'squeeze', 'flatten_using_ints', 'dim', 'size', 'Bool_Tensor', 'is_floating_point', '_shape_as_tensor', 'all', 'all_bool', 'any', 'any_dim', 'arange_start_out', 'argmax', 'bucketize_Tensor', 'clone', 'lift_fresh_copy', 'contiguous', 'copy', 'copy_', 'detach', 'embedding', 'embedding_bag_padding_idx', '_embedding_bag', 'expand', 'expand_as', 'broadcast_to', 'index_Tensor_hacked_twin', 'index_select', 'item', 'masked_select', 'numel', 'repeat', 'reshape', '_reshape_alias', 'resize_', 'select_int', 'size_int', 'stack', 'sum', 'max', 'max_dim', 'amax', 'to_dtype', 'to_other', 'type_as', 'view', '_unsafe_view', 'where_self', 'where_Scalar', 'where_ScalarOther', 'where_ScalarSelf', 'slice_Tensor', 'len_Tensor', 'cpu', 'gather', 'scatter_add', 'IntImplicit', 'FloatImplicit', 'Int_Tensor', 'Float_Tensor', 'dropout', 'dropout_', 'native_dropout', 't', 'numpy_T', 'baddbmm', 'baddbmm_', 'fft_fft', 'alias_copy', 'as_strided_copy', 'diagonal_copy', 'expand_copy', 'permute_copy', '_reshape_alias_copy', 'select_copy_int', 'detach_copy', 'slice_copy_Tensor', 'squeeze_copy', 'squeeze_copy_dim', 't_copy', 'transpose_copy_int', 'unsqueeze_copy', 'view_copy', 'view_copy_dtype', 'unfold_copy', 'select_scatter', 'slice_scatter', 'diagonal_scatter', 'as_strided_scatter', 'upsample_nearest2d', '__contains___int_list', 'cat', 'append_t', 'add_t', 'eq_int_list', 'list_t', 'slice_t', 'insert_t', 'ne_int_list', 'any_bool', 'sort_int', 'add_str', 'eq_str', 'len_str', 'str', 'join', 'Float_Scalar', 'Float_str', 'Int_float', 'Int_Scalar', '__range_length', '__derive_index', 'gt_int', 'ge_int', 'lt_int', 'le_int', 'ne_int', 'eq_int', 'floordiv_int', 'remainder_int', 'remainder_Scalar', 'add_int', 'sub_int', 'mul_int', 'div_int', 'neg_int', 'log_int', 'add_float_int', 'sub_float', 'mul_float', 'div_float', 'neg_float', 'eq_float', 'gt_float', 'ge_float', 'lt_float', 'lt_float_int', 'ge_float_int', 'ne_float_int', 'gt_float_int', '__and___bool', 'ne_bool', '__is__', '__isnot__', '__not__', 'len_t', '__getitem___t', '_set_item_t', 'div', 'add', 'sub', 'ceil_Scalar', 'sqrt_int', 'Bool_float', 'Bool_int', 'ceil_float', 'narrow', 'ScalarImplicit', '_softmax_backward_data', 'tanh_backward', 'gelu_backward', '_log_softmax_backward_data', 'native_layer_norm_backward', 'embedding_dense_backward', 'native_batch_norm_backward', 'native_dropout_backward', 'prim_layout', 'prim_TupleIndex', 'prim_dtype', 'prim_NumToTensor_Scalar', 'prim_min_self_int', 'prim_min_int', 'prim_max_self_int', 'prim_max_int', 'prim_RaiseException', 'prim_Uninitialized', 'prim_unchecked_cast', 'prim_abs_Scalar', 'prims_convert_element_type']
