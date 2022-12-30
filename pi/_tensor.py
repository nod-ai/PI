from __future__ import annotations
import warnings
from typing import Tuple, Optional, Any

# noinspection PyUnresolvedReferences
import numpy as np
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.dialects._ods_common import get_op_result_or_value
from torch_mlir.ir import (
    DenseElementsAttr,
)
from torch_mlir.ir import (
    Value as MLIRValue,
)

import pi
from .types_ import dtype as shark_dtype


class TorchTensorWrapper(type):
    # def __new__(mcs, name, bases, class_dict):
    #     for k, f in class_dict.items():
    #         if k in {"__init__", "__hash__", "_version", "value", "__class__", "type"}:
    #             continue
    #         if inspect.isfunction(f) and not isinstance(f, property):
    #             def run_on_actual_value(*args, **kwargs):
    #                 self = args[0]
    #                 return f((self.value, *args[1:]), **kwargs)
    #
    #             class_dict[k] = run_on_actual_value
    #     return type.__new__(mcs, name, bases, class_dict)

    def __subclasscheck__(cls, subclass):
        print(cls, subclass)
        return False

    @classmethod
    def __instancecheck__(cls, instance):
        try:
            return instance.is_shark_tensor
        except:
            return False


class Tensor(metaclass=TorchTensorWrapper):
    @property
    def is_shark_tensor(self):
        return True

    @property
    def __class__(self):
        return MLIRValue

    @property
    def type(self):
        return self._value.type

    @property
    def value(self):
        return self._value

    def __init__(self, tensor: MLIRValue):
        self._value = get_op_result_or_value(tensor)

    def abs(self):
        raise NotImplementedError

    def absolute(self):
        raise NotImplementedError

    def absolute_(self):
        raise NotImplementedError

    def abs_(self):
        raise NotImplementedError

    def acos(self):
        raise NotImplementedError

    def acosh(self):
        raise NotImplementedError

    def acosh_(self):
        raise NotImplementedError

    def acos_(self):
        raise NotImplementedError

    def add(self, other, *args, **kwargs):
        raise NotImplementedError

    def addbmm(self, batch1, batch2, *args, **kwargs):
        raise NotImplementedError

    def addbmm_(self, batch1, batch2, *args, **kwargs):
        raise NotImplementedError

    def addcdiv(self, tensor1, tensor2, *args, **kwargs):
        raise NotImplementedError

    def addcdiv_(self, tensor1, tensor2, *args, **kwargs):
        raise NotImplementedError

    def addcmul(self, tensor1, tensor2, *args, **kwargs):
        raise NotImplementedError

    def addcmul_(self, tensor1, tensor2, *args, **kwargs):
        raise NotImplementedError

    def addmm(self, mat1, mat2, *args, **kwargs):
        raise NotImplementedError

    def addmm_(self, mat1, mat2, *args, **kwargs):
        raise NotImplementedError

    def addmv(self, mat, vec, *args, **kwargs):
        raise NotImplementedError

    def addmv_(self, mat, vec, *args, **kwargs):
        raise NotImplementedError

    def addr(self, vec1, vec2, *args, **kwargs):
        raise NotImplementedError

    def addr_(self, vec1, vec2, *args, **kwargs):
        raise NotImplementedError

    def add_(self, other, *args, **kwargs):
        raise NotImplementedError

    def adjoint(self):
        raise NotImplementedError

    def align_as(self, other):
        raise NotImplementedError

    def align_to(self, *args, **kwargs):
        raise NotImplementedError

    def all(self, dim=None, keepdim=False):
        raise NotImplementedError

    def allclose(self, other, rtol=1, *args, **kwargs):
        raise NotImplementedError

    def amax(self, dim=None, keepdim=False):
        raise NotImplementedError

    def amin(self, dim=None, keepdim=False):
        raise NotImplementedError

    def aminmax(self, *args, **kwargs):
        raise NotImplementedError

    def angle(self):
        raise NotImplementedError

    def any(self, dim=None, keepdim=False):
        raise NotImplementedError

    def apply_(self, callable):
        raise NotImplementedError

    def arccos(self):
        raise NotImplementedError

    def arccosh(self, *args, **kwargs):
        raise NotImplementedError

    def arccosh_(self, *args, **kwargs):
        raise NotImplementedError

    def arccos_(self):
        raise NotImplementedError

    def arcsin(self):
        raise NotImplementedError

    def arcsinh(self):
        raise NotImplementedError

    def arcsinh_(self):
        raise NotImplementedError

    def arcsin_(self):
        raise NotImplementedError

    def arctan(self):
        raise NotImplementedError

    def arctan2(self, other):
        raise NotImplementedError

    def arctan2_(self, *args, **kwargs):
        raise NotImplementedError

    def arctanh(self):
        raise NotImplementedError

    def arctanh_(self, other):
        raise NotImplementedError

    def arctan_(self):
        raise NotImplementedError

    def argmax(self, dim=None, keepdim=False):
        raise NotImplementedError

    def argmin(self, dim=None, keepdim=False):
        raise NotImplementedError

    def argsort(self, dim=-1, descending=False):
        raise NotImplementedError

    def argwhere(self):
        raise NotImplementedError

    def asin(self):
        raise NotImplementedError

    def asinh(self):
        raise NotImplementedError

    def asinh_(self):
        raise NotImplementedError

    def asin_(self):
        raise NotImplementedError

    def as_strided(self, size, stride, storage_offset=None):
        raise NotImplementedError

    def as_strided_(self, *args, **kwargs):
        raise NotImplementedError

    def as_strided_scatter(self, src, size, stride, storage_offset=None):
        raise NotImplementedError

    def as_subclass(self, cls):
        raise NotImplementedError

    def atan(self):
        raise NotImplementedError

    def atan2(self, other):
        raise NotImplementedError

    def atan2_(self, other):
        raise NotImplementedError

    def atanh(self):
        raise NotImplementedError

    def atanh_(self, other):
        raise NotImplementedError

    def atan_(self):
        raise NotImplementedError

    def baddbmm(self, batch1, batch2, *args, **kwargs):
        raise NotImplementedError

    def baddbmm_(self, batch1, batch2, *args, **kwargs):
        raise NotImplementedError

    def bernoulli(self, *args, **kwargs):
        raise NotImplementedError

    def bernoulli_(self, p=0.5, *args, **kwargs):
        raise NotImplementedError

    def bfloat16(self, memory_format=None):
        raise NotImplementedError

    def bincount(self, weights=None, minlength=0):
        raise NotImplementedError

    def bitwise_and(self):
        raise NotImplementedError

    def bitwise_and_(self):
        raise NotImplementedError

    def bitwise_left_shift(self, other):
        raise NotImplementedError

    def bitwise_left_shift_(self, other):
        raise NotImplementedError

    def bitwise_not(self):
        raise NotImplementedError

    def bitwise_not_(self):
        raise NotImplementedError

    def bitwise_or(self):
        raise NotImplementedError

    def bitwise_or_(self):
        raise NotImplementedError

    def bitwise_right_shift(self, other):
        raise NotImplementedError

    def bitwise_right_shift_(self, other):
        raise NotImplementedError

    def bitwise_xor(self):
        raise NotImplementedError

    def bitwise_xor_(self):
        raise NotImplementedError

    def bmm(self, batch2):
        raise NotImplementedError

    def bool(self, memory_format=None):
        raise NotImplementedError

    def broadcast_to(self, shape):
        raise NotImplementedError

    def byte(self, memory_format=None):
        raise NotImplementedError

    def cauchy_(self, median=0, sigma=1, *args, **kwargs):
        raise NotImplementedError

    def ccol_indices(self, *args, **kwargs):
        raise NotImplementedError

    def cdouble(self, memory_format=None):
        raise NotImplementedError

    def ceil(self):
        raise NotImplementedError

    def ceil_(self):
        raise NotImplementedError

    def cfloat(self, memory_format=None):
        raise NotImplementedError

    def chalf(self, memory_format=None):
        raise NotImplementedError

    def char(self, memory_format=None):
        raise NotImplementedError

    def cholesky(self, upper=False):
        raise NotImplementedError

    def cholesky_inverse(self, upper=False):
        raise NotImplementedError

    def cholesky_solve(self, input2, upper=False):
        raise NotImplementedError

    def chunk(self, chunks, dim=0):
        raise NotImplementedError

    def clamp(self, min=None, max=None):
        raise NotImplementedError

    def clamp_(self, min=None, max=None):
        raise NotImplementedError

    def clamp_max(self, *args, **kwargs):
        raise NotImplementedError

    def clamp_max_(self, *args, **kwargs):
        raise NotImplementedError

    def clamp_min(self, *args, **kwargs):
        raise NotImplementedError

    def clamp_min_(self, *args, **kwargs):
        raise NotImplementedError

    def clip(self, min=None, max=None):
        raise NotImplementedError

    def clip_(self, min=None, max=None):
        raise NotImplementedError

    def clone(self, *args, **kwargs):
        raise NotImplementedError

    def coalesce(self):
        raise NotImplementedError

    def col_indices(self):
        raise NotImplementedError

    def conj(self):
        raise NotImplementedError

    def conj_physical(self):
        raise NotImplementedError

    def conj_physical_(self):
        raise NotImplementedError

    def contiguous(self, memory_format=None):
        raise NotImplementedError

    def copysign(self, other):
        raise NotImplementedError

    def copysign_(self, other):
        raise NotImplementedError

    def copy_(self, src, non_blocking=False):
        raise NotImplementedError

    def corrcoef(self):
        raise NotImplementedError

    def cos(self):
        raise NotImplementedError

    def cosh(self):
        raise NotImplementedError

    def cosh_(self):
        raise NotImplementedError

    def cos_(self):
        raise NotImplementedError

    def count_nonzero(self, dim=None):
        raise NotImplementedError

    def cov(self, *args, **kwargs):
        raise NotImplementedError

    def cpu(self, memory_format=None):
        raise NotImplementedError

    def cross(self, other, dim=None):
        raise NotImplementedError

    def crow_indices(self):
        raise NotImplementedError

    def cuda(self, device=None, non_blocking=False, memory_format=None):
        raise NotImplementedError

    def cummax(self, dim):
        raise NotImplementedError

    def cummin(self, dim):
        raise NotImplementedError

    def cumprod(self, dim, dtype=None):
        raise NotImplementedError

    def cumprod_(self, dim, dtype=None):
        raise NotImplementedError

    def cumsum(self, dim, dtype=None):
        raise NotImplementedError

    def cumsum_(self, dim, dtype=None):
        raise NotImplementedError

    def data_ptr(self):

        return 0

    def deg2rad(self):
        raise NotImplementedError

    def deg2rad_(self):
        raise NotImplementedError

    def dense_dim(self):

        return 0

    def dequantize(self):
        raise NotImplementedError

    def det(self):
        raise NotImplementedError

    def detach(self, *args, **kwargs):
        raise NotImplementedError

    def detach_(self, *args, **kwargs):
        raise NotImplementedError

    def diag(self, diagonal=0):
        raise NotImplementedError

    def diagflat(self, offset=0):
        raise NotImplementedError

    def diagonal(self, offset=0, dim1=0, dim2=1):
        raise NotImplementedError

    def diagonal_scatter(self, src, offset=0, dim1=0, dim2=1):
        raise NotImplementedError

    def diag_embed(self, offset=0, dim1=-2, dim2=-1):
        raise NotImplementedError

    def diff(self, n=1, dim=-1, prepend=None, append=None):
        raise NotImplementedError

    def digamma(self):
        raise NotImplementedError

    def digamma_(self):
        raise NotImplementedError

    def dim(self):

        return 0

    def dist(self, other, p=2):
        raise NotImplementedError

    def div(self, value, *args, **kwargs):
        raise NotImplementedError

    def divide(self, value, *args, **kwargs):
        raise NotImplementedError

    def divide_(self, value, *args, **kwargs):
        raise NotImplementedError

    def div_(self, value, *args, **kwargs):
        raise NotImplementedError

    def dot(self, other):
        raise NotImplementedError

    def double(self, memory_format=None):
        raise NotImplementedError

    def dsplit(self, split_size_or_sections):
        raise NotImplementedError

    def element_size(self):

        return 0

    def eq(self, other):
        raise NotImplementedError

    def equal(self, other):

        return False

    def eq_(self, other):
        raise NotImplementedError

    def erf(self):
        raise NotImplementedError

    def erfc(self):
        raise NotImplementedError

    def erfc_(self):
        raise NotImplementedError

    def erfinv(self):
        raise NotImplementedError

    def erfinv_(self):
        raise NotImplementedError

    def erf_(self):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def exp2(self):
        raise NotImplementedError

    def exp2_(self):
        raise NotImplementedError

    def expand(self, *sizes):
        raise NotImplementedError

    def expand_as(self, other):
        raise NotImplementedError

    def expm1(self):
        raise NotImplementedError

    def expm1_(self):
        raise NotImplementedError

    def exponential_(self, lambd=1, *args, **kwargs):
        raise NotImplementedError

    def exp_(self):
        raise NotImplementedError

    def fill_(self, value):
        raise NotImplementedError

    def fill_diagonal_(self, fill_value, wrap=False):
        raise NotImplementedError

    def fix(self):
        raise NotImplementedError

    def fix_(self):
        raise NotImplementedError

    def flatten(self, start_dim=0, end_dim=-1):
        raise NotImplementedError

    def flip(self, dims):
        raise NotImplementedError

    def fliplr(self):
        raise NotImplementedError

    def flipud(self):
        raise NotImplementedError

    def float(self, memory_format=None):
        raise NotImplementedError

    def float_power(self, exponent):
        raise NotImplementedError

    def float_power_(self, exponent):
        raise NotImplementedError

    def floor(self):
        raise NotImplementedError

    def floor_(self):
        raise NotImplementedError

    def floor_divide(self, value):
        raise NotImplementedError

    def floor_divide_(self, value):
        raise NotImplementedError

    def fmax(self, other):
        raise NotImplementedError

    def fmin(self, other):
        raise NotImplementedError

    def fmod(self, divisor):
        raise NotImplementedError

    def fmod_(self, divisor):
        raise NotImplementedError

    def frac(self):
        raise NotImplementedError

    def frac_(self):
        raise NotImplementedError

    def frexp(self, input):
        raise NotImplementedError

    def gather(self, dim, index):
        raise NotImplementedError

    def gcd(self, other):
        raise NotImplementedError

    def gcd_(self, other):
        raise NotImplementedError

    def ge(self, other):
        raise NotImplementedError

    def geometric_(self, p, *args, **kwargs):
        raise NotImplementedError

    def geqrf(self):
        raise NotImplementedError

    def ger(self, vec2):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError

    def ge_(self, other):
        raise NotImplementedError

    def greater(self, other):
        raise NotImplementedError

    def greater_(self, other):
        raise NotImplementedError

    def greater_equal(self, other):
        raise NotImplementedError

    def greater_equal_(self, other):
        raise NotImplementedError

    def gt(self, other):
        raise NotImplementedError

    def gt_(self, other):
        raise NotImplementedError

    def half(self, memory_format=None):
        raise NotImplementedError

    def hardshrink(self, lambd=0.5):
        raise NotImplementedError

    def has_names(self, *args, **kwargs):
        raise NotImplementedError

    def heaviside(self, values):
        raise NotImplementedError

    def heaviside_(self, values):
        raise NotImplementedError

    def histc(self, bins=100, min=0, max=0):
        raise NotImplementedError

    def histogram(self, input, bins, *args, **kwargs):
        raise NotImplementedError

    def hsplit(self, split_size_or_sections):
        raise NotImplementedError

    def hypot(self, other):
        raise NotImplementedError

    def hypot_(self, other):
        raise NotImplementedError

    def i0(self):
        raise NotImplementedError

    def i0_(self):
        raise NotImplementedError

    def igamma(self, other):
        raise NotImplementedError

    def igammac(self, other):
        raise NotImplementedError

    def igammac_(self, other):
        raise NotImplementedError

    def igamma_(self, other):
        raise NotImplementedError

    def index_add(self, dim, index, source, *args, **kwargs):
        raise NotImplementedError

    def index_add_(self, dim, index, source, *args, **kwargs):
        raise NotImplementedError

    def index_copy(self, dim, index, tensor2):
        raise NotImplementedError

    def index_copy_(self, dim, index, tensor):
        raise NotImplementedError

    def index_fill(self, dim, index, value):
        raise NotImplementedError

    def index_fill_(self, dim, index, value):
        raise NotImplementedError

    def index_put(self, indices, values, accumulate=False):
        raise NotImplementedError

    def index_put_(self, indices, values, accumulate=False):
        raise NotImplementedError

    def index_reduce(self, *args, **kwargs):
        raise NotImplementedError

    def index_reduce_(self, dim, index, source, reduce, *args, **kwargs):
        raise NotImplementedError

    def index_select(self, dim, index):
        raise NotImplementedError

    def indices(self):
        raise NotImplementedError

    def inner(self, other):
        raise NotImplementedError

    def int(self, memory_format=None):
        raise NotImplementedError

    def int_repr(self):
        raise NotImplementedError

    def inverse(self):
        raise NotImplementedError

    def ipu(self, device=None, non_blocking=False, memory_format=None):
        raise NotImplementedError

    def isclose(self, other, rtol=1, *args, **kwargs):
        raise NotImplementedError

    def isfinite(self):
        raise NotImplementedError

    def isinf(self):
        raise NotImplementedError

    def isnan(self):
        raise NotImplementedError

    def isneginf(self):
        raise NotImplementedError

    def isposinf(self):
        raise NotImplementedError

    def isreal(self):
        raise NotImplementedError

    def istft(
        self,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        center=True,
        normalized=False,
        onesided=True,
        length=None,
    ):
        raise NotImplementedError

    def is_coalesced(self):

        return False

    def is_complex(self):

        return False

    def is_conj(self):

        return False

    def is_contiguous(self, memory_format=None):

        return False

    def is_distributed(self, *args, **kwargs):
        raise NotImplementedError

    def is_floating_point(self):

        return False

    def is_inference(self):

        return False

    def is_neg(self):

        return False

    def is_nonzero(self, *args, **kwargs):
        raise NotImplementedError

    def is_pinned(self, *args, **kwargs):
        raise NotImplementedError

    def is_same_size(self, *args, **kwargs):
        raise NotImplementedError

    def is_set_to(self, tensor):

        return False

    def is_signed(self):

        return False

    def item(self):

        return 0

    def kron(self, other):
        raise NotImplementedError

    def kthvalue(self, k, dim=None, keepdim=False):
        raise NotImplementedError

    def lcm(self, other):
        raise NotImplementedError

    def lcm_(self, other):
        raise NotImplementedError

    def ldexp(self, other):
        raise NotImplementedError

    def ldexp_(self, other):
        raise NotImplementedError

    def le(self, other):
        raise NotImplementedError

    def lerp(self, end, weight):
        raise NotImplementedError

    def lerp_(self, end, weight):
        raise NotImplementedError

    def less(self, *args, **kwargs):
        raise NotImplementedError

    def less_(self, other):
        raise NotImplementedError

    def less_equal(self, other):
        raise NotImplementedError

    def less_equal_(self, other):
        raise NotImplementedError

    def le_(self, other):
        raise NotImplementedError

    def lgamma(self):
        raise NotImplementedError

    def lgamma_(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def log10(self):
        raise NotImplementedError

    def log10_(self):
        raise NotImplementedError

    def log1p(self):
        raise NotImplementedError

    def log1p_(self):
        raise NotImplementedError

    def log2(self):
        raise NotImplementedError

    def log2_(self):
        raise NotImplementedError

    def logaddexp(self, other):
        raise NotImplementedError

    def logaddexp2(self, other):
        raise NotImplementedError

    def logcumsumexp(self, dim):
        raise NotImplementedError

    def logdet(self):
        raise NotImplementedError

    def logical_and(self):
        raise NotImplementedError

    def logical_and_(self):
        raise NotImplementedError

    def logical_not(self):
        raise NotImplementedError

    def logical_not_(self):
        raise NotImplementedError

    def logical_or(self):
        raise NotImplementedError

    def logical_or_(self):
        raise NotImplementedError

    def logical_xor(self):
        raise NotImplementedError

    def logical_xor_(self):
        raise NotImplementedError

    def logit(self):
        raise NotImplementedError

    def logit_(self):
        raise NotImplementedError

    def logsumexp(self, dim, keepdim=False):
        raise NotImplementedError

    def log_(self):
        raise NotImplementedError

    def log_normal_(self, mean=1, std=2, *args, **kwargs):
        raise NotImplementedError

    def log_softmax(self, *args, **kwargs):
        raise NotImplementedError

    def long(self, memory_format=None):
        raise NotImplementedError

    def lt(self, other):
        raise NotImplementedError

    def lt_(self, other):
        raise NotImplementedError

    def lu_solve(self, LU_data, LU_pivots):
        raise NotImplementedError

    def map2_(self, *args, **kwargs):
        raise NotImplementedError

    def map_(self, tensor, callable):
        raise NotImplementedError

    def masked_fill(self, mask, value):
        raise NotImplementedError

    def masked_fill_(self, mask, value):
        raise NotImplementedError

    def masked_scatter(self, mask, tensor):
        raise NotImplementedError

    def masked_scatter_(self, mask, source):
        raise NotImplementedError

    def masked_select(self, mask):
        raise NotImplementedError

    def matmul(self, tensor2):
        raise NotImplementedError

    def matrix_exp(self):
        raise NotImplementedError

    def matrix_power(self, n):
        raise NotImplementedError

    def max(self, dim=None, keepdim=False):
        raise NotImplementedError

    def maximum(self, other):
        raise NotImplementedError

    def mean(self, dim=None, keepdim=False, *args, **kwargs):
        raise NotImplementedError

    def median(self, dim=None, keepdim=False):
        raise NotImplementedError

    def min(self, dim=None, keepdim=False):
        raise NotImplementedError

    def minimum(self, other):
        raise NotImplementedError

    def mm(self, mat2):
        raise NotImplementedError

    def mode(self, dim=None, keepdim=False):
        raise NotImplementedError

    def moveaxis(self, source, destination):
        raise NotImplementedError

    def movedim(self, source, destination):
        raise NotImplementedError

    def msort(self):
        raise NotImplementedError

    def mul(self, value):
        raise NotImplementedError

    def multinomial(self, num_samples, replacement=False, *args, **kwargs):
        raise NotImplementedError

    def multiply(self, value):
        raise NotImplementedError

    def multiply_(self, value):
        raise NotImplementedError

    def mul_(self, value):
        raise NotImplementedError

    def mv(self, vec):
        raise NotImplementedError

    def mvlgamma(self, p):
        raise NotImplementedError

    def mvlgamma_(self, p):
        raise NotImplementedError

    def nanmean(self, dim=None, keepdim=False, *args, **kwargs):
        raise NotImplementedError

    def nanmedian(self, dim=None, keepdim=False):
        raise NotImplementedError

    def nanquantile(self, q, dim=None, keepdim=False, *args, **kwargs):
        raise NotImplementedError

    def nansum(self, dim=None, keepdim=False, dtype=None):
        raise NotImplementedError

    def nan_to_num(self, nan=0.0, posinf=None, neginf=None):
        raise NotImplementedError

    def nan_to_num_(self, nan=0.0, posinf=None, neginf=None):
        raise NotImplementedError

    def narrow(self, dimension, start, length):
        raise NotImplementedError

    def narrow_copy(self, dimension, start, length):
        raise NotImplementedError

    def ndimension(self):

        return 0

    def ne(self, other):
        raise NotImplementedError

    def neg(self):
        raise NotImplementedError

    def negative(self):
        raise NotImplementedError

    def negative_(self):
        raise NotImplementedError

    def neg_(self):
        raise NotImplementedError

    def nelement(self):

        return 0

    def new(self, *args, **kwargs):
        raise NotImplementedError

    def new_empty(self, size, *args, **kwargs):
        raise NotImplementedError

    def new_empty_strided(
        self,
        size,
        stride,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False,
    ):
        raise NotImplementedError

    def new_full(self, size, fill_value, *args, **kwargs):
        raise NotImplementedError

    def new_ones(self, size, *args, **kwargs):
        raise NotImplementedError

    def new_tensor(self, data, *args, **kwargs):
        raise NotImplementedError

    def new_zeros(self, size, *args, **kwargs):
        raise NotImplementedError

    def nextafter(self, other):
        raise NotImplementedError

    def nextafter_(self, other):
        raise NotImplementedError

    def ne_(self, other):
        raise NotImplementedError

    def nonzero(self):
        raise NotImplementedError

    def norm(self, p=2, dim=None, keepdim=False):
        raise NotImplementedError

    def normal_(self, mean=0, std=1, *args, **kwargs):
        raise NotImplementedError

    def not_equal(self, other):
        raise NotImplementedError

    def not_equal_(self, other):
        raise NotImplementedError

    def numel(self):

        return 0

    def numpy(self, *args, **kwargs):
        raise NotImplementedError

    def orgqr(self, input2):
        raise NotImplementedError

    def ormqr(self, input2, input3, left=True, transpose=False):
        raise NotImplementedError

    def outer(self, vec2):
        raise NotImplementedError

    def permute(self, *dims):
        raise NotImplementedError

    def pinverse(self):
        raise NotImplementedError

    def pin_memory(self):
        raise NotImplementedError

    def polygamma(self, n):
        raise NotImplementedError

    def polygamma_(self, n):
        raise NotImplementedError

    def positive(self):
        raise NotImplementedError

    def pow(self, exponent):
        raise NotImplementedError

    def pow_(self, exponent):
        raise NotImplementedError

    def prelu(self, *args, **kwargs):
        raise NotImplementedError

    def prod(self, dim=None, keepdim=False, dtype=None):
        raise NotImplementedError

    def put(self, input, index, source, accumulate=False):
        raise NotImplementedError

    def put_(self, index, source, accumulate=False):
        raise NotImplementedError

    def qr(self, some=True):
        raise NotImplementedError

    def qscheme(self):
        raise NotImplementedError

    def quantile(self, q, dim=None, keepdim=False, *args, **kwargs):
        raise NotImplementedError

    def q_per_channel_axis(self):

        return 0

    def q_per_channel_scales(self):
        raise NotImplementedError

    def q_per_channel_zero_points(
        self,
    ):
        raise NotImplementedError

    def q_scale(self):

        return 0.0

    def q_zero_point(self):

        return 0

    def rad2deg(self):
        raise NotImplementedError

    def rad2deg_(self):
        raise NotImplementedError

    def random_(self, from_=0, to=None, *args, **kwargs):
        raise NotImplementedError

    def ravel(self):
        raise NotImplementedError

    def reciprocal(self):
        raise NotImplementedError

    def reciprocal_(self):
        raise NotImplementedError

    def record_stream(self, stream):
        raise NotImplementedError

    def refine_names(self, *args, **kwargs):
        raise NotImplementedError

    def relu(self, *args, **kwargs):
        raise NotImplementedError

    def relu_(self, *args, **kwargs):
        raise NotImplementedError

    def remainder(self, divisor):
        raise NotImplementedError

    def remainder_(self, divisor):
        raise NotImplementedError

    def rename(self, *args, **kwargs):
        raise NotImplementedError

    def rename_(self, *args, **kwargs):
        raise NotImplementedError

    def renorm(self, p, dim, maxnorm):
        raise NotImplementedError

    def renorm_(self, p, dim, maxnorm):
        raise NotImplementedError

    def repeat(self, *sizes):
        raise NotImplementedError

    def repeat_interleave(self, repeats, dim=None, *args, **kwargs):
        raise NotImplementedError

    def requires_grad_(self, requires_grad=True):
        raise NotImplementedError

    def reshape(self, *shape):
        raise NotImplementedError

    def reshape_as(self, other):
        raise NotImplementedError

    def resize_(self, *sizes, memory_format=None):
        raise NotImplementedError

    def resize_as_(self, tensor, memory_format=None):
        raise NotImplementedError

    def resize_as_sparse_(self, *args, **kwargs):
        raise NotImplementedError

    def resolve_conj(self):
        raise NotImplementedError

    def resolve_neg(self):
        raise NotImplementedError

    def retain_grad(self):
        raise NotImplementedError

    def roll(self, shifts, dims):
        raise NotImplementedError

    def rot90(self, k, dims):
        raise NotImplementedError

    def round(self, decimals=0):
        raise NotImplementedError

    def round_(self, decimals=0):
        raise NotImplementedError

    def row_indices(self, *args, **kwargs):
        raise NotImplementedError

    def rsqrt(self):
        raise NotImplementedError

    def rsqrt_(self):
        raise NotImplementedError

    def scatter(self, dim, index, src):
        raise NotImplementedError

    def scatter_(self, dim, index, src, reduce=None):
        raise NotImplementedError

    def scatter_add(self, dim, index, src):
        raise NotImplementedError

    def scatter_add_(self, dim, index, src):
        raise NotImplementedError

    def scatter_reduce(self, dim, index, src, reduce, *args, **kwargs):
        raise NotImplementedError

    def scatter_reduce_(self, dim, index, src, reduce, *args, **kwargs):
        raise NotImplementedError

    def select(self, dim, index):
        raise NotImplementedError

    def select_scatter(self, src, dim, index):
        raise NotImplementedError

    def set_(self, source=None, storage_offset=0, size=None, stride=None):
        raise NotImplementedError

    def sgn(self):
        raise NotImplementedError

    def sgn_(self):
        raise NotImplementedError

    def short(self, memory_format=None):
        raise NotImplementedError

    def sigmoid(self):
        raise NotImplementedError

    def sigmoid_(self):
        raise NotImplementedError

    def sign(self):
        raise NotImplementedError

    def signbit(self):
        raise NotImplementedError

    def sign_(self):
        raise NotImplementedError

    def sin(self):
        raise NotImplementedError

    def sinc(self):
        raise NotImplementedError

    def sinc_(self):
        raise NotImplementedError

    def sinh(self):
        raise NotImplementedError

    def sinh_(self):
        raise NotImplementedError

    def sin_(self):
        raise NotImplementedError

    def size(self, dim=None):
        raise NotImplementedError

    def slice_scatter(self, src, dim=0, start=None, end=None, step=1):
        raise NotImplementedError

    def slogdet(self):
        raise NotImplementedError

    def smm(self, mat):
        raise NotImplementedError

    def softmax(self, *args, **kwargs):
        raise NotImplementedError

    def sort(self, dim=-1, descending=False):
        raise NotImplementedError

    def sparse_dim(self):

        return 0

    def sparse_mask(self, mask):
        raise NotImplementedError

    def sparse_resize_(self, size, sparse_dim, dense_dim):
        raise NotImplementedError

    def sparse_resize_and_clear_(self, size, sparse_dim, dense_dim):
        raise NotImplementedError

    def split(self, *args, **kwargs):
        raise NotImplementedError

    def split_with_sizes(self, *args, **kwargs):
        raise NotImplementedError

    def sqrt(self):
        raise NotImplementedError

    def sqrt_(self):
        raise NotImplementedError

    def square(self):
        raise NotImplementedError

    def square_(self):
        raise NotImplementedError

    def squeeze(self, dim=None):
        raise NotImplementedError

    def squeeze_(self, dim=None):
        raise NotImplementedError

    def sspaddmm(self, mat1, mat2, *args, **kwargs):
        raise NotImplementedError

    def std(self, dim, unbiased=True, keepdim=False):
        raise NotImplementedError

    def stft(
        self,
        frame_length,
        hop,
        fft_size=None,
        return_onesided=True,
        window=None,
        pad_end=0,
    ):
        raise NotImplementedError

    def storage_offset(self):

        return 0

    def stride(self, dim):

        return ()

    def sub(self, other, *args, **kwargs):
        raise NotImplementedError

    def subtract(self, other, *args, **kwargs):
        raise NotImplementedError

    def subtract_(self, other, *args, **kwargs):
        raise NotImplementedError

    def sub_(self, other, *args, **kwargs):
        raise NotImplementedError

    def sum(self, dim=None, keepdim=False, dtype=None):
        raise NotImplementedError

    def sum_to_size(self, *size):
        raise NotImplementedError

    def svd(self, some=True, compute_uv=True):
        raise NotImplementedError

    def swapaxes(self, axis0, axis1):
        raise NotImplementedError

    def swapaxes_(self, axis0, axis1):
        raise NotImplementedError

    def swapdims(self, dim0, dim1):
        raise NotImplementedError

    def swapdims_(self, dim0, dim1):
        raise NotImplementedError

    def symeig(self, eigenvectors=False, upper=True):
        raise NotImplementedError

    def t(self):
        raise NotImplementedError

    def take(self, indices):
        raise NotImplementedError

    def take_along_dim(self, indices, dim):
        raise NotImplementedError

    def tan(self):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def tanh_(self):
        raise NotImplementedError

    def tan_(self):
        raise NotImplementedError

    def tensor_split(self, indices_or_sections, dim=0):
        raise NotImplementedError

    def tile(self, *reps):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        raise NotImplementedError

    def tolist(self):
        raise NotImplementedError

    def topk(self, k, dim=None, largest=True, sorted=True):
        raise NotImplementedError

    def to_dense(self):
        raise NotImplementedError

    def to_mkldnn(self):
        raise NotImplementedError

    def to_padded_tensor(self, padding, output_size=None):
        raise NotImplementedError

    def to_sparse(self, sparseDims):
        raise NotImplementedError

    def to_sparse_bsc(self, blocksize):
        raise NotImplementedError

    def to_sparse_bsr(self, blocksize):
        raise NotImplementedError

    def to_sparse_csc(self):
        raise NotImplementedError

    def to_sparse_csr(self):
        raise NotImplementedError

    def trace(self):
        raise NotImplementedError

    def transpose(self, dim0, dim1):
        raise NotImplementedError

    def transpose_(self, dim0, dim1):
        raise NotImplementedError

    def triangular_solve(self, A, upper=True, transpose=False, unitriangular=False):
        raise NotImplementedError

    def tril(self, diagonal=0):
        raise NotImplementedError

    def tril_(self, diagonal=0):
        raise NotImplementedError

    def triu(self, diagonal=0):
        raise NotImplementedError

    def triu_(self, diagonal=0):
        raise NotImplementedError

    def true_divide(self, value):
        raise NotImplementedError

    def true_divide_(self, value):
        raise NotImplementedError

    def trunc(self):
        raise NotImplementedError

    def trunc_(self):
        raise NotImplementedError

    # def type(
    #     self, dtype=None, non_blocking=False, **kwargs
    # ):
    #     return ""

    def type_as(self, tensor):
        raise NotImplementedError

    def t_(self):
        raise NotImplementedError

    def unbind(self, dim=0):
        raise NotImplementedError

    def unflatten(self, *args, **kwargs):
        raise NotImplementedError

    def unfold(self, dimension, size, step):
        raise NotImplementedError

    def uniform_(self, from_=0, to=1):
        raise NotImplementedError

    def unsafe_chunk(self, chunks, dim=0):
        raise NotImplementedError

    def unsafe_split(self, split_size, dim=0):
        raise NotImplementedError

    def unsafe_split_with_sizes(self, *args, **kwargs):
        raise NotImplementedError

    def unsqueeze(self, dim):
        raise NotImplementedError

    def unsqueeze_(self, dim):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def var(self, dim, unbiased=True, keepdim=False):
        raise NotImplementedError

    def vdot(self, other):
        raise NotImplementedError

    def view(self, *shape):
        raise NotImplementedError

    def view_as(self, other):
        raise NotImplementedError

    def vsplit(self, split_size_or_sections):
        raise NotImplementedError

    def where(self, condition, y):
        raise NotImplementedError

    def xlogy(self, other):
        raise NotImplementedError

    def xlogy_(self, other):
        raise NotImplementedError

    def xpu(self, device=None, non_blocking=False, memory_format=None):
        raise NotImplementedError

    def zero_(self):
        raise NotImplementedError

    def _addmm_activation(self, *args, **kwargs):
        raise NotImplementedError

    def _autocast_to_full_precision(self, *args, **kwargs):
        raise NotImplementedError

    def _autocast_to_reduced_precision(self, *args, **kwargs):
        raise NotImplementedError

    def _coalesced_(self, *args, **kwargs):
        raise NotImplementedError

    def _conj(self, *args, **kwargs):
        raise NotImplementedError

    def _conj_physical(self, *args, **kwargs):
        raise NotImplementedError

    def _dimI(self, *args, **kwargs):
        raise NotImplementedError

    def _dimV(self, *args, **kwargs):
        raise NotImplementedError

    def _fix_weakref(self, *args, **kwargs):
        raise NotImplementedError

    def _indices(self, *args, **kwargs):
        raise NotImplementedError

    def _is_view(self, *args, **kwargs):
        raise NotImplementedError

    def _is_zerotensor(self, *args, **kwargs):
        raise NotImplementedError

    def _make_subclass(self, *args, **kwargs):
        raise NotImplementedError

    def _make_wrapper_subclass(self, *args, **kwargs):
        raise NotImplementedError

    def _neg_view(self, *args, **kwargs):
        raise NotImplementedError

    def _nested_tensor_size(self, *args, **kwargs):
        raise NotImplementedError

    def _nnz(self, *args, **kwargs):
        raise NotImplementedError

    def _storage(self, *args, **kwargs):
        raise NotImplementedError

    def _to_dense(self, *args, **kwargs):
        raise NotImplementedError

    def _values(self, *args, **kwargs):
        raise NotImplementedError

    def __add__(self, *args, **kwargs):
        return pi.add_Tensor(self, args[0])

    def __and__(self, *args, **kwargs):
        raise NotImplementedError

    def __bool__(self, *args, **kwargs):
        raise NotImplementedError

    def __complex__(self, *args, **kwargs):
        raise NotImplementedError

    def __delitem__(self, *args, **kwargs):
        raise NotImplementedError

    def __div__(self, *args, **kwargs):
        raise NotImplementedError

    def __eq__(self, *args, **kwargs):
        raise NotImplementedError

    def __float__(self, *args, **kwargs):
        raise NotImplementedError

    def __floordiv__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, *args, **kwargs):
        raise NotImplementedError

    def __ge__(self, *args, **kwargs):
        raise NotImplementedError

    def __gt__(self, *args, **kwargs):
        raise NotImplementedError

    #
    # def __iadd__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __iand__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __idiv__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __ifloordiv__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __ilshift__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __imod__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __imul__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __index__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    def __int__(self, *args, **kwargs):
        raise NotImplementedError

    def __invert__(self, *args, **kwargs):
        raise NotImplementedError

    #
    # def __ior__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __irshift__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __isub__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __ixor__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    def __len__(self, *args, **kwargs):
        raise NotImplementedError

    def __le__(self, *args, **kwargs):
        raise NotImplementedError

    def __long__(self, *args, **kwargs):
        raise NotImplementedError

    #
    # def __lshift__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    def __lt__(self, *args, **kwargs):
        raise NotImplementedError

    #
    # def __matmul__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    def __mod__(self, *args, **kwargs):
        raise NotImplementedError

    #
    def __mul__(self, *args, **kwargs):
        return pi.mul_Tensor(self, args[0])

    def __ne__(self, *args, **kwargs):
        raise NotImplementedError

    def __nonzero__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self, *args, **kwargs):
        raise NotImplementedError

    # def __radd__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __rand__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __rmul__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __ror__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __rshift__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def __rxor__(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError

    def __sub__(self, *args, **kwargs):
        raise NotImplementedError

    def __truediv__(self, *args, **kwargs):
        raise NotImplementedError

    # def __xor__(self, *args, **kwargs):
    #     raise NotImplementedError

    # data = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # device = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # dtype = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # grad = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # grad_fn = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # H = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # imag = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_cpu = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_cuda = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_ipu = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_leaf = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_meta = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_mkldnn = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_mps = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_nested = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_ort = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_quantized = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_sparse = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_sparse_csr = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_vulkan = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # is_xpu = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # layout = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # mH = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # mT = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # name = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # names = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # ndim = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # output_nr = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # real = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # requires_grad = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # retains_grad = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # shape = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # T = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # volatile = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # _backward_hooks = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # _base = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # _cdata = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # _grad = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # _grad_fn = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # _has_symbolic_sizes_strides = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # _python_dispatch = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default
    #
    # _version = property(
    #     lambda self: object(), lambda self, v: None, lambda self: None
    # )  # default


def from_numpy(arr: np.ndarray):
    from pi import DEBUG

    if DEBUG:
        arr = np.ones_like(arr, dtype=np.float32)
    attr = DenseElementsAttr.get(arr)
    vt = Tensor(torch_dialect.ValueTensorLiteralOp(attr))
    return vt


def empty(shape: Tuple[int, ...], dtype: "pi.dtype" = None, **kwargs) -> Tensor:
    if np.prod(shape) == 0:
        return Tensor(None)
    else:
        if dtype is not None:
            dtype = dtype.to_np_type()

    return from_numpy(np.empty(shape, dtype))


def randint(low: int, high: int, size: Tuple[int, ...]) -> Tensor:
    return from_numpy(np.random.randint(low, high, size))


def randn(*size: Tuple[int, ...]) -> Tensor:
    return from_numpy(np.random.randn(*size))


def uniform(low: float, high: float, size: Tuple[int, ...]) -> Tensor:
    return from_numpy(np.random.uniform(low, high, size))


def rand(*size: Tuple[int, ...], **kwargs) -> Tensor:
    dtype = kwargs.get("dtype", None)
    if dtype is not None:
        dtype = dtype.to_np_type()
    return from_numpy(np.random.rand(*size))


def ones(*size: Tuple[int, ...], **kwargs) -> Tensor:
    # dtype: "pi.dtype" = None, _device: Any = None
    dtype = kwargs.get("dtype", None)
    if dtype is not None:
        dtype = dtype.to_np_type()
    return from_numpy(np.ones(size, dtype=dtype))


def zeros(*size: Tuple[int, ...], **kwargs) -> Tensor:
    dtype = kwargs.get("dtype", None)
    if dtype is not None:
        dtype = dtype.to_np_type()
    return from_numpy(np.zeros(size, dtype))


def tensor(data: Any, dtype: Optional["pi.dtype"] = None) -> Tensor:
    if dtype is not None:
        dtype = dtype.to_np_type()

    return from_numpy(np.array(data, dtype=dtype))


def LongTensor(data: Any) -> Tensor:
    return from_numpy(np.array(data, dtype=shark_dtype.int64.to_np_type()))


def clone(x: Tensor, **kwargs):
    # TODO(max): is this necessary?
    warnings.warn(f"not actually cloning")
    return x


__all__ = [
    "from_numpy",
    "empty",
    "randint",
    "randn",
    "rand",
    "uniform",
    "ones",
    "tensor",
    "Tensor",
    "LongTensor",
    "zeros",
]
