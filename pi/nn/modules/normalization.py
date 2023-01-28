import pi
import numbers
from .module import Module

# from ._functions import CrossMapLRN2d as _cross_map_lrn2d
from .. import functional as F
from .. import init
from ..parameter import UninitializedParameter

from pi import Tensor
from typing import Union, List, Tuple

__all__ = [
    "LocalResponseNorm",
    # "CrossMapLRN2d",
    "LayerNorm",
    "GroupNorm",
]


class LocalResponseNorm(Module):

    __constants__ = ["size", "alpha", "beta", "k"]
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
    ) -> None:
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        return F.local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self):
        return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)


# class CrossMapLRN2d(Module):
#     size: int
#     alpha: float
#     beta: float
#     k: float
#
#     def __init__(
#         self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1
#     ) -> None:
#         super(CrossMapLRN2d, self).__init__()
#         self.size = size
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k
#
#     def forward(self, input: Tensor) -> Tensor:
#         return _cross_map_lrn2d.apply(input, self.size, self.alpha, self.beta, self.k)
#
#     def extra_repr(self) -> str:
#         return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)


_shape_t = Union[int, List[int]]


class LayerNorm(Module):

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = UninitializedParameter(
                self.normalized_shape, **factory_kwargs
            )
            self.bias = UninitializedParameter(self.normalized_shape, **factory_kwargs)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class GroupNorm(Module):

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GroupNorm, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = UninitializedParameter(num_channels, **factory_kwargs)
            self.bias = UninitializedParameter(num_channels, **factory_kwargs)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
