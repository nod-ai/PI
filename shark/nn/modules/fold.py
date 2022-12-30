# -*- coding: utf-8 -*-
from .module import Module
from .. import functional as F

from shark import Tensor
from ..common_types import _size_any_t

__all__ = ["Fold", "Unfold"]


class Fold(Module):

    __constants__ = ["output_size", "kernel_size", "dilation", "padding", "stride"]
    output_size: _size_any_t
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t

    def __init__(
        self,
        output_size: _size_any_t,
        kernel_size: _size_any_t,
        dilation: _size_any_t = 1,
        padding: _size_any_t = 0,
        stride: _size_any_t = 1,
    ) -> None:
        super(Fold, self).__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        return F.fold(
            input,
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

    def extra_repr(self) -> str:
        return (
            "output_size={output_size}, kernel_size={kernel_size}, "
            "dilation={dilation}, padding={padding}, stride={stride}".format(
                **self.__dict__
            )
        )


class Unfold(Module):

    __constants__ = ["kernel_size", "dilation", "padding", "stride"]
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t

    def __init__(
        self,
        kernel_size: _size_any_t,
        dilation: _size_any_t = 1,
        padding: _size_any_t = 0,
        stride: _size_any_t = 1,
    ) -> None:
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        return F.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride
        )

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, dilation={dilation}, padding={padding},"
            " stride={stride}".format(**self.__dict__)
        )
