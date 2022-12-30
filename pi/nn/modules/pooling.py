from typing import List, Optional

from pi import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F

from ..common_types import (
    _size_any_t,
    _size_1_t,
    _size_2_t,
    _size_3_t,
    _ratio_3_t,
    _ratio_2_t,
    _size_any_opt_t,
    _size_2_opt_t,
    _size_3_opt_t,
)

__all__ = [
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "LPPool1d",
    "LPPool2d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
]


class _MaxPoolNd(Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ]
    return_indices: bool
    ceil_mode: bool

    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", dilation={dilation}, ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class MaxPool1d(_MaxPoolNd):

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t

    def forward(self, input: Tensor):
        return F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MaxPool2d(_MaxPoolNd):

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: Tensor):
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MaxPool3d(_MaxPoolNd):
    # noqa: E501

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    def forward(self, input: Tensor):
        return F.max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class _MaxUnpoolNd(Module):
    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}".format(
            self.kernel_size, self.stride, self.padding
        )


class MaxUnpool1d(_MaxUnpoolNd):

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
    ) -> None:
        super(MaxUnpool1d, self).__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if (stride is not None) else kernel_size)
        self.padding = _single(padding)

    def forward(
        self, input: Tensor, indices: Tensor, output_size: Optional[List[int]] = None
    ) -> Tensor:
        return F.max_unpool1d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )


class MaxUnpool2d(_MaxUnpoolNd):

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
    ) -> None:
        super(MaxUnpool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if (stride is not None) else kernel_size)
        self.padding = _pair(padding)

    def forward(
        self, input: Tensor, indices: Tensor, output_size: Optional[List[int]] = None
    ) -> Tensor:
        return F.max_unpool2d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )


class MaxUnpool3d(_MaxUnpoolNd):

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
    ) -> None:
        super(MaxUnpool3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if (stride is not None) else kernel_size)
        self.padding = _triple(padding)

    def forward(
        self, input: Tensor, indices: Tensor, output_size: Optional[List[int]] = None
    ) -> Tensor:
        return F.max_unpool3d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )


class _AvgPoolNd(Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
    ]

    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}".format(
            self.kernel_size, self.stride, self.padding
        )


class AvgPool1d(_AvgPoolNd):

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super(AvgPool1d, self).__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class AvgPool2d(_AvgPoolNd):

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )


class AvgPool3d(_AvgPoolNd):

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super(AvgPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    def __setstate__(self, d):
        super(AvgPool3d, self).__setstate__(d)
        self.__dict__.setdefault("padding", 0)
        self.__dict__.setdefault("ceil_mode", False)
        self.__dict__.setdefault("count_include_pad", True)


class FractionalMaxPool2d(Module):

    __constants__ = ["kernel_size", "return_indices", "output_size", "output_ratio"]

    kernel_size: _size_2_t
    return_indices: bool
    output_size: _size_2_t
    output_ratio: _ratio_2_t

    def __init__(
        self,
        kernel_size: _size_2_t,
        output_size: Optional[_size_2_t] = None,
        output_ratio: Optional[_ratio_2_t] = None,
        return_indices: bool = False,
        _random_samples=None,
    ) -> None:
        super(FractionalMaxPool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.return_indices = return_indices
        self.register_buffer("_random_samples", _random_samples)
        self.output_size = _pair(output_size) if output_size is not None else None
        self.output_ratio = _pair(output_ratio) if output_ratio is not None else None
        if output_size is None and output_ratio is None:
            raise ValueError(
                "FractionalMaxPool2d requires specifying either "
                "an output size, or a pooling ratio"
            )
        if output_size is not None and output_ratio is not None:
            raise ValueError(
                "only one of output_size and output_ratio may be specified"
            )
        if self.output_ratio is not None:
            if not (0 < self.output_ratio[0] < 1 and 0 < self.output_ratio[1] < 1):
                raise ValueError(
                    "output_ratio must be between 0 and 1 (got {})".format(output_ratio)
                )

    def forward(self, input: Tensor):
        return F.fractional_max_pool2d(
            input,
            self.kernel_size,
            self.output_size,
            self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples,
        )


class FractionalMaxPool3d(Module):

    __constants__ = ["kernel_size", "return_indices", "output_size", "output_ratio"]
    kernel_size: _size_3_t
    return_indices: bool
    output_size: _size_3_t
    output_ratio: _ratio_3_t

    def __init__(
        self,
        kernel_size: _size_3_t,
        output_size: Optional[_size_3_t] = None,
        output_ratio: Optional[_ratio_3_t] = None,
        return_indices: bool = False,
        _random_samples=None,
    ) -> None:
        super(FractionalMaxPool3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.return_indices = return_indices
        self.register_buffer("_random_samples", _random_samples)
        self.output_size = _triple(output_size) if output_size is not None else None
        self.output_ratio = _triple(output_ratio) if output_ratio is not None else None
        if output_size is None and output_ratio is None:
            raise ValueError(
                "FractionalMaxPool3d requires specifying either "
                "an output size, or a pooling ratio"
            )
        if output_size is not None and output_ratio is not None:
            raise ValueError(
                "only one of output_size and output_ratio may be specified"
            )
        if self.output_ratio is not None:
            if not (
                0 < self.output_ratio[0] < 1
                and 0 < self.output_ratio[1] < 1
                and 0 < self.output_ratio[2] < 1
            ):
                raise ValueError(
                    "output_ratio must be between 0 and 1 (got {})".format(output_ratio)
                )

    def forward(self, input: Tensor):
        return F.fractional_max_pool3d(
            input,
            self.kernel_size,
            self.output_size,
            self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples,
        )


class _LPPoolNd(Module):
    __constants__ = ["norm_type", "kernel_size", "stride", "ceil_mode"]

    norm_type: float
    ceil_mode: bool

    def __init__(
        self,
        norm_type: float,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        ceil_mode: bool = False,
    ) -> None:
        super(_LPPoolNd, self).__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return (
            "norm_type={norm_type}, kernel_size={kernel_size}, stride={stride}, "
            "ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class LPPool1d(_LPPoolNd):

    kernel_size: _size_1_t
    stride: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        return F.lp_pool1d(
            input, float(self.norm_type), self.kernel_size, self.stride, self.ceil_mode
        )


class LPPool2d(_LPPoolNd):

    kernel_size: _size_2_t
    stride: _size_2_t

    def forward(self, input: Tensor) -> Tensor:
        return F.lp_pool2d(
            input, float(self.norm_type), self.kernel_size, self.stride, self.ceil_mode
        )


class _AdaptiveMaxPoolNd(Module):
    __constants__ = ["output_size", "return_indices"]
    return_indices: bool

    def __init__(
        self, output_size: _size_any_opt_t, return_indices: bool = False
    ) -> None:
        super(_AdaptiveMaxPoolNd, self).__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def extra_repr(self) -> str:
        return "output_size={}".format(self.output_size)


# FIXME (by @ssnl): Improve adaptive pooling docs: specify what the input and
#   output shapes are, and how the operation computes output.


class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):

    output_size: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_max_pool1d(input, self.output_size, self.return_indices)


class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):

    output_size: _size_2_opt_t

    def forward(self, input: Tensor):
        return F.adaptive_max_pool2d(input, self.output_size, self.return_indices)


class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):

    output_size: _size_3_opt_t

    def forward(self, input: Tensor):
        return F.adaptive_max_pool3d(input, self.output_size, self.return_indices)


class _AdaptiveAvgPoolNd(Module):
    __constants__ = ["output_size"]

    def __init__(self, output_size: _size_any_opt_t) -> None:
        super(_AdaptiveAvgPoolNd, self).__init__()
        self.output_size = output_size

    def extra_repr(self) -> str:
        return "output_size={}".format(self.output_size)


class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):

    output_size: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):

    output_size: _size_2_opt_t

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(input, self.output_size)


class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd):

    output_size: _size_3_opt_t

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool3d(input, self.output_size)
