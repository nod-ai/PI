from typing import Sequence, Tuple

from .module import Module
from .utils import _pair, _quadruple, _ntuple
from .. import functional as F

from pi import Tensor
from ..common_types import _size_2_t, _size_4_t, _size_6_t


__all__ = [
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad2d",
]


class _ConstantPadNd(Module):
    __constants__ = ["padding", "value"]
    value: float
    padding: Sequence[int]

    def __init__(self, value: float) -> None:
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "constant", self.value)

    def extra_repr(self) -> str:
        return "padding={}, value={}".format(self.padding, self.value)


class ConstantPad1d(_ConstantPadNd):
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t, value: float):
        super(ConstantPad1d, self).__init__(value)
        self.padding = _pair(padding)


class ConstantPad2d(_ConstantPadNd):
    __constants__ = ["padding", "value"]
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t, value: float) -> None:
        super(ConstantPad2d, self).__init__(value)
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t, value: float) -> None:
        super(ConstantPad3d, self).__init__(value)
        self.padding = _ntuple(6)(padding)


class _ReflectionPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "reflect")

    def extra_repr(self) -> str:
        return "{}".format(self.padding)


class ReflectionPad1d(_ReflectionPadNd):
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super(ReflectionPad1d, self).__init__()
        self.padding = _pair(padding)


class ReflectionPad2d(_ReflectionPadNd):
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super(ReflectionPad2d, self).__init__()
        self.padding = _quadruple(padding)


class ReflectionPad3d(_ReflectionPadNd):
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)


class _ReplicationPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "replicate")

    def extra_repr(self) -> str:
        return "{}".format(self.padding)


class ReplicationPad1d(_ReplicationPadNd):
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = _pair(padding)


class ReplicationPad2d(_ReplicationPadNd):
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super(ReplicationPad2d, self).__init__()
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super(ReplicationPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)


class ZeroPad2d(ConstantPad2d):
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super(ZeroPad2d, self).__init__(padding, 0.0)

    def extra_repr(self) -> str:
        return "{}".format(self.padding)
