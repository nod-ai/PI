from .module import Module
from .. import functional as F

from pi import Tensor
from typing import Optional
from ..common_types import _size_2_t, _ratio_2_t, _size_any_t, _ratio_any_t

__all__ = ["Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d"]


class Upsample(Module):

    __constants__ = [
        "size",
        "scale_factor",
        "mode",
        "align_corners",
        "name",
        "recompute_scale_factor",
    ]
    name: str
    size: Optional[_size_any_t]
    scale_factor: Optional[_ratio_any_t]
    mode: str
    align_corners: Optional[bool]
    recompute_scale_factor: Optional[bool]

    def __init__(
        self,
        size: Optional[_size_any_t] = None,
        scale_factor: Optional[_ratio_any_t] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ) -> None:
        super(Upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            info = "scale_factor=" + str(self.scale_factor)
        else:
            info = "size=" + str(self.size)
        info += ", mode=" + self.mode
        return info


class UpsamplingNearest2d(Upsample):
    def __init__(
        self,
        size: Optional[_size_2_t] = None,
        scale_factor: Optional[_ratio_2_t] = None,
    ) -> None:
        super(UpsamplingNearest2d, self).__init__(size, scale_factor, mode="nearest")


class UpsamplingBilinear2d(Upsample):
    def __init__(
        self,
        size: Optional[_size_2_t] = None,
        scale_factor: Optional[_ratio_2_t] = None,
    ) -> None:
        super(UpsamplingBilinear2d, self).__init__(
            size, scale_factor, mode="bilinear", align_corners=True
        )
