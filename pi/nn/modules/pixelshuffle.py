from .module import Module
from .. import functional as F

from pi import Tensor

__all__ = ["PixelShuffle", "PixelUnshuffle"]


class PixelShuffle(Module):

    __constants__ = ["upscale_factor"]
    upscale_factor: int

    def __init__(self, upscale_factor: int) -> None:
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self) -> str:
        return "upscale_factor={}".format(self.upscale_factor)


class PixelUnshuffle(Module):

    __constants__ = ["downscale_factor"]
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        return "downscale_factor={}".format(self.downscale_factor)
