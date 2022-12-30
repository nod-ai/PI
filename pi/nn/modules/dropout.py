from .module import Module
from .. import functional as F

from pi import Tensor

__all__ = [
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
]


class _DropoutNd(Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)


class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)


class Dropout1d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout1d(input, self.p, self.training, self.inplace)


class Dropout2d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout2d(input, self.p, self.training, self.inplace)


class Dropout3d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout3d(input, self.p, self.training, self.inplace)


class AlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.alpha_dropout(input, self.p, self.training)


class FeatureAlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.feature_alpha_dropout(input, self.p, self.training)
