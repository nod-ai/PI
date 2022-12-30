from shark import Tensor
from .module import Module
from .. import functional as F

__all__ = ["ChannelShuffle"]


class ChannelShuffle(Module):

    __constants__ = ["groups"]
    groups: int

    def __init__(self, groups: int) -> None:
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, input: Tensor) -> Tensor:
        return F.channel_shuffle(input, self.groups)

    def extra_repr(self) -> str:
        return "groups={}".format(self.groups)
