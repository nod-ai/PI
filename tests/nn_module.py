from shark.nn.module import Module

from shark import Empty
from shark import nn


class MyConv2d(Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        y = self.conv(x)
        z = y + y
        w = z * z
        return w


def simple_conv2d():
    x = Empty((1, 3, 32, 32))
    my_conv = MyConv2d()
    y = my_conv(x)
    return y


simple_conv2d()
