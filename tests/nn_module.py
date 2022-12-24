import shark
from shark import nn


def simple_conv2d():
    x = nn.Tensor((1, 3, 32, 32))
    conv = nn.Conv2d(3, 1, 3)
    y = conv(x)
    return y


simple_conv2d()
