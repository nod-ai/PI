from textwrap import dedent

import numpy as np

import pi
from pi.mlir.utils import mlir_mod_ctx
from pi import Tensor
from pi.nn.modules import (
    activation,
    adaptive,
    batchnorm,
    channelshuffle,
    container,
    conv,
    distance,
    dropout,
    flatten,
    fold,
    hooks,
    instancenorm,
    linear,
    loss,
    module,
    normalization,
    padding,
    pixelshuffle,
    pooling,
    sparse,
    transformer,
    upsampling,
)
from util import check_correct
from pi.mlir import torch_dialect as torch


class TestNnModule:
    def test_linear(self):
        with mlir_mod_ctx():
            t = pi.rand(10, 10)
            lin = linear.Linear(10, 10)
            t = lin(t)
            check_correct(
                "%3 = torch.aten.linear %0, %2, %1 : !torch.tensor<[10,10],f64>, !torch.tensor<[10,10],f64>, !torch.tensor<[10],f64> -> !torch.tensor",
                t.owner,
            )

    def test_conv(self):
        with mlir_mod_ctx():
            t = pi.rand(1, 3, 32, 32)
            con = conv.Conv2d(3, 3, 3)
            t = con(t)
            check_correct(
                "%6 = torch.aten.conv2d %0, %2, %1, %3, %4, %5, %int1_4 : !torch.tensor<[1,3,32,32],f64>, !torch.tensor<[3,3,3,3],f64>, !torch.tensor<[3],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.tensor",
                t.owner,
            )
