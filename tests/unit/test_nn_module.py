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
                str(t.owner),
            )
