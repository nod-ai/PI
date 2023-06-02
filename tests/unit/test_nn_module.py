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

    def test_activations(self):
        with mlir_mod_ctx():
            t = pi.rand(1, 3, 32, 32)
            t = activation.ReLU()(t)
            check_correct(
                "%1 = torch.aten.relu %0 : !torch.tensor<[1,3,32,32],f64> -> !torch.tensor",
                t.owner,
            )

            t = activation.Sigmoid()(t)
            check_correct(
                "%2 = torch.aten.sigmoid %1 : !torch.tensor -> !torch.tensor",
                t.owner,
            )

            t = activation.LeakyReLU()(t)
            check_correct(
                "%3 = torch.aten.leaky_relu %2, %float9.999990e-03 : !torch.tensor, !torch.float -> !torch.tensor",
                t.owner,
            )

            t = activation.LogSoftmax()(t)
            check_correct(
                "%5 = torch.aten.log_softmax.int %3, %int1, %none : !torch.tensor, !torch.int, !torch.none -> !torch.tensor",
                t.owner,
            )

    def test_pooling(self):
        with mlir_mod_ctx() as module:
            t = pi.rand(1, 3, 32, 32)

            t = pooling.MaxPool2d(3)(t)
            check_correct(
                "%5 = torch.aten.max_pool2d %0, %1, %2, %3, %4, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor",
                t.owner,
            )

            t = pooling.MaxPool2d((3, 3))(t)
            check_correct(
                "%5 = torch.aten.max_pool2d %0, %1, %2, %3, %4, %false : !torch.tensor, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor",
                t.owner,
            )

            t = pooling.AvgPool2d(3)(t)
            check_correct(
                "%14 = torch.aten.avg_pool2d %10, %11, %12, %13, %false_20, %true, %none : !torch.tensor, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.tensor",
                t.owner,
            )

            t = pooling.AvgPool2d((3, 3))(t)
            check_correct(
                "%15 = torch.aten.avg_pool2d %10, %11, %12, %13, %false_20, %true, %none : !torch.tensor, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.tensor",
                t.owner,
            )

    def test_normalize(self):
        with mlir_mod_ctx() as module:
            t = pi.rand(1, 3, 32, 32)

            t = batchnorm.BatchNorm2d(3)(t)
            check_correct(
                "%6 = torch.aten.batch_norm %0, %5, %4, %2, %3, %false, %float1.000000e-01, %float9.999990e-06, %false_0 : !torch.tensor<[1,3,32,32],f64>, !torch.tensor<[3],f64>, !torch.tensor<[3],f64>, !torch.tensor<[3],f64>, !torch.tensor<[3],f64>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.tensor",
                t.owner,
            )

    def test_padding(self):
        with mlir_mod_ctx() as module:
            t = pi.rand(1, 3, 32, 32)

            t = padding.ZeroPad2d((10, 20, 30, 40))(t)

            check_correct(
                "%2 = torch.aten.pad %0, %1, %str, %float0.000000e00 : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.str, !torch.float -> !torch.tensor",
                t.owner,
            )
