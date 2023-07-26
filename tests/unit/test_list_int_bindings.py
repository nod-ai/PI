import pi
from pi import ops
from pi.mlir.utils import mlir_mod_ctx, non_value_tensor_op
from pi import Tensor
import numpy as np

# noinspection PyUnresolvedReferences
from pi.nn.functional import max_pool2d, max_pool2d_with_indices
from pi.nn.modules import (
    pooling,
)
from util import check_correct


class TestListIntBindings:
    def test_max_pool2d(self):
        with mlir_mod_ctx():
            t = pi.rand(1, 3, 32, 32)
            l = ops.max_pool2d(t, 3, 1)
            check_correct(
                " %4 = torch.aten.max_pool2d %arg0, %1, %2, %3, %0, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor",
                l.owner,
            )

            t = pi.rand(1, 3, 32, 32)
            l = ops.max_pool2d(t, (3, 3), 1)
            check_correct(
                " %4 = torch.aten.max_pool2d %arg0, %1, %2, %3, %0, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor",
                l.owner,
            )

            t = pi.rand(1, 3, 32, 32)
            l = ops.max_pool2d(t, 3, (2, 2))
            check_correct(
                " %4 = torch.aten.max_pool2d %arg0, %1, %2, %3, %0, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor",
                l.owner,
            )

            t = pi.rand(1, 3, 32, 32)
            l = ops.max_pool2d(t, 3, 1, 2, 3)
            check_correct(
                " %4 = torch.aten.max_pool2d %arg0, %1, %2, %3, %0, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor",
                l.owner,
            )

            t = pi.rand(1, 3, 32, 32)
            l = ops.max_pool2d(t, 3, 1, [2, 2], 3)
            check_correct(
                " %4 = torch.aten.max_pool2d %arg0, %1, %2, %3, %0, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor",
                l.owner,
            )

    def test_max_pool2d_with_indices(self):
        with mlir_mod_ctx():
            t = pi.rand(1, 3, 32, 32)
            l, s = ops.max_pool2d_with_indices(t, 3)
            check_correct(
                "%result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %1, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor, !torch.tensor",
                l.owner,
            )

            t = pi.rand(1, 3, 32, 32)
            l, s = ops.max_pool2d_with_indices(t, 3, (1, 1), 2)
            check_correct(
                "%result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %1, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor, !torch.tensor",
                l.owner,
            )

            t = pi.rand(1, 3, 32, 32)
            l, s = ops.max_pool2d_with_indices(t, 3, (1, 1), 2, [4, 4])
            check_correct(
                "%result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %1, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor, !torch.tensor",
                l.owner,
            )

            t = pi.rand(1, 3, 32, 32)
            l, s = ops.max_pool2d_with_indices(t, 3, 1, 2, 4)
            check_correct(
                "%result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %1, %false : !torch.tensor<[1,3,32,32],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.tensor, !torch.tensor",
                l.owner,
            )
