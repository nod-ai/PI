import numpy as np

from pi import ops
from pi.mlir import (
    torch_dialect as torch,
)
from pi.mlir.utils import mlir_mod_ctx
from util import check_correct


class TestOverloadCast:
    def test_simple_values(self):
        with mlir_mod_ctx():
            one_int = torch.ConstantIntOp(1)
            two_int = torch.ConstantIntOp(2)
            res = ops.add(one_int, two_int)
            check_correct(
                "Torch_IntValue(%0 = torch.aten.add.int %int1, %int2 : !torch.int, !torch.int -> !torch.int)",
                res,
            )

            one_float = torch.ConstantFloatOp(1.0)
            res = ops.add(one_float, two_int)
            check_correct(
                "Torch_FloatValue(%1 = torch.aten.add.float_int %float1.000000e00, %int2 : !torch.float, !torch.int -> !torch.float)",
                res,
            )

    def test_non_infer_returns(self):
        with mlir_mod_ctx():
            one_int = torch.ConstantIntOp(1)
            two_int = torch.ConstantIntOp(2)
            res = ops.eq(one_int, two_int)
            check_correct(
                "Torch_BoolValue(%0 = torch.aten.eq.int %int1, %int2 : !torch.int, !torch.int -> !torch.bool)",
                res,
            )

            one_float = torch.ConstantFloatOp(1)
            two_float = torch.ConstantFloatOp(2)
            res = ops.eq(one_float, two_float)
            check_correct(
                "Torch_BoolValue(%0 = torch.aten.eq.float %1.000000e00, %2.000000e00 : !torch.float, !torch.float -> !torch.bool)",
                res,
            )

            t = torch.NonValueTensorLiteralOp(np.random.randint(0, 10, (10, 10)))
            check_correct(
                "Tensor(%2 = torch.tensor.literal(dense<> : tensor<10x10xsi64>) : !torch.tensor<[10,10],si64>)",
                t,
            )
            t = ops.eq(t, t)
            check_correct(
                "Tensor(%3 = torch.aten.eq.Tensor %2, %2 : !torch.tensor<[10,10],si64>, !torch.tensor<[10,10],si64> -> !torch.tensor)",
                t,
            )
