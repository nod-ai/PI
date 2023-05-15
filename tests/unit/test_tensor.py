import numpy as np

from pi.mlir.utils import mlir_mod_ctx
from pi.mlir import (
    ops,
    Tensor,
    torch_dialect as torch,
)
from util import check_correct


class TestOverloadCast:
    def test_non_infer_returns(self):
        with mlir_mod_ctx():
            t = torch.NonValueTensorLiteralOp(np.random.randint(0, 10, (10, 10)))
            check_correct(
                "Tensor(%2 = torch.tensor.literal(dense<> : tensor<10x10xf64>) : !torch.tensor<[10,10],f64>)",
                t,
            )
            t = ops.eq(t, t)
            check_correct(
                "Tensor(%3 = torch.aten.eq.Tensor %2, %2 : !torch.tensor<[10,10],f64>, !torch.tensor<[10,10],f64> -> !torch.tensor)",
                t,
            )

            tt = t == t
            check_correct(
                "Tensor(%2 = torch.aten.eq.Tensor %1, %1 : !torch.tensor, !torch.tensor -> !torch.tensor)",
                tt,
            )

            zero_int = torch.ConstantIntOp(0)
            one_int = torch.ConstantIntOp(1)
            tt = t.transpose(zero_int, one_int)
            check_correct(
                "Tensor(%3 = torch.aten.transpose.int %1, %int0, %int1 : !torch.tensor, !torch.int, !torch.int -> !torch.tensor)",
                tt,
            )

            ttt = tt * tt
            check_correct(
                "Tensor(%4 = torch.aten.mul.Tensor %3, %3 : !torch.tensor, !torch.tensor -> !torch.tensor)",
                ttt,
            )

            try:
                tttt = ttt % 3
            except NotImplementedError as e:
                assert (
                    str(e)
                    == "__mod__ with signature __mod__(self, other Any) -> Tensor"
                )
