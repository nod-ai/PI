from textwrap import dedent

import numpy as np

from pi import ops
from pi.mlir import (
    torch_dialect as torch,
)
from pi.mlir.utils import mlir_mod_ctx
from util import check_correct


class TestOverloadCast:
    def test_non_infer_returns(self):
        with mlir_mod_ctx():
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

            tt = t == t
            check_correct(
                "Tensor(%2 = torch.aten.eq.Tensor %1, %1 : !torch.tensor, !torch.tensor -> !torch.tensor)",
                tt,
            )

            tt = t + t
            check_correct(
                "%3 = torch.aten.add.Tensor %1, %1, %float1.000000e00 : !torch.tensor, !torch.tensor, !torch.float -> !torch.tensor",
                tt.owner,
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
                    == "NotImplementedError: __mod__ with signature __mod__(self, other: Any) -> Tensor"
                )

    def test_optional_args(self):
        with mlir_mod_ctx():
            ttt = torch.NonValueTensorLiteralOp(np.random.randint(0, 10, (10, 10)))
            r = ttt.argmax(keepdim=torch.ConstantBoolOp(False))
            check_correct(
                "Tensor(%5 = torch.aten.argmax %4, %none, %false : !torch.tensor<[10,10],si64>, !torch.none, !torch.bool -> !torch.tensor)",
                r,
            )
            try:
                r = ttt.argmax(torch.ConstantBoolOp(False))
            except TypeError as e:
                check_correct(
                    dedent(
                        """\
                        argmax(): incompatible function arguments. The following argument types are supported:
                            1. (self: pi.mlir._mlir_libs._pi_mlir.Tensor, dim: pi.mlir._mlir_libs._pi_mlir.AnyTorchOptionalIntValue = None, keepdim: pi.mlir._mlir_libs._pi_mlir.Torch_BoolValue = False) -> pi.mlir._mlir_libs._pi_mlir.Tensor

                        Invoked with: <pi.mlir._mlir_libs._pi_mlir.Tensor object at %DONT_CARE>, <pi.mlir._mlir_libs._pi_mlir.Torch_BoolValue object at %DONT_CARE>
                        """
                    ),
                    str(e),
                )

            # keepdim is a kwonly arg
            try:
                r = ttt.argmax(None, torch.ConstantBoolOp(False))
            except TypeError as e:
                check_correct(
                    dedent(
                        """\
                        argmax(): incompatible function arguments. The following argument types are supported:
                            1. (self: pi.mlir._mlir_libs._pi_mlir.Tensor, dim: pi.mlir._mlir_libs._pi_mlir.AnyTorchOptionalIntValue = None, *, keepdim: pi.mlir._mlir_libs._pi_mlir.Torch_BoolValue = False) -> object

                        Invoked with: <pi.mlir._mlir_libs._pi_mlir.Tensor object at %DONT_CARE>, None, <pi.mlir._mlir_libs._pi_mlir.Torch_BoolValue object at %DONT_CARE>
                        """
                    ),
                    str(e),
                )

            zero_int = torch.ConstantIntOp(0)
            r = ops.gather(ttt, zero_int, ttt, False)
            check_correct(
                "Tensor(%2 = torch.aten.gather %0, %int0, %0, %false_3 : !torch.tensor<[10,10],si64>, !torch.int, !torch.tensor<[10,10],si64>, !torch.bool -> !torch.tensor)",
                r,
            )

            r = ops.gather(ttt, zero_int, ttt)
            check_correct(
                "Tensor(%2 = torch.aten.gather %0, %int0, %0, %false_3 : !torch.tensor<[10,10],si64>, !torch.int, !torch.tensor<[10,10],si64>, !torch.bool -> !torch.tensor)",
                r,
            )
