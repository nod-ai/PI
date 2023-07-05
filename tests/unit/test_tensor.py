from textwrap import dedent

import numpy as np

import pi
from pi import ops
from pi.utils import mlir_mod_ctx, int_op, non_value_tensor_op, bool_op, tensor_op
from util import check_correct


class TestOverloadCast:
    def test_non_infer_returns(self):
        with mlir_mod_ctx():
            t = non_value_tensor_op(np.random.randint(0, 10, (10, 10)))
            check_correct(
                "Torch_NonValueTensorValue(%2 = torch.tensor.literal(dense<> : tensor<10x10xsi64>) : !torch.tensor<[10,10],si64>)",
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

            zero_int = int_op(0)
            one_int = int_op(1)
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
            ttt = tensor_op(5 * np.ones((10, 10)).astype(int))
            r = ttt.argmax(keepdim=bool_op(False))
            check_correct(
                "Tensor(%5 = torch.aten.argmax %4, %none, %false : !torch.tensor<[10,10],si64>, !torch.none, !torch.bool -> !torch.tensor)",
                r,
            )
            try:
                r = ttt.argmax(bool_op(False))
            except TypeError as e:
                check_correct(
                    dedent(
                        """\
                        argmax(): incompatible function arguments. The following argument types are supported:
                            1. (self: pi.mlir._mlir_libs._pi_mlir.Tensor, dim: pi.mlir._mlir_libs._pi_mlir.AnyTorchOptionalIntValue = None, keepdim: pi.mlir._mlir_libs._pi_mlir.Torch_BoolValue = False, *, loc: mlir.ir.Location = None, ip: mlir.ir.InsertionPoint = None) -> pi.mlir._mlir_libs._pi_mlir.Tensor

                        Invoked with: Tensor(%0 = torch.tensor.literal(dense<5> : tensor<10x10xsi64>) : !torch.tensor<[10,10],si64>), Torch_BoolValue(%false_0 = torch.constant.bool false)
                        """
                    ),
                    str(e),
                )

            # keepdim is a kwonly arg
            try:
                r = ttt.argmax(None, bool_op(False))
            except TypeError as e:
                check_correct(
                    dedent(
                        """\
                        argmax(): incompatible function arguments. The following argument types are supported:
                            1. (self: pi.mlir._mlir_libs._pi_mlir.Tensor, dim: pi.mlir._mlir_libs._pi_mlir.AnyTorchOptionalIntValue = None, *, keepdim: pi.mlir._mlir_libs._pi_mlir.Torch_BoolValue = False) -> object

                        Invoked with: Tensor(%DONT_CARE = torch.tensor.literal(%DONT_CARE : tensor<1%DONT_CARE>) : !torch.tensor<[3,3],si64>), Torch_BoolValue(%DONT_CARE = torch.constant.bool false)
                        """
                    ),
                    str(e),
                )

            zero_int = int_op(0)
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

    def test_tensor_div_overload(self):
        with mlir_mod_ctx():
            x = pi.ones(3)
            y = pi.ones(3)
            z = 2

            d = x / z
            check_correct(
                "Tensor(%2 = torch.aten.div.Scalar %0, %int2 : !torch.tensor<[3],f64>, !torch.int -> !torch.tensor)",
                d,
            )

            d = x / y
            check_correct(
                "Tensor(%3 = torch.aten.div.Tensor %0, %1 : !torch.tensor<[3],f64>, !torch.tensor<[3],f64> -> !torch.tensor)",
                d,
            )

            # __rtruediv__ is computed via a reciprocal(tensor) and a scalar multiplication operator
            d = z / x
            check_correct(
                "Tensor(%5 = torch.aten.mul.Scalar %4, %int2 : !torch.tensor, !torch.int -> !torch.tensor)",
                d,
            )

    def test_tensor_indexing(self):
        with mlir_mod_ctx():
            x = pi.ones((4, 4))

            v = x[0]
            check_correct(
                "Tensor(%1 = torch.aten.select.int %0, %int0_0, %int0 : !torch.tensor<[4,4],f64>, !torch.int, !torch.int -> !torch.tensor)",
                v,
            )

            v = x[None]
            check_correct(
                "Tensor(%3 = torch.aten.unsqueeze %0, %int0_1 : !torch.tensor<[4,4],f64>, !torch.int -> !torch.tensor)",
                v,
            )

            v = x[1:3:2]
            check_correct(
                "Tensor(%4 = torch.aten.slice.Tensor %0, %int0_4, %int1_2, %int3, %int2_3 : !torch.tensor<[4,4],f64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tensor)",
                v,
            )

            v = x[:]
            check_correct(
                "Tensor(%5 = torch.aten.slice.Tensor %0, %int0_7, %none, %none_5, %int1_6 : !torch.tensor<[4,4],f64>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.tensor)",
                v,
            )
