import numpy as np
from textwrap import dedent

from pi.mlir.utils import mlir_mod_ctx
from pi.mlir import (
    # AnyTorchDictKeyValue,
    # AnyTorchListOfOptionalIntValue,
    # AnyTorchListOfOptionalTensorValue,
    AnyTorchListOfTorchBoolValue,
    AnyTorchListOfTorchIntValue,
    AnyTorchListOfTorchStringValue,
    AnyTorchListValue,
    AnyTorchListOfTensorValue,
    AnyTorchListOfTensorType,
    AnyTorchListType,
    AnyTorchOptionalBoolValue,
    AnyTorchOptionalDeviceValue,
    AnyTorchOptionalFloatValue,
    AnyTorchOptionalGeneratorValue,
    AnyTorchOptionalIntValue,
    AnyTorchOptionalStringValue,
    AnyTorchOptionalTensorValue,
    AnyTorchOptionalTensorType,
    AnyTorchOptionalValue,
    # AnyTorchOptionalListOfTorchIntValue,
    # AnyTorchTensorValue,
    AnyTorchScalarValue,
    Torch_BoolType,
    Torch_DeviceType,
    Torch_DictType,
    Torch_FloatType,
    Torch_IntType,
    Torch_BoolValue,
    Torch_DeviceValue,
    Torch_DictValue,
    Torch_FloatValue,
    Torch_IntValue,
    Torch_LinearParamsValue,
    Torch_NnModuleValue,
    Torch_NonValueTensorValue,
    Torch_NoneValue,
    Torch_NumberValue,
    Torch_StringValue,
    Torch_TupleValue,
    Torch_TupleType,
    Torch_ValueTensorValue,
    Torch_NonValueTensorValue,
    torch_dialect as torch,
    _fp64ElementsAttr,
)
from pi.mlir import F32, F64, ops
from util import check_correct


class TestTorchValues:
    def test_smoke(self):
        src = dedent(
            """\
        module {
          func.func @method() {
            %true = torch.constant.bool true
            return
          }
        }
        """
        )
        with mlir_mod_ctx(src) as module:
            pass

        correct = dedent(
            """\
        module {
          func.func @method() {
            %true = torch.constant.bool true
            return
          }
        }
        """
        )
        check_correct(correct, module)

    def test_simple_values(self):
        with mlir_mod_ctx():
            tfloat = torch.ConstantFloatOp(1.0)
            check_correct(
                str(tfloat.owner),
                "%float1.000000e00 = torch.constant.float 1.000000e+00",
            )
            assert Torch_FloatValue.isinstance(tfloat)
            check_correct(
                str(tfloat),
                "Torch_FloatValue(%float1.000000e00 = torch.constant.float 1.000000e+00)",
            )

            tint = torch.ConstantIntOp(1)
            check_correct(str(tint.owner), "%int1 = torch.constant.int 1")
            assert Torch_IntValue.isinstance(tint)
            check_correct(str(tint), "Torch_IntValue(%int1 = torch.constant.int 1)")

            tbool = torch.ConstantBoolOp(True)
            check_correct(str(tbool.owner), "%true = torch.constant.bool true")
            assert Torch_BoolValue.isinstance(tbool)
            check_correct(
                str(tbool), "Torch_BoolValue(%true = torch.constant.bool true)"
            )

            tdevice = torch.ConstantDeviceOp("cuda")
            check_correct(str(tdevice.owner), '%cuda = torch.constant.device "cuda"')
            assert Torch_DeviceValue.isinstance(tdevice)
            check_correct(
                str(tdevice), 'Torch_DeviceValue(%cuda = torch.constant.device "cuda")'
            )

            tnone = torch.ConstantNoneOp()
            check_correct(str(tnone.owner), "%none = torch.constant.none")
            assert Torch_NoneValue.isinstance(tnone)
            check_correct(str(tnone), "Torch_NoneValue(%none = torch.constant.none)")

    def test_agg_values(self):
        with mlir_mod_ctx():
            tint = Torch_IntType.get()
            tfloat = Torch_FloatType.get()
            tbool = Torch_BoolType.get()

            tintv = torch.ConstantIntOp(1)
            tfloatv = torch.ConstantFloatOp(1.0)
            tboolv = torch.ConstantBoolOp(True)

            t = Torch_TupleType.get((tint, tfloat, tbool))
            tup = torch.PrimTupleConstructOp(t, (tintv, tfloatv, tboolv))
            check_correct(
                str(tup),
                "Torch_TupleValue(%0 = torch.prim.TupleConstruct %int1, %float1.000000e00, %true : !torch.int, !torch.float, !torch.bool -> !torch.tuple<int, float, bool>)",
            )

            t = AnyTorchListType.get(tint)
            lis = torch.PrimListConstructOp(t, (tintv, tintv, tintv))
            check_correct(
                str(lis),
                "AnyTorchListValue(%1 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>)",
            )

            l = ops.len(lis)
            check_correct(
                str(l),
                "Torch_IntValue(%2 = torch.aten.len.t %1 : !torch.list<int> -> !torch.int)",
            )

    def test_tensor_values(self):
        with mlir_mod_ctx():
            t = torch.NonValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
            check_correct(
                str(t),
                "Tensor(%0 = torch.tensor.literal(dense<1.000000e+00> : tensor<2x2xf64>) : !torch.tensor<[2,2],f64>)",
            )

            t = torch.ValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
            check_correct(
                str(t),
                "Tensor(%1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<2x2xf64>) : !torch.vtensor<[2,2],f64>)",
            )

            tintv = torch.ConstantIntOp(1)
            res = t.add_(t, tintv)
            check_correct(
                str(res),
                "Tensor(%2 = torch.aten.add_.Tensor %1, %1, %int1 : !torch.vtensor<[2,2],f64>, !torch.vtensor<[2,2],f64>, !torch.int -> !torch.tensor)",
            )

            tfloatv = torch.ConstantFloatOp(1)
            res = t.add_(t, tfloatv)
            check_correct(
                str(res),
                "Tensor(%3 = torch.aten.add_.Tensor %1, %1, %float1.000000e00 : !torch.vtensor<[2,2],f64>, !torch.vtensor<[2,2],f64>, !torch.float -> !torch.tensor)",
            )

            tintv = torch.ConstantIntOp(1)
            t = torch.ValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
            lis_t = AnyTorchListOfTensorType.get(t.type)
            lis = torch.PrimListConstructOp(lis_t, (t, t))
            tcat = ops.cat(lis, tintv)
            check_correct(
                str(tcat),
                "Tensor(%6 = torch.aten.cat %5, %int1_0 : !torch.list<vtensor<[2,2],f64>>, !torch.int -> !torch.tensor)",
            )

            t = torch.ValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
            opt_t = torch.ConstantNoneOp()
            clamped_t = ops.clamp(t, opt_t, opt_t)
            check_correct(
                str(clamped_t),
                "Tensor(%8 = torch.aten.clamp.Tensor %7, %none, %none : !torch.vtensor<[2,2],f64>, !torch.none, !torch.none -> !torch.tensor)",
            )

    def test_scalar_type(self):
        with mlir_mod_ctx():
            tintv = torch.ConstantIntOp(1)
            t = ops.NumToTensor(AnyTorchScalarValue(tintv))
            check_correct(
                str(t),
                "Tensor(%0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.tensor)",
            )

            t = ops.NumToTensor(tintv)
            check_correct(
                str(t),
                "Tensor(%0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.tensor)",
            )

            try:
                t = torch.NonValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
                t = ops.NumToTensor(t)
            except TypeError as e:
                msg = " ".join(list(map(lambda x: x.strip(), e.args[0].splitlines())))
                assert msg.strip().startswith(
                    "NumToTensor(): incompatible function arguments. The following argument types are supported: 1. (arg0: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue) -> object  Invoked with: <pi.mlir._mlir_libs._pi_mlir.Tensor"
                )

            t = ops.Float(tintv)
            check_correct(
                str(t),
                "Torch_FloatValue(%3 = torch.aten.Float.Scalar %int1 : !torch.int -> !torch.float)",
            )

            t1 = torch.NonValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
            t2 = torch.NonValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
            t = ops.Float(tintv)
            t3 = ops.add(t1, t2, t)
            check_correct(
                str(t3),
                "Tensor(%7 = torch.aten.add.Tensor %4, %5, %6 : !torch.tensor<[2,2],f64>, !torch.tensor<[2,2],f64>, !torch.float -> !torch.tensor)",
            )

            t3 = ops.add(t1, t, t)
            check_correct(
                str(t3),
                "Tensor(%8 = torch.aten.add.Scalar %4, %6, %6 : !torch.tensor<[2,2],f64>, !torch.float, !torch.float -> !torch.tensor)",
            )

            try:
                t3 = ops.add(t, t, t)
                check_correct(
                    str(t3),
                    "Tensor(%8 = torch.aten.add.Scalar %4, %6, %6 : !torch.tensor<[2,2],f64>, !torch.float, !torch.float -> !torch.tensor)",
                )
            except TypeError as e:
                check_correct(
                    dedent(
                        """
                        add(): incompatible function arguments. The following argument types are supported:
                            1. (arg0: pi.mlir._mlir_libs._pi_mlir.Tensor, arg1: pi.mlir._mlir_libs._pi_mlir.Tensor, arg2: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue) -> object
                            2. (arg0: pi.mlir._mlir_libs._pi_mlir.Tensor, arg1: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue, arg2: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue) -> object
                            3. (arg0: pi.mlir._mlir_libs._pi_mlir.Torch_StringValue, arg1: pi.mlir._mlir_libs._pi_mlir.Torch_StringValue) -> object
                            4. (arg0: pi.mlir._mlir_libs._pi_mlir.Torch_IntValue, arg1: pi.mlir._mlir_libs._pi_mlir.Torch_IntValue) -> object
                            5. (arg0: pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue, arg1: pi.mlir._mlir_libs._pi_mlir.Torch_IntValue) -> object
                            6. (arg0: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue, arg1: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue) -> object
                            
                        Invoked with: <pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue object at 0x7f5a97e94130>, <pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue object at 0x7f5a97e94130>, <pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue object at 0x7f5a97e94130>
                        """
                    ),
                    str(e),
                )

            t = ops.abs(tintv)
            check_correct(
                str(t),
                "Torch_IntValue(%9 = torch.prim.abs.Scalar %int1 : !torch.int -> !torch.int)",
            )

            t = ops.ceil(tintv)
            check_correct(
                str(t),
                "Torch_IntValue(%9 = torch.aten.ceil.Scalar %int1 : !torch.int -> !torch.int)",
            )

            tintv = torch.ConstantIntOp(1)
            res = ops.add(tintv, tintv)
            check_correct(
                str(res),
                "Torch_IntValue(%11 = torch.aten.add.int %int1_0, %int1_0 : !torch.int, !torch.int -> !torch.int)",
            )
            tfloatv = torch.ConstantFloatOp(1)
            res = ops.add(tfloatv, tfloatv)
            check_correct(
                str(res),
                "Torch_FloatValue(%12 = torch.aten.add %float1.000000e00, %float1.000000e00 : !torch.float, !torch.float -> !torch.float)",
            )

            try:
                res = ops.add(tintv, tfloatv)
            except NotImplementedError as e:
                check_correct(
                    str(e),
                    "Arithmetic ops on Scalar values with like types supported; type a: AnyTorchScalarValue(%int1_0 = torch.constant.int 1), type b: AnyTorchScalarValue(%float1.000000e00 = torch.constant.float 1.000000e+00)",
                )
