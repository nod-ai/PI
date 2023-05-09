import numpy as np
from textwrap import dedent

from pi.mlir.utils import mlir_mod_ctx
from pi.mlir import (
    TorchAnyValue,
    TorchBoolValue,
    TorchDeviceValue,
    TorchDictValue,
    TorchFloatValue,
    TorchGeneratorValue,
    TorchIntValue,
    TorchLinearParamsValue,
    TorchListValue,
    TorchNnModuleValue,
    TorchNonValueTensorValue,
    TorchNoneValue,
    TorchNumberValue,
    TorchOptionalValue,
    TorchQInt8Value,
    TorchQUInt8Value,
    TorchStringValue,
    TorchTupleValue,
    TorchUnionValue,
    TorchValueTensorValue,
    TorchIntType,
    TorchBoolType,
    TorchFloatType,
    TorchTupleType,
    TorchListType,
    TorchNonValueTensorType,
    TorchValueTensorType,
    torch_dialect as torch,
    _fp64ElementsAttr,
)
from pi.mlir import F32, F64
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
            tfloat = torch.ConstantFloatOp(1.0).result
            assert (
                str(tfloat.owner)
                == "%float1.000000e00 = torch.constant.float 1.000000e+00"
            )
            t = TorchFloatValue(tfloat)
            assert TorchFloatValue.isinstance(t)
            assert (
                str(t)
                == "TorchFloatValue(%float1.000000e00 = torch.constant.float 1.000000e+00)"
            )

            tint = torch.ConstantIntOp(1).result
            assert str(tint.owner) == "%int1 = torch.constant.int 1"
            t = TorchIntValue(tint)
            assert TorchIntValue.isinstance(t)
            assert str(t) == "TorchIntValue(%int1 = torch.constant.int 1)"

            tbool = torch.ConstantBoolOp(True).result
            assert str(tbool.owner) == "%true = torch.constant.bool true"
            t = TorchBoolValue(tbool)
            assert TorchBoolValue.isinstance(t)
            assert str(t) == "TorchBoolValue(%true = torch.constant.bool true)"

            tdevice = torch.ConstantDeviceOp("cuda").result
            assert str(tdevice.owner) == '%cuda = torch.constant.device "cuda"'
            t = TorchDeviceValue(tdevice)
            assert TorchDeviceValue.isinstance(t)
            assert str(t) == 'TorchDeviceValue(%cuda = torch.constant.device "cuda")'

            tnone = torch.ConstantNoneOp().result
            assert str(tnone.owner) == "%none = torch.constant.none"
            t = TorchNoneValue(tnone)
            assert TorchNoneValue.isinstance(t)
            assert str(t) == "TorchNoneValue(%none = torch.constant.none)"

    def test_agg_values(self):
        with mlir_mod_ctx():
            tint = TorchIntType.get()
            tfloat = TorchFloatType.get()
            tbool = TorchBoolType.get()

            tintv = torch.ConstantIntOp(1).result
            tfloatv = torch.ConstantFloatOp(1.0).result
            tboolv = torch.ConstantBoolOp(True).result

            t = TorchTupleType.get((tint, tfloat, tbool))
            tup = TorchTupleValue(
                torch.PrimTupleConstructOp(t, (tintv, tfloatv, tboolv)).result
            )
            assert (
                str(tup)
                == "TorchTupleValue(%0 = torch.prim.TupleConstruct %int1, %float1.000000e00, %true : !torch.int, !torch.float, !torch.bool -> !torch.tuple<int, float, bool>)"
            )

            t = TorchListType.get(tint)
            lis = TorchListValue(
                torch.PrimListConstructOp(t, (tintv, tintv, tintv)).result
            )
            assert (
                str(lis)
                == "TorchListValue(%1 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>)"
            )

    def test_tensor_values(self):
        with mlir_mod_ctx():
            t = TorchNonValueTensorValue(
                torch.NonValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2)))).result
            )
            assert (
                str(t)
                == "TorchNonValueTensorValue(%0 = torch.tensor.literal(dense<1.000000e+00> : tensor<2x2xf64>) : !torch.tensor<[2,2],f64>)"
            )

            t = TorchValueTensorValue(
                torch.ValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2)))).result
            )
            assert (
                str(t)
                == "TorchValueTensorValue(%1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<2x2xf64>) : !torch.vtensor<[2,2],f64>)"
            )
