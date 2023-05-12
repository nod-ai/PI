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
            tfloat = torch.ConstantFloatOp(1.0)
            assert (
                str(tfloat.owner)
                == "%float1.000000e00 = torch.constant.float 1.000000e+00"
            )
            assert TorchFloatValue.isinstance(tfloat)
            assert (
                str(tfloat)
                == "TorchFloatValue(%float1.000000e00 = torch.constant.float 1.000000e+00)"
            )

            tint = torch.ConstantIntOp(1)
            assert str(tint.owner) == "%int1 = torch.constant.int 1"
            assert TorchIntValue.isinstance(tint)
            assert str(tint) == "TorchIntValue(%int1 = torch.constant.int 1)"

            tbool = torch.ConstantBoolOp(True)
            assert str(tbool.owner) == "%true = torch.constant.bool true"
            assert TorchBoolValue.isinstance(tbool)
            assert str(tbool) == "TorchBoolValue(%true = torch.constant.bool true)"

            tdevice = torch.ConstantDeviceOp("cuda")
            assert str(tdevice.owner) == '%cuda = torch.constant.device "cuda"'
            assert TorchDeviceValue.isinstance(tdevice)
            assert (
                str(tdevice) == 'TorchDeviceValue(%cuda = torch.constant.device "cuda")'
            )

            tnone = torch.ConstantNoneOp()
            assert str(tnone.owner) == "%none = torch.constant.none"
            assert TorchNoneValue.isinstance(tnone)
            assert str(tnone) == "TorchNoneValue(%none = torch.constant.none)"

    def test_agg_values(self):
        with mlir_mod_ctx():
            tint = TorchIntType.get()
            tfloat = TorchFloatType.get()
            tbool = TorchBoolType.get()

            tintv = torch.ConstantIntOp(1)
            tfloatv = torch.ConstantFloatOp(1.0)
            tboolv = torch.ConstantBoolOp(True)

            t = TorchTupleType.get((tint, tfloat, tbool))
            tup = torch.PrimTupleConstructOp(t, (tintv, tfloatv, tboolv))
            assert (
                str(tup)
                == "TorchTupleValue(%0 = torch.prim.TupleConstruct %int1, %float1.000000e00, %true : !torch.int, !torch.float, !torch.bool -> !torch.tuple<int, float, bool>)"
            )

            t = TorchListType.get(tint)
            lis = torch.PrimListConstructOp(t, (tintv, tintv, tintv))
            assert (
                str(lis)
                == "TorchListValue(%1 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>)"
            )

    def test_tensor_values(self):
        with mlir_mod_ctx():
            t = torch.NonValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
            assert (
                str(t)
                == "TorchNonValueTensorValue(%0 = torch.tensor.literal(dense<1.000000e+00> : tensor<2x2xf64>) : !torch.tensor<[2,2],f64>)"
            )

            t = torch.ValueTensorLiteralOp(_fp64ElementsAttr(np.ones((2, 2))))
            assert (
                str(t)
                == "TorchValueTensorValue(%1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<2x2xf64>) : !torch.vtensor<[2,2],f64>)"
            )
