from textwrap import dedent

from pi.mlir.utils import mlir_mod_ctx
from pi.mlir import (
    TorchNnModuleType,
    TorchDeviceType,
    TorchGeneratorType,
    TorchBoolType,
    TorchIntType,
    TorchFloatType,
    TorchLinearParamsType,
    TorchQInt8Type,
    TorchQUInt8Type,
    TorchNoneType,
    TorchStringType,
    TorchAnyType,
    TorchNumberType,
    TorchOptionalType,
    TorchTupleType,
    TorchUnionType,
    TorchListType,
    TorchNonValueTensorType,
    TorchValueTensorType,
    TorchDictType,
)
from pi.mlir import F32, F64
from util import check_correct


class TestTorchTypes:
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

    def test_simple_types(self):
        with mlir_mod_ctx():
            t = TorchDeviceType.get()
            assert str(t) == "!torch.Device"
            assert TorchDeviceType.isinstance(t)

            t = TorchGeneratorType.get()
            assert str(t) == "!torch.Generator"
            assert TorchGeneratorType.isinstance(t)

            t = TorchBoolType.get()
            assert str(t) == "!torch.bool"
            assert TorchBoolType.isinstance(t)

            t = TorchIntType.get()
            assert str(t) == "!torch.int"
            assert TorchIntType.isinstance(t)

            t = TorchFloatType.get()
            assert str(t) == "!torch.float"
            assert TorchFloatType.isinstance(t)

            t = TorchLinearParamsType.get()
            assert str(t) == "!torch.LinearParams"
            assert TorchLinearParamsType.isinstance(t)

            t = TorchQInt8Type.get()
            assert str(t) == "!torch.qint8"
            assert TorchQInt8Type.isinstance(t)

            t = TorchQUInt8Type.get()
            assert str(t) == "!torch.quint8"
            assert TorchQUInt8Type.isinstance(t)

            t = TorchNoneType.get()
            assert str(t) == "!torch.none"
            assert TorchNoneType.isinstance(t)

            t = TorchStringType.get()
            assert str(t) == "!torch.str"
            assert TorchStringType.isinstance(t)

            t = TorchAnyType.get()
            assert str(t) == "!torch.any"
            assert TorchAnyType.isinstance(t)

            t = TorchNumberType.get()
            assert str(t) == "!torch.number"
            assert TorchNumberType.isinstance(t)

    def test_agg_types(self):
        with mlir_mod_ctx():
            t = TorchNnModuleType.get("bob")
            assert str(t) == '!torch.nn.Module<"bob">'
            assert TorchNnModuleType.isinstance(t)

            tint = TorchIntType.get()
            tfloat = TorchFloatType.get()
            tbool = TorchBoolType.get()

            t = TorchOptionalType.get(tint)
            assert str(t) == "!torch.optional<int>"
            assert TorchOptionalType.isinstance(t)

            t = TorchTupleType.get((tint, tfloat, tbool))
            assert str(t) == "!torch.tuple<int, float, bool>"
            assert TorchTupleType.isinstance(t)
            assert len(t) == 3
            assert TorchIntType.isinstance(t[0])
            assert TorchFloatType.isinstance(t[1])
            assert TorchBoolType.isinstance(t[2])

            t = TorchUnionType.get((tint, tfloat, tbool))
            assert TorchUnionType.isinstance(t)
            assert str(t) == "!torch.union<int, float, bool>"
            assert len(t) == 3
            assert TorchIntType.isinstance(t[0])
            assert TorchFloatType.isinstance(t[1])
            assert TorchBoolType.isinstance(t[2])

            t = TorchListType.get(tint)
            assert TorchListType.isinstance(t)
            assert str(t) == "!torch.list<int>"

            d = TorchDictType.get(tint, tfloat)
            assert TorchDictType.isinstance(d)
            assert str(d) == "!torch.dict<int, float>"
            assert TorchIntType.isinstance(d.get_key_type())
            assert TorchFloatType.isinstance(d.get_value_type())

    def test_tensor_types(self):
        with mlir_mod_ctx():
            t = TorchNonValueTensorType.get([1, 2, 3], F32)
            assert TorchNonValueTensorType.isinstance(t)
            assert str(t) == "!torch.tensor<[1,2,3],f32>"
            assert t.sizes() == (1, 2, 3)
            assert F32.isinstance(t.dtype())

            t = TorchNonValueTensorType.get([], F32)
            assert str(t) == "!torch.tensor<[],f32>"
            assert t.sizes() == ()

            t = TorchNonValueTensorType.get([-1], F32)
            assert str(t) == "!torch.tensor<[?],f32>"
            assert t.sizes() == (-1,)
            assert F32.isinstance(t.dtype())

            t = TorchNonValueTensorType.get([-1, -1], F32)
            assert str(t) == "!torch.tensor<[?,?],f32>"
            assert t.sizes() == (-1, -1)
            assert F32.isinstance(t.dtype())

            t = TorchValueTensorType.get([1, 2, 3], F32)
            assert TorchValueTensorType.isinstance(t)
            assert str(t) == "!torch.vtensor<[1,2,3],f32>"
            assert t.sizes() == (1, 2, 3)
            assert F32.isinstance(t.dtype())

            t = TorchValueTensorType.get([], F32)
            assert str(t) == "!torch.vtensor<[],f32>"
            assert t.sizes() == ()

            t = TorchValueTensorType.get([-1], F32)
            assert str(t) == "!torch.vtensor<[?],f32>"
            assert t.sizes() == (-1,)
            assert F32.isinstance(t.dtype())

            t = TorchValueTensorType.get([-1, -1], F32)
            assert str(t) == "!torch.vtensor<[?,?],f32>"
            assert t.sizes() == (-1, -1)
            assert F32.isinstance(t.dtype())
