from textwrap import dedent

from pi.mlir.utils import mlir_mod_ctx
from pi.mlir import (
    # AnyTorchDictKeyType,
    # AnyTorchListOfOptionalIntType,
    # AnyTorchListOfOptionalTensorType,
    # AnyTorchListOfTensorType,
    AnyTorchListOfTorchBoolType,
    AnyTorchListOfTorchIntType,
    AnyTorchListOfTorchStringType,
    AnyTorchListType,
    AnyTorchOptionalBoolType,
    AnyTorchOptionalDeviceType,
    AnyTorchOptionalFloatType,
    AnyTorchOptionalGeneratorType,
    AnyTorchOptionalIntType,
    AnyTorchOptionalStringType,
    # AnyTorchOptionalTensorType,

    AnyTorchOptionalType,

    # AnyTorchOptionalListOfTorchIntType,
    # AnyTorchTensorType,
    Torch_BoolType,
    Torch_DeviceType,
    Torch_DictType,
    Torch_FloatType,
    Torch_IntType,
    Torch_LinearParamsType,
    Torch_NnModuleType,
    Torch_NonValueTensorType,
    Torch_NoneType,
    Torch_NumberType,
    Torch_StringType,
    Torch_TupleType,
    Torch_ValueTensorType,
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
            t = Torch_DeviceType.get()
            assert str(t) == "!torch.Device"
            assert Torch_DeviceType.isinstance(t)

            t = Torch_BoolType.get()
            assert str(t) == "!torch.bool"
            assert Torch_BoolType.isinstance(t)

            t = Torch_IntType.get()
            assert str(t) == "!torch.int"
            assert Torch_IntType.isinstance(t)

            t = Torch_FloatType.get()
            assert str(t) == "!torch.float"
            assert Torch_FloatType.isinstance(t)

            t = Torch_LinearParamsType.get()
            assert str(t) == "!torch.LinearParams"
            assert Torch_LinearParamsType.isinstance(t)

            t = Torch_NoneType.get()
            assert str(t) == "!torch.none"
            assert Torch_NoneType.isinstance(t)

            t = Torch_StringType.get()
            assert str(t) == "!torch.str"
            assert Torch_StringType.isinstance(t)

            # t = TorchAnyType.get()
            # assert str(t) == "!torch.any"
            # assert TorchAnyType.isinstance(t)

            t = Torch_NumberType.get()
            assert str(t) == "!torch.number"
            assert Torch_NumberType.isinstance(t)

    def test_agg_types(self):
        with mlir_mod_ctx():
            t = Torch_NnModuleType.get("bob")
            assert str(t) == '!torch.nn.Module<"bob">'
            assert Torch_NnModuleType.isinstance(t)

            tint = Torch_IntType.get()
            tfloat = Torch_FloatType.get()
            tbool = Torch_BoolType.get()

            t = AnyTorchOptionalType.get(tint)
            assert str(t) == "!torch.optional<int>"
            assert AnyTorchOptionalType.isinstance(t)

            t = Torch_TupleType.get((tint, tfloat, tbool))
            assert str(t) == "!torch.tuple<int, float, bool>"
            assert Torch_TupleType.isinstance(t)
            assert len(t) == 3
            assert Torch_IntType.isinstance(t[0])
            assert Torch_FloatType.isinstance(t[1])
            assert Torch_BoolType.isinstance(t[2])

            # t = TorchUnionType.get((tint, tfloat, tbool))
            # assert TorchUnionType.isinstance(t)
            # assert str(t) == "!torch.union<int, float, bool>"
            # assert len(t) == 3
            # assert TorchIntType.isinstance(t[0])
            # assert TorchFloatType.isinstance(t[1])
            # assert TorchBoolType.isinstance(t[2])

            t = AnyTorchListType.get(tint)
            assert AnyTorchListType.isinstance(t)
            assert str(t) == "!torch.list<int>"

            # d = TorchDictType.get(tint, tfloat)
            # assert TorchDictType.isinstance(d)
            # assert str(d) == "!torch.dict<int, float>"
            # assert TorchIntType.isinstance(d.get_key_type())
            # assert TorchFloatType.isinstance(d.get_value_type())

    def test_tensor_types(self):
        with mlir_mod_ctx():
            t = Torch_NonValueTensorType.get([1, 2, 3], F32)
            assert Torch_NonValueTensorType.isinstance(t)
            assert str(t) == "!torch.tensor<[1,2,3],f32>"
            assert t.sizes() == (1, 2, 3)
            assert F32.isinstance(t.dtype())

            t = Torch_NonValueTensorType.get([], F32)
            assert str(t) == "!torch.tensor<[],f32>"
            assert t.sizes() == ()

            t = Torch_NonValueTensorType.get([-1], F32)
            assert str(t) == "!torch.tensor<[?],f32>"
            assert t.sizes() == (-1,)
            assert F32.isinstance(t.dtype())

            t = Torch_NonValueTensorType.get([-1, -1], F32)
            assert str(t) == "!torch.tensor<[?,?],f32>"
            assert t.sizes() == (-1, -1)
            assert F32.isinstance(t.dtype())

            t = Torch_ValueTensorType.get([1, 2, 3], F32)
            assert Torch_ValueTensorType.isinstance(t)
            assert str(t) == "!torch.vtensor<[1,2,3],f32>"
            assert t.sizes() == (1, 2, 3)
            assert F32.isinstance(t.dtype())

            t = Torch_ValueTensorType.get([], F32)
            assert str(t) == "!torch.vtensor<[],f32>"
            assert t.sizes() == ()

            t = Torch_ValueTensorType.get([-1], F32)
            assert str(t) == "!torch.vtensor<[?],f32>"
            assert t.sizes() == (-1,)
            assert F32.isinstance(t.dtype())

            t = Torch_ValueTensorType.get([-1, -1], F32)
            assert str(t) == "!torch.vtensor<[?,?],f32>"
            assert t.sizes() == (-1, -1)
            assert F32.isinstance(t.dtype())
