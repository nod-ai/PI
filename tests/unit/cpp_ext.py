def test_abi_compatibility():
    from pi import _mlir

    from torch_mlir import ir

    torch_mlir_pybind11_module_local_id = next(
        d for d in dir(ir.Value) if "pybind" in d
    )

    assert _mlir.get_pybind11_module_local_id() == torch_mlir_pybind11_module_local_id


def test_torch_types():
    import pi
    from pi.mlir_utils import mlir_cm
    from pi._mlir import _Torch_ValueTensorType
    from torch_mlir import ir

    with mlir_cm() as module:
        z = pi.empty((1, 2, 3))
        assert isinstance(z.type, _Torch_ValueTensorType)
        assert isinstance(z.shape, list)
        assert z.shape == [1, 2, 3]

        torch_types = [
            "f16",
            "bf16",
            "f32",
            "f64",
            "ui8",
            "si8",
            "si16",
            "si32",
            "si64",
            "i1",
            "!torch.qint8",
            "!torch.quint8",
        ]
        # for tt_str in torch_types:
        #     t = ir.Type.parse(f"!torch.vtensor<[-1, -1, -1],{tt_str}>")
        #     tt = _Torch_ValueTensorType(t)
        #     assert tt.sizes == [-1, -1, -1]
        #     assert str(tt_str) == str(tt.dtype)

        for p in pi.types_.dtype:
            if p in {pi.types_.dtype.qint8, pi.types_.dtype.quint8}:
                continue
            z = pi.empty((1, 2, 3), dtype=p)
            z = pi.ones((1, 2, 3), dtype=p)
            z = pi.zeros((1, 2, 3), dtype=p)
            # z = pi.rand((1, 2, 3), dtype=p)
            # z = pi.randn((1, 2, 3), dtype=p)
            z = pi.tensor((1, 2, 3), dtype=p)
            # assert z.dtype == p.to_mlir_type()


def test_macro_torch_types():
    # cmake-build-debug/tools/torch-mlir/include/torch-mlir/Dialect/Torch/IR/TorchTypes.h.inc
    from pi.mlir_utils import mlir_cm
    from torch_mlir.ir import F32Type
    from pi._mlir import (
        is_a_TorchListOfTorchBoolType,
        is_a_TorchListOfTorchIntType,
        is_a_TorchListOfTorchStringType,
        is_a_Torch_ListType,
        is_a_TorchScalarType,
        is_a_TorchTensorType,
        is_a_TorchType,
        is_a_Torch_BoolType,
        is_a_Torch_DeviceType,
        is_a_Torch_DictType,
        is_a_Torch_FloatType,
        is_a_Torch_GeneratorType,
        is_a_Torch_IntType,
        is_a_Torch_StringType,
        is_a_Torch_ValueTensorType,
        _Torch_BoolType,
        # _Torch_DeviceType,
        # _Torch_DictType,
        _Torch_FloatType,
        # _Torch_GeneratorType,
        _Torch_IntType,
        _Torch_StringType,
        _Torch_ValueTensorType,
    )

    with mlir_cm() as module:
        f = F32Type.get()
        t = _Torch_ValueTensorType()
        print(t)

def test_wrappers():
    from pi.mlir_utils import mlir_cm
    import pi
    from pi._mlir import is_a_Torch_ValueTensorType
    with mlir_cm() as module:
        f = pi.Float(1.0)

        input = pi.zeros((1, 3, 32, 32))
        assert is_a_Torch_ValueTensorType(input.type)
        weight = pi.zeros((9, 3, 3, 3))
        bias = pi.zeros((1, 9, 30, 30))
        out = pi.conv2d(input, weight, bias)

    print(module)

def test_numpy_wrappers():
    import numpy as np
    print(np.random.rand((1,2,3)))
    print(np.random.rand((1,2,3)))


if __name__ == "__main__":
    # test_abi_compatibility()
    test_torch_types()
    # test_macro_torch_types()
    # test_wrappers()
    # test_numpy_wrappers()