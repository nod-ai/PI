import numpy as np
from textwrap import dedent

from pi.mlir.utils import mlir_mod_ctx, _elementsAttr
from pi.mlir import (
    # AnyTorchDictKeyValue,
    AnyTorchListOfOptionalTensorValue,
    AnyTorchOptionalListOfTorchIntValue,
    # AnyTorchTensorValue,
    AnyTorchListOfTensorType,
    AnyTorchListOfTensorValue,
    AnyTorchListOfTorchBoolValue,
    AnyTorchListOfTorchIntValue,
    AnyTorchListOfTorchFloatValue,
    AnyTorchListOfTorchStringValue,
    AnyTorchListType,
    AnyTorchListValue,
    AnyTorchOptionalBoolValue,
    AnyTorchOptionalDeviceValue,
    AnyTorchOptionalFloatValue,
    AnyTorchOptionalGeneratorValue,
    AnyTorchOptionalIntValue,
    AnyTorchOptionalStringValue,
    AnyTorchOptionalTensorType,
    AnyTorchOptionalTensorValue,
    AnyTorchOptionalValue,
    AnyTorchScalarValue,
    Torch_BoolType,
    Torch_BoolValue,
    Torch_DeviceType,
    Torch_DeviceValue,
    Torch_DictType,
    Torch_DictValue,
    Torch_FloatType,
    Torch_FloatValue,
    Torch_IntType,
    Torch_IntValue,
    Torch_LinearParamsValue,
    Torch_NnModuleValue,
    Torch_NonValueTensorValue,
    Torch_NonValueTensorValue,
    Torch_NoneValue,
    Torch_NumberValue,
    Torch_StringValue,
    Torch_TupleType,
    Torch_TupleValue,
    Torch_ValueTensorValue,
    torch_dialect as torch,
)
from pi import ops
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
                "%float1.000000e00 = torch.constant.float 1.000000e+00",
                tfloat.owner,
            )
            assert Torch_FloatValue.isinstance(tfloat)
            check_correct(
                "Torch_FloatValue(%float1.000000e00 = torch.constant.float 1.000000e+00)",
                tfloat,
            )
            assert float(tfloat) == 1.0

            tint = torch.ConstantIntOp(1)
            check_correct(
                "%int1 = torch.constant.int 1",
                tint.owner,
            )
            assert Torch_IntValue.isinstance(tint)
            check_correct(
                "Torch_IntValue(%int1 = torch.constant.int 1)",
                tint,
            )
            assert int(tint) == 1

            tbool = torch.ConstantBoolOp(True)
            check_correct(
                "%true = torch.constant.bool true",
                tbool.owner,
            )
            assert Torch_BoolValue.isinstance(tbool)
            check_correct("Torch_BoolValue(%true = torch.constant.bool true)", tbool)
            assert bool(tbool) == True

            tdevice = torch.ConstantDeviceOp("cuda")
            check_correct(
                '%cuda = torch.constant.device "cuda"',
                tdevice.owner,
            )
            assert Torch_DeviceValue.isinstance(tdevice)
            check_correct(
                'Torch_DeviceValue(%cuda = torch.constant.device "cuda")',
                tdevice,
            )

            tnone = torch.ConstantNoneOp()
            check_correct(
                "%none = torch.constant.none",
                tnone.owner,
            )
            assert Torch_NoneValue.isinstance(tnone)
            check_correct(
                "Torch_NoneValue(%none = torch.constant.none)",
                tnone,
            )

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
                "Torch_TupleValue(%0 = torch.prim.TupleConstruct %int1, %float1.000000e00, %true : !torch.int, !torch.float, !torch.bool -> !torch.tuple<int, float, bool>)",
                tup,
            )

            t = AnyTorchListType.get(tint)
            lis = torch.PrimListConstructOp(t, (tintv, tintv, tintv))
            check_correct(
                "AnyTorchListValue(%1 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>)",
                lis,
            )

            l = ops.len(lis)
            check_correct(
                "Torch_IntValue(%2 = torch.aten.len.t %1 : !torch.list<int> -> !torch.int)",
                l,
            )

            t = AnyTorchListOfTorchBoolValue([True, False])
            check_correct(
                "AnyTorchListOfTorchBoolValue(%3 = torch.prim.ListConstruct %true_0, %false : (!torch.bool, !torch.bool) -> !torch.list<bool>)",
                t,
            )

            t = AnyTorchListOfTorchIntValue([0, 1])
            check_correct(
                "AnyTorchListOfTorchIntValue(%4 = torch.prim.ListConstruct %int0, %int1_1 : (!torch.int, !torch.int) -> !torch.list<int>)",
                t,
            )

            t = AnyTorchListOfTorchFloatValue([6.0, 8.0])
            check_correct(
                "AnyTorchListOfTorchFloatValue(%5 = torch.prim.ListConstruct %float0.000000e00, %float1.000000e00 : (!torch.float, !torch.float) -> !torch.list<float>)",
                t,
            )

            t = AnyTorchListOfTorchStringValue(["a", "b"])
            check_correct(
                "AnyTorchListOfTorchStringValue(%6 = torch.prim.ListConstruct %str, %str_3 : (!torch.str, !torch.str) -> !torch.list<str>)",
                t,
            )

            t = AnyTorchOptionalListOfTorchIntValue([1, 2])
            check_correct(
                "%7 = torch.prim.ListConstruct %int1_4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>",
                t.owner,
            )

            t = AnyTorchOptionalListOfTorchIntValue((1, 2))
            check_correct(
                "%7 = torch.prim.ListConstruct %int1_4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>",
                t.owner,
            )

            t = AnyTorchOptionalListOfTorchIntValue(None)
            check_correct(
                "%7 = torch.constant.none",
                t.owner,
            )

            t1 = torch.NonValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            t = AnyTorchOptionalListOfTorchIntValue([1, 2])
            st = ops.linalg_vector_norm(t1, 2.0, t, True, 1)
            check_correct(
                "%10 = torch.aten.linalg_vector_norm %8, %float1.000000e00, %9, %true_8, %int1_9 : !torch.tensor<[2,2],f64>, !torch.float, !torch.list<int>, !torch.bool, !torch.int -> !torch.tensor",
                st.owner,
            )

            st = ops.linalg_vector_norm(t1, 3.0, None, True, 1)
            check_correct(
                "%11 = torch.aten.linalg_vector_norm %8, %float1.000000e00, %none_11, %true_12, %int1_13 : !torch.tensor<[2,2],f64>, !torch.float, !torch.none, !torch.bool, !torch.int -> !torch.tensor",
                st.owner,
            )

            st = ops.linalg_vector_norm(t1, 4.0, None, True)
            check_correct(
                "%12 = torch.aten.linalg_vector_norm %8, %float1.000000e00, %none_15, %true_16, %none_17 : !torch.tensor<[2,2],f64>, !torch.float, !torch.none, !torch.bool, !torch.none -> !torch.tensor",
                st.owner,
            )

    def test_tensor_values(self):
        with mlir_mod_ctx():
            t = torch.NonValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            check_correct(
                "Tensor(%0 = torch.tensor.literal(dense<1.000000e+00> : tensor<2x2xf64>) : !torch.tensor<[2,2],f64>)",
                t,
            )

            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            check_correct(
                "Tensor(%1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<2x2xf64>) : !torch.vtensor<[2,2],f64>)",
                t,
            )

            tintv = torch.ConstantIntOp(1)
            res = t.add_(t, alpha=tintv)
            check_correct(
                "Tensor(%2 = torch.aten.add_.Tensor %1, %1, %int1 : !torch.vtensor<[2,2],f64>, !torch.vtensor<[2,2],f64>, !torch.int -> !torch.tensor)",
                res,
            )

            tfloatv = torch.ConstantFloatOp(1)
            res = t.add_(t, alpha=tfloatv)
            check_correct(
                "Tensor(%3 = torch.aten.add_.Tensor %1, %1, %float1.000000e00 : !torch.vtensor<[2,2],f64>, !torch.vtensor<[2,2],f64>, !torch.float -> !torch.tensor)",
                res,
            )

            tintv = torch.ConstantIntOp(1)
            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            lis_t = AnyTorchListOfTensorType.get(t.type)
            lis = torch.PrimListConstructOp(lis_t, (t, t))
            tcat = ops.cat(lis, tintv)
            check_correct(
                "Tensor(%6 = torch.aten.cat %5, %int1_0 : !torch.list<vtensor<[2,2],f64>>, !torch.int -> !torch.tensor)",
                tcat,
            )

    def test_defaulting_optional_values(self):
        with mlir_mod_ctx():
            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            opt_t = torch.ConstantNoneOp()
            clamped_t = ops.clamp(t, opt_t, opt_t)
            check_correct(
                "Tensor(%8 = torch.aten.clamp %7, %none, %none : !torch.vtensor<[2,2],f64>, !torch.none, !torch.none -> !torch.tensor)",
                clamped_t,
            )

            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            opt_t = torch.ConstantNoneOp()
            clamped_t = ops.clamp(t, opt_t, None)
            check_correct(
                "Tensor(%8 = torch.aten.clamp %7, %none, %none : !torch.vtensor<[2,2],f64>, !torch.none, !torch.none -> !torch.tensor)",
                clamped_t,
            )

            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            clamped_t = ops.clamp(t, None, None)
            check_correct(
                "Tensor(%8 = torch.aten.clamp %7, %none, %none : !torch.vtensor<[2,2],f64>, !torch.none, !torch.none -> !torch.tensor)",
                clamped_t,
            )

            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            clamped_t = ops.clamp(t)
            check_correct(
                "Tensor(%8 = torch.aten.clamp %7, %none, %none : !torch.vtensor<[2,2],f64>, !torch.none, !torch.none -> !torch.tensor)",
                clamped_t,
            )

            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            clamped_t = ops.clamp(t, t)
            check_correct(
                "Tensor(%9 = torch.aten.clamp.Tensor %8, %8, %none_13 : !torch.vtensor<[2,2],f64>, !torch.vtensor<[2,2],f64>, !torch.none -> !torch.tensor)",
                clamped_t,
            )

            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            clamped_t = ops.clamp(t, t, t)
            check_correct(
                "Tensor(%11 = torch.aten.clamp.Tensor %10, %10, %10 : !torch.vtensor<[2,2],f64>, !torch.vtensor<[2,2],f64>, !torch.vtensor<[2,2],f64> -> !torch.tensor)",
                clamped_t,
            )

    def test_scalar_type(self):
        with mlir_mod_ctx():
            tintv = torch.ConstantIntOp(1)
            t = ops.NumToTensor(AnyTorchScalarValue(tintv))
            check_correct(
                "Tensor(%0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.tensor)",
                t,
            )

            t = ops.NumToTensor(tintv)
            check_correct(
                "Tensor(%0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.tensor)",
                t,
            )

            try:
                t = torch.NonValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
                t = ops.NumToTensor(t)
            except TypeError as e:
                check_correct(
                    dedent(
                        """
                NumToTensor(): incompatible function arguments. The following argument types are supported:
                    1. (a: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue) -> pi.mlir._mlir_libs._pi_mlir.Tensor
                    
                Invoked with: <pi.mlir._mlir_libs._pi_mlir.Tensor object at 0x1153a9cf0>
                """
                    ),
                    e,
                )

            t = ops.Float(tintv)
            check_correct(
                "Torch_FloatValue(%3 = torch.aten.Float.Scalar %int1 : !torch.int -> !torch.float)",
                t,
            )

            t1 = torch.NonValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            t2 = torch.NonValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            t = ops.Float(tintv)
            t3 = ops.add(t1, t2, t)
            check_correct(
                "Tensor(%7 = torch.aten.add.Tensor %4, %5, %6 : !torch.tensor<[2,2],f64>, !torch.tensor<[2,2],f64>, !torch.float -> !torch.tensor)",
                t3,
            )

            t3 = ops.add(t1, t, t)
            check_correct(
                "Tensor(%8 = torch.aten.add.Scalar %4, %6, %6 : !torch.tensor<[2,2],f64>, !torch.float, !torch.float -> !torch.tensor)",
                t3,
            )

            try:
                t3 = ops.add(t, t, t)
                check_correct(
                    "Tensor(%8 = torch.aten.add.Scalar %4, %6, %6 : !torch.tensor<[2,2],f64>, !torch.float, !torch.float -> !torch.tensor)",
                    t3,
                )
            except TypeError as e:
                # sort alpha to get a deterministic order for the overloads because
                # they keep getting shuffled
                check_correct(
                    "\n".join(
                        sorted(
                            dedent(
                                """
                        add(): incompatible function arguments. The following argument types are supported:
                            1. (a: pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue, b: pi.mlir._mlir_libs._pi_mlir.Torch_IntValue) -> pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue
                            2. (a: pi.mlir._mlir_libs._pi_mlir.Torch_IntValue, b: pi.mlir._mlir_libs._pi_mlir.Torch_IntValue) -> pi.mlir._mlir_libs._pi_mlir.Torch_IntValue
                            3. (self: pi.mlir._mlir_libs._pi_mlir.Tensor, other: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue, alpha: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue = 1) -> pi.mlir._mlir_libs._pi_mlir.Tensor
                            4. (a: pi.mlir._mlir_libs._pi_mlir.Torch_StringValue, b: pi.mlir._mlir_libs._pi_mlir.Torch_StringValue) -> pi.mlir._mlir_libs._pi_mlir.Torch_StringValue
                            5. (a: pi.mlir._mlir_libs._pi_mlir.AnyTorchListValue, b: pi.mlir._mlir_libs._pi_mlir.AnyTorchListValue) -> pi.mlir._mlir_libs._pi_mlir.AnyTorchListValue
                            6. (self: pi.mlir._mlir_libs._pi_mlir.Tensor, other: pi.mlir._mlir_libs._pi_mlir.Tensor, alpha: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue = 1) -> pi.mlir._mlir_libs._pi_mlir.Tensor
                            7. (arg0: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue, arg1: pi.mlir._mlir_libs._pi_mlir.AnyTorchScalarValue) -> object
                        Invoked with: <pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue object at 0x111f7e870>, <pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue object at 0x111f7e870>, <pi.mlir._mlir_libs._pi_mlir.Torch_FloatValue object at 0x111f7e870>
                        """
                            ).splitlines(keepends=False)
                        )
                    ).strip(),
                    "\n".join(sorted(str(e).splitlines(keepends=False))).strip(),
                )

            t = ops.abs(tintv)
            check_correct(
                "Torch_IntValue(%9 = torch.prim.abs.Scalar %int1 : !torch.int -> !torch.int)",
                t,
            )

            t = ops.ceil(tintv)
            check_correct(
                "Torch_IntValue(%9 = torch.aten.ceil.Scalar %int1 : !torch.int -> !torch.int)",
                t,
            )

            tintv = torch.ConstantIntOp(1)
            res = ops.add(tintv, tintv)
            check_correct(
                "Torch_IntValue(%11 = torch.aten.add.int %int1_0, %int1_0 : !torch.int, !torch.int -> !torch.int)",
                res,
            )
            tfloatv = torch.ConstantFloatOp(1)
            res = ops.add(tfloatv, tfloatv)
            check_correct(
                "Torch_FloatValue(%12 = torch.aten.add %float1.000000e00, %float1.000000e00 : !torch.float, !torch.float -> !torch.float)",
                res,
            )

            try:
                res = ops.add(tintv, tfloatv)
            except NotImplementedError as e:
                check_correct(
                    "Arithmetic ops on Scalar values with like types supported; type a: AnyTorchScalarValue(%int1_0 = torch.constant.int 1), type b: AnyTorchScalarValue(%float1.000000e00 = torch.constant.float 1.000000e+00)",
                    e,
                )

    def test_multiple_returns(self):
        with mlir_mod_ctx():
            # aten::max.dim : (Tensor, int, bool) -> (Tensor, Tensor)
            t = torch.ValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            r = ops.max(t, 0, True)
            assert isinstance(r, tuple)
            check_correct(
                "%values, %indices = torch.aten.max.dim %0, %int0, %true : !torch.vtensor<[2,2],f64>, !torch.int, !torch.bool -> !torch.tensor, !torch.tensor",
                r[0].owner,
            )
            assert r[0].owner.results[0].result_number == 0
            assert r[0].owner.results[0] == r[0]
            check_correct(
                "%values, %indices = torch.aten.max.dim %0, %int0, %true : !torch.vtensor<[2,2],f64>, !torch.int, !torch.bool -> !torch.tensor, !torch.tensor",
                r[1].owner,
            )
            assert r[1].owner.results[1].result_number == 1
            assert r[1].owner.results[1] == r[1]

    def test_AnyTorchListOfOptionalTensorValue(self):
        with mlir_mod_ctx():

            t = torch.NonValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            l = AnyTorchListOfOptionalTensorValue([t, t])
            check_correct(
                "%1 = torch.prim.ListConstruct %0, %0 : (!torch.tensor<[2,2],f64>, !torch.tensor<[2,2],f64>) -> !torch.list<tensor<[2,2],f64>>",
                l.owner,
            )

            l = AnyTorchListOfOptionalTensorValue([None, None])
            check_correct(
                "%3 = torch.prim.ListConstruct %none, %none_0 : (!torch.none, !torch.none) -> !torch.list<none>",
                l.owner,
            )

            # aten::index.Tensor_hacked_twin : (Tensor, Tensor[]) -> (Tensor)
            r = ops.index(t, [t, t])
            check_correct(
                "%5 = torch.aten.index.Tensor_hacked_twin %0, %4 : !torch.tensor<[2,2],f64>, !torch.list<tensor<[2,2],f64>> -> !torch.tensor",
                r.owner,
            )

            r = ops.index(t, [None, None])
            check_correct(
                "%7 = torch.aten.index.Tensor %0, %6 : !torch.tensor<[2,2],f64>, !torch.list<none> -> !torch.tensor",
                r.owner,
            )

    def test_AnyTorchListType(self):
        with mlir_mod_ctx():
            t = torch.NonValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))

            r = ops.add([t, t], [t, t])
            check_correct(
                "%3 = torch.aten.add.t %1, %2 : !torch.list<tensor<[2,2],f64>>, !torch.list<tensor<[2,2],f64>> -> !torch.list<tensor<[2,2],f64>>",
                r.owner,
            )

            r = ops.mean(t, (0, 2))
            check_correct(
                "%5 = torch.aten.mean.dim %0, %4, %false, %none : !torch.tensor<[2,2],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.tensor",
                r.owner,
            )

            t = AnyTorchListValue([1, 2, 3])
            check_correct(
                "AnyTorchListValue(%1 = torch.prim.ListConstruct %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>)",
                t,
            )

    def test_AnyTorchListOfTensorType(self):
        with mlir_mod_ctx():
            t = torch.NonValueTensorLiteralOp(_elementsAttr(np.ones((2, 2))))
            tintv = torch.ConstantIntOp(1)
            r = ops.unbind(t, tintv)
            check_correct(
                "%1 = torch.aten.unbind.int %0, %int1 : !torch.tensor<[2,2],f64>, !torch.int -> !torch.list<tensor>",
                r.owner,
            )
