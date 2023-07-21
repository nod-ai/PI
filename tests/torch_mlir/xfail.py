# These test cases crash, preventing the remaining tests from executing
CRASHING = {
    "ArangeNegativeStartIntModule_basic",
    "ArangeStartNegativeStepIntModule_basic",
    "Aten_EmbeddingBagExample_basic",
    "BernoulliFloatModule_basic",
    "BernoulliPModule_basic",
    "ElementwiseFlattenBroadcastModule_basic",
    "ElementwiseEqBoolScalarModule_basic",
    "FlattenDynamicModule_basic",
    "FlattenRank0Module_basic",
    "FlipNegativeIndexModule_basic",
    "MobilenetV3Module_basic",
    "ResNet18Module",
    "ResNet18Module_basic",
    "ResNet18StaticModule_basic",
    "SliceCopyStartGreaterThanDimSize_Module_basic",
    "UniformModule_basic",
    "UniformStaticShapeModule_basic",
    "NativeDropoutTrainModule_basic",
    "NativeDropoutTrainStaticShapeModule_basic",
}

# These test cases are expected to fail due to an acceptable difference in generated IR
PI_XFAIL_SET = {
    # In these cases, torch-mlir spuriously initializes constants as double precision and truncates to floating point,
    # we simply initialize as single-precision causing an IR diff
    "ElementwiseGeFloatScalarModule_basic",
    "ArangeStartNegativeStepFloatModule_basic",
    "ArangeStartStepFloatModule_basic",
    "BaddbmmWithAlphaBetaModule_basic",
    "ElementwiseLeakyReluModule_basic",
    "ElementwiseLeakyReluStaticModule_basic",
    "ElementwiseLeFloatScalarModule_basic",
    "ElementwiseLtFloatScalarModule_basic",
    "ElementwiseGtFloatScalarModule_basic",
    "ElementwiseSubScalarFloatModule_basic",
    "HardtanhBackward_basic",
    "LeakyReluBackwardModule_basic",
    "LeakyReluBackwardStaticModule_basic",
    "Threshold3dFloatModule_basic",
    "ThresholdBackward3dFloatModule_basic",
    "TypePromotionAlphaWiderModule_basic",
    "TypePromotionSameCategoryZeroRankWider_basic",
    "NormalizeModule_basic",
    "NativeBatchNormNoneWeightModule_basic",
    "NativeBatchNorm3DModule_basic",
    "NativeBatchNorm2DModule_basic",
    "NativeBatchNorm1DModule_basic",
    # These test cases fail due to a difference in how PI generates some constant tensors as opposed to torch-mlir.
    # PI straightforwardly emits a constant tensor (arith.constant dense<0.0>...), whereas torch-mlir instead emits a
    # tensor.empty() and linalg.fill() instruction to achieve the same result
    "OnesModuleCPUDevice_basic",
    "OnesModuleDefaultDtype_basic",
    "OnesModuleFalsePinMemory_basic",
    "OnesModuleFloat_basic",
    "OnesModuleInt_basic",
    "TensorFloatModule_basic",
    "TensorIntModule_basic",
    "ZerosModuleDefaultDtype_basic",
    "ZerosModuleFalsePinMemory_basic",
    "ZerosModuleFloat2D_basic",
    "ZerosModuleFloat3D_basic",
    "ZerosModuleInt2D_basic",
    "ZerosModuleInt3D_basic",
    # These test cases fail due to trivial differences in constant floating point values, for example:
    # %cst = arith.constant -0.0099999997764825821 : f64 versus %cst = arith.constant -1.000000e-02 : f64
    # %cst = arith.constant 15.300000190734863 : f64 versus %cst = arith.constant 1.530000e+01 : f64
    "MaskedFillScalarFloatValueStaticModule_basic",
    "MaskedFillScalarFloatValueModule_basic",
    "FullLikeModuleFloat3DStatic_basic",
}

# These test cases are expected to fail due to an exception when attempting to generate the corresponding IR
PI_XFAIL_EXCEPTION_SET = {
    # Failure as a result of calling torch ops outside the test_module (i.e. in the program_invoker), this throws an
    # error as `TensorPlaceholder` is passed in place of the expected `Tensor` object. Currently, PI only supports
    # calling torch ops within the `forward` call of a `Module`
    "AtenComplexRealModule_basic",
    "BucketizeTensorFloatModule_basic",
    "BucketizeTensorStaticFloatModule_basic",
    "AtenComplexImagModule_basic",
    "HBC_basic",
    "ElementwiseAtenLogicalOrOpNegativeModule_basic",
}
