PI_XFAIL_SET = {
    # view op
    "BernoulliFloatModule_basic",
    "ElementwiseFlattenBroadcastModule_basic",
    "FlattenDynamicModule_basic",
    "ElementwiseExpm1Module_basic",
    "FlattenRank0Module_basic",
    "UniformModule_basic",

    # torchvision
    "IouOfModule_basic",
    "ResNet18Module",
    "ResNet18Module_basic",
    "ResNet18StaticModule_basic",
    "MobilenetV3Module_basic",

    # tuple returns
    "AtenEmbeddingBagSumExample_basic",
    "Aten_EmbeddingBagExample_basic",
    "TestMultipleTensorAndPrimitiveTypesReturn_basic",
    "TestMultipleTensorReturn_basic",

    # python cast type return
    "BoolFloatConstantModule_basic",
    "BoolIntConstantModule_basic",
    "SqrtIntConstantModule_basic",

    # type/api overload
    "ArangeNegativeStartIntModule_basic",
    "ArangeStartNegativeStepIntModule_basic",
    "BernoulliModule_basic",
    "HBC_basic",
    "SortIntListReverse_basic",
    "SortIntList_basic",
    "UniformNoCorrelationModule_basic", # TypeError: 'torch_mlir._mlir_libs._mlir.ir.OpResult' object is not subscriptable

    # segfault (lol)
    "CopyModule_basic",
    "CopyWithDifferentDTypesAndSizesModule_basic",
    "CopyWithDifferentDTypesModule_basic",
    "CopyWithDifferentSizesModule_basic",
    "Fill_TensorFloat32WithFloat32_basic",
    "Fill_TensorFloat32WithFloat64_basic",
    "Fill_TensorFloat32WithInt64_basic",
    "Fill_TensorFloat64WithFloat32_basic",
    "Fill_TensorFloat64WithFloat64_basic",
    "Fill_TensorFloat64WithInt64_basic",
    "IndexPut1DFloatNonAccumulateModule_basic",
    "IndexPutImpl1DFloatAccumulateModule_basic",
    "IndexPutImpl1DFloatNonAccumulateModule_basic",
    "IndexPutImpl1DIntAccumulateModule_basic",
    "IndexPutImpl1DIntNonAccumulateModule_basic",
    "IndexPutImpl2DFloatAccumulateModule_basic",
    "IndexPutImpl2DFloatNonAccumulateModule_basic",
    "IndexPutImpl2DIntAccumulateModule_basic",
    "IndexPutImpl2DIntNonAccumulateModule_basic",
    "IndexPutImpl3DFloatAccumulateModule_basic",
    "IndexPutImpl3DFloatNonAccumulateModule_basic",
    "IndexPutImpl3DIntAccumulateModule_basic",
    "IndexPutImpl3DIntNonAccumulateModule_basic",
    "MaxPool2dCeilModeTrueModule_basic",
    "MaxPool2dModule_basic",
    "ZeroFloat32Module_basic",
    "ZeroInt32Module_basic",
    "ZeroInt64Module_basic",

    # eager/lazy materialization
    "TorchPrimLoopWhileLikeModule_basic",
    "TorchPrimLoopForLikeModule_basic",
    "EmbeddingModule1DIndices_basic",
    "EmbeddingModuleF16_basic",
    "EmbeddingModuleI32Static_basic",
    "EmbeddingModuleI32_basic",
    "EmbeddingModuleI64_basic",
    "AdaptiveAvgPool2dUnitOutputSizeStaticModule_basic",
    "AdaptiveAvgPool2dUnitOutputSizeDynamicModule_basic",
    "AdaptiveAvgPool2dNonUnitOutputSizeStaticModule_basic",
    "AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule_basic",
    "LayerNormNormalizeOverAllDimsModule_basic",
    "LayerNormModule_basic",
    "LayerNormLastDimModule_basic",
    "BatchNorm3DModule_basic",
    "BatchNorm2DModule_basic",
    "BatchNorm1DWith2DInputModule_basic",
    "BatchNorm1DModule_basic",
    "TensorIntModule_basic",
    "TensorLiteralModule_basic",
    "TensorOpaqueLiteralModule_basic",
    "TensorToBoolZeroRank_basic",
    "TensorToBool_basic",

    # backends
    "ConvolutionBackwardModule2DPadded_basic",
    "ConvolutionBackwardModule2D_basic",

    # failed to legalize operation
    "NumpyTRank0Module_basic",
    "TModuleRank0_basic",
    "TModuleRank1_basic",
    "TModuleRank2_basic",

    # error: found an op that was marked as backend illegal
    "AtenToDeviceModule_basic",
}
