PI_XFAIL_SET = {
    # view op (Which doesn't pass regular torch-mlir either???)
    "BernoulliFloatModule_basic",
    "ElementwiseFlattenBroadcastModule_basic",
    "FlattenDynamicModule_basic",
    "FlattenRank0Module_basic",
    "UniformModule_basic",

    # torchvision
    "IouOfModule_basic",
    "MobilenetV3Module_basic",
    "ResNet18Module",
    "ResNet18Module_basic",
    "ResNet18StaticModule_basic",

    # tuple returns
    # torhc-mlir handles this incorrectly - if you tuple up the returns, it'll then unpack them during the
    # AdjustCallingConventionsPass but won't update the funcop type correctly (and thus the verifier throws
    # error: 'func.return' op has 2 operands, but enclosing function (@forward) returns 1)
    "AtenEmbeddingBagSumExample_basic",
    "TestMultipleTensorAndPrimitiveTypesReturn_basic",
    "TestMultipleTensorReturn_basic",

    # segfault
    "Aten_EmbeddingBagExample_basic",

    # type/api overload
    "NumpyTRank0Module_basic",
    "SortIntListReverse_basic", # just compares lists...
    "SortIntList_basic",
    "TModuleRank0_basic",
    "TModuleRank1_basic",
    "TModuleRank2_basic",
    "UniformNoCorrelationModule_basic",  # TypeError: 'torch_mlir._mlir_libs._mlir.ir.OpResult' object is not subscriptable

    # linalg lowering fail
    "Fill_TensorFloat64WithFloat64_basic",
    "Fill_TensorFloat64WithInt64_basic",

    # eager/lazy materialization (24) TypeError: object of type 'Torch_Value' has no len()
    "AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule_basic",
    "AdaptiveAvgPool2dNonUnitOutputSizeStaticModule_basic",
    "AdaptiveAvgPool2dUnitOutputSizeDynamicModule_basic",
    "AdaptiveAvgPool2dUnitOutputSizeStaticModule_basic",
    "EmbeddingModule1DIndices_basic",
    "EmbeddingModuleF16_basic",
    "EmbeddingModuleI32Static_basic",
    "EmbeddingModuleI32_basic",
    "EmbeddingModuleI64_basic",
    "HBC_basic",
    "TorchPrimLoopForLikeModule_basic",
    "TorchPrimLoopWhileLikeModule_basic",
    "ElementwiseAtenLogicalOrOpNegativeModule", # torch.neg(tu.randint(2, 3, 4, 5, low=10, high=100)))
    "ElementwiseAtenLogicalOrOpNegativeModule_basic", # torch.neg(tu.randint(2, 3, 4, 5, low=10, high=100)))

    # backends
    "ConvolutionBackwardModule2DPadded_basic",
    "ConvolutionBackwardModule2D_basic",


    # error: found an op that was marked as backend illegal
    "AtenToDeviceModule_basic", # basically an type match issue around dtype/device, pi lowers to torch.aten.to.dtype_layout while torch_mlir lowers to torch.aten.to.device

    # int/float casts (29)
    "AddIntModule_basic",
    "AtenIntBoolOpConstFalseModule_basic",
    "AtenIntBoolOpConstTrueModule_basic",
    "AtenIntBoolOpModule_basic",
    "AtenIntTensorByteDtypeModule_basic",
    "AtenIntTensorCharDtypeModule_basic",
    "BoolFloatConstantModule_basic",
    "BoolFloatFalseModule_basic",
    "BoolFloatTrueModule_basic",
    "BoolIntConstantModule_basic",
    "BoolIntFalseModule_basic",
    "BoolIntTrueModule_basic",
    "CeilFloatModule_basic",
    "DivFloatModule_basic",
    "DivIntModule_basic",
    "EqIntModule_basic",
    "GeFloatIntModule_basic",
    "GeFloatModule_basic",
    "GeIntModule_basic",
    "GtFloatIntModule_basic",
    "GtIntModule_basic",
    "MulIntModule_basic",
    "NeFloatIntModule_basic",
    "NeIntModule_basic",
    "ScalarImplicitFloatModule_basic",
    "ScalarImplicitIntModule_basic",
    "SqrtIntConstantModule_basic",
    "SqrtIntModule_basic",
    "SubFloatModule_basic",
    "SubIntModule_basic",
    "TensorToFloatZeroRank_basic",
    "TensorToFloat_basic",
    "TensorToIntZeroRank_basic",
    "TensorToInt_basic",
    "UnsafeViewCollapseDynamicWithAtenSizeIntModule_basic",
    "ViewCollapseDynamicWithAtenSizeIntModule_basic",

    # doesn't abide by schema (mostly empty dim) - maxpool needs to be rerouted through functional (where _pair, _triple, etc)
    "MaxPool2dWithIndicesAllNegativeValuesModule_basic",
    "MaxPool2dWithIndicesFullSizeKernelModule_basic",
    "MaxPool2dWithIndicesNonDefaultDilationModule_basic",
    "MaxPool2dWithIndicesNonDefaultPaddingModule_basic",
    "MaxPool2dWithIndicesNonDefaultStrideModule_basic",
    "MaxPool2dWithIndicesStaticModule_basic",
    "MeanDimEmptyDimModule_basic",
    "Permute0RankModule_basic",
    "ReduceAmaxSingleDim_basic",
    "ReduceL1NormModule_basic",
    "ReduceL1NormWithDTypeModule_basic",
    "ReduceL2NormModule_basic",
    "ReduceLN3NormModule_basic",
    "ReduceSumDimIntListEmptyDimModule_basic",
    "StdCorrectionEmptyDimModule_basic",
    "StdDimEmptyDimModule_basic",
    "VarCorrectionEmptyDimModule_basic",
    "VarDimEmptyDimModule_basic",
    "ReduceSumDimIntListElementTypeBoolModule_basic",
    "ReduceSumDimIntListKeepDimNegativeDimStaticModule_basic",
    "ViewDynamicExpandWithAtenSizeIntModule_basic",
    "ViewFlattenAndExpandModule_basic",

    # dtype issue (number conflates with int)
    # error: unsupported: conversion to byte or char type for convertScalarToDtype 'i64'(scalar type) -> 'i8'(dtype)
    "ArangeStartNegativeStepIntModule_basic",
}
