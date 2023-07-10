CRASHING = {
    "ArangeNegativeStartIntModule_basic",
    "ArangeStartNegativeStepIntModule_basic",
    "Aten_EmbeddingBagExample_basic",
    "BernoulliFloatModule_basic",
    "BernoulliPModule_basic",
    "ElementwiseFlattenBroadcastModule_basic",
    "FlattenDynamicModule_basic",
    "FlattenRank0Module_basic",
    "FlipNegativeIndexModule_basic",
    "MobilenetV3Module_basic",
    "ResNet18Module",
    "ResNet18Module_basic",
    "ResNet18StaticModule_basic",
    "SliceCopyStartGreaterThanDimSize_Module_basic",
    "UniformModule_basic",
}

PI_XFAIL_SET = {

    # In these, torch-mlir spuriously initializes tensors as double precision and truncates to floating point, we simply initialize as single-precision causing an IR diff
    "ElementwiseGeFloatScalarModule_basic",
    "ArangeStartNegativeStepFloatModule_basic",
    "ArangeStartStepFloatModule_basic",
    "BaddbmmWithAlphaBetaModule_basic",
    "ElementwiseLeakyReluModule_basic",
    "ElementwiseLeakyReluStaticModule_basic",
    "ElementwiseLeFloatScalarModule_basic",
    "ElementwiseGtFloatScalarModule_basic",
    "ElementwiseSubScalarFloatModule_basic",
    "HardtanhBackward_basic",
    "LeakyReluBackwardModule_basic",
    "LeakyReluBackwardStaticModule_basic",
    "Threshold3dFloatModule_basic",
    "ThresholdBackward3dFloatModule_basic",
    "TypePromotionAlphaWiderModule_basic",
    "TypePromotionSameCategoryZeroRankWider_basic",

    # An IR difference due to an additional pass in torch-mlir, but functionally the same
    "NormalizeModule_basic"
}
