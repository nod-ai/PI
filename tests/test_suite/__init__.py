COMMON_TORCH_MLIR_LOWERING_XFAILS = {
    "QuantizedMLP_basic",
}

def register_all_tests():
    """Registers all the built-in E2E tests that Torch-MLIR provides."""
    # Side-effecting import statements.
    from . import conv
    from . import basic
    # from . import vision_models
    # from . import mlp
    # from . import norm_like
    # from . import quantized_models
    # from . import elementwise
    # from . import type_promotion
    # from . import type_conversion
    # from . import backprop
    # from . import reduction
    # from . import argmax
    # from . import matmul
    # from . import reshape_like
    # from . import scalar
    # from . import scalar_comparison
    # from . import elementwise_comparison
    # from . import squeeze
    # from . import slice_like
    # from . import nll_loss
    # from . import index_select
    # from . import arange
    # from . import constant_alloc
    # from . import threshold
    # from . import histogram_binning_calibration
    # from . import rng
    # from . import cast
    # from . import index_put
    # from . import pooling
    # from . import return_types
    # from . import control_flow
    # from . import stats
