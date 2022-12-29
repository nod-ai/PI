import contextlib
import shark.dialects

from torch_mlir.ir import Context, Location, Module, InsertionPoint
from torch_mlir.dialects import torch as torch_dialect


@contextlib.contextmanager
def mlir_cm(enable_multithreading=False):
    with Context() as ctx, Location.unknown():
        ctx.enable_multithreading(enable_multithreading)
        torch_dialect.register_dialect(ctx, True)
        module = Module.create()
        with InsertionPoint(module.body):
            yield module


from ._tensor import Tensor
from .types_ import *
from shark import nn as nn

# from shark import ops

from .mlir_wrappers import (
    conv2d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    max_pool2d,
    relu,
    relu_,
    embedding,
    batch_norm,
    layer_norm,
    nll_loss_forward as nll_loss_nd,
    unsqueeze,
    any,
    log,
    div,
    cat,
    _embedding_bag as embedding_bag
)
