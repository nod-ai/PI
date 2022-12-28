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
from .variable_functions import *
from shark.nn.functional import *
# from shark import ops