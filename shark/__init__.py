import contextlib
import shark.dialects

from torch_mlir.ir import Context, Location, Module, InsertionPoint
from torch_mlir.dialects import torch


@contextlib.contextmanager
def mlir_cm():
    with Context() as ctx, Location.unknown():
        torch.register_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            yield module


from shark import nn
