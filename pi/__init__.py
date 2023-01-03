import logging
import types

logger = logging.getLogger(__name__)
# noinspection PyUnresolvedReferences
from .dialects import patch_meta_path_non_context

if __name__ == "pi":
    # prevent double patching of path during testing
    # where we've already patched torch -> pi
    patch_meta_path_non_context()
else:
    logger.debug(f"reimporting pi as {__name__}")

# this has to go before the above (otherwise torch extensions won't be picked up)
# noinspection PyUnresolvedReferences
from torch_mlir import ir
import torch_mlir

assert (
    len(torch_mlir.dialects.torch.AtenConv2dOp.__bases__) > 1
), "failed to import torch dialect extensions"

from .types_ import *
from . import _torch_wrappers
from ._torch_wrappers import *

# prefer the rand, empty, etc in tensors over the one in wrappers
from ._tensor import *


class FakeModule(types.ModuleType):
    def __init__(self, name, inner_modules=None):
        if inner_modules is not None:
            self.inner_modules = inner_modules
        else:
            self.inner_modules = []
        super(FakeModule, self).__init__(name)

    def __getattr__(self, attr):
        if attr in self.inner_modules:
            return _torch_wrappers
        else:
            return getattr(_torch_wrappers, attr)


ops = FakeModule("ops", ["aten", "prim", "prims"])
_C = FakeModule("_C", ["_nn"])
_VF = FakeModule("_VF")
special = FakeModule("special")
linalg = FakeModule("linalg")


def manual_seed(*_, **__):
    return


DEBUG = True

from pi import nn as nn
