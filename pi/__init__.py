import contextlib
import ctypes
import logging
import sys
import types

logger = logging.getLogger(__name__)
# noinspection PyUnresolvedReferences
# from .dialects import patch_meta_path_non_context

# if __name__ == "pi":
#     # prevent double patching of path during testing
#     # where we've already patched torch -> pi
#     patch_meta_path_non_context()
# else:
#     logger.debug(f"reimporting pi as {__name__}")


@contextlib.contextmanager
def dl_open_guard():
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    sys.setdlopenflags(old_flags)


with dl_open_guard():
    # noinspection PyUnresolvedReferences
    from torch_mlir import ir
    import torch_mlir

assert (
    len(torch_mlir.dialects.torch.AtenConv2dOp.__bases__) == 1
), "failed to import torch dialect extensions; you probably tried to import torch_mlir before pi"

from .types_ import *
from . import _torch_wrappers
from ._torch_wrappers import *

from ._tensor import *

from .tensor_helpers import *
from . import tensor_helpers


class torch_wrappers:
    def __getattr__(self, attr):
        # prefer the rand, empty, etc in tensors over the one in wrappers
        if hasattr(tensor_helpers, attr):
            return getattr(tensor_helpers, attr)
        if hasattr(_torch_wrappers, attr):
            return getattr(_torch_wrappers, attr)
        else:
            raise NotImplementedError(f"_torch_wrappers.{attr}")


t = torch_wrappers()


class FakeModule(types.ModuleType):
    def __init__(self, name, inner_modules=None):
        if inner_modules is not None:
            self.inner_modules = inner_modules
        else:
            self.inner_modules = []
        super(FakeModule, self).__init__(name)

    def __getattr__(self, attr):
        if attr in self.inner_modules:
            return t
        else:
            return getattr(t, attr)


ops = FakeModule("ops", ["aten", "prim", "prims"])
_C = FakeModule("_C", ["_nn"])
_VF = FakeModule("_VF")
special = FakeModule("special")
linalg = FakeModule("linalg")


def manual_seed(*_, **__):
    return


DEBUG = True

from pi import nn as nn
