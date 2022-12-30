import logging

logger = logging.getLogger(__name__)
# noinspection PyUnresolvedReferences
from .dialects import patch_meta_path_non_context

if __name__ == "pi":
    # prevent double patching of path during testing
    # where we've already patched torch -> sharkpy
    patch_meta_path_non_context()
else:
    logger.debug(f"reimporting sharkpy as {__name__}")

# this has to go before the above (otherwise torch extensions won't be picked up)
# noinspection PyUnresolvedReferences
from torch_mlir import ir
import torch_mlir

assert (
    len(torch_mlir.dialects.torch.AtenConv2dOp.__bases__) > 1
), "failed to import torch dialect extensions"

from ._tensor import *
from .types_ import *
from .dialects._torch_wrappers import *
from ._ops import _OpNamespace

ops = _OpNamespace("ops")
_nn = _OpNamespace("_nn")
_C = _OpNamespace("_C")
_VF = _OpNamespace("_VF")
special = _OpNamespace("special")
linalg = _OpNamespace("linalg")


from . import nn as nn


def manual_seed(*_, **__):
    return


DEBUG = True
