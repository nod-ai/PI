import contextlib
import ctypes
import sys


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

import atexit

# Push a default context onto the context stack at import time.
DefaultContext = ir.Context()
DefaultContext.__enter__()
DefaultContext.allow_unregistered_dialects = True

torch_mlir.dialects.torch.register_dialect(DefaultContext, True)

@atexit.register
def __exit_ctxt():
    DefaultContext.__exit__(None, None, None)


DefaultLocation = ir.Location.unknown()
DefaultLocation.__enter__()


@atexit.register
def __exit_loc():
    DefaultLocation.__exit__(None, None, None)
