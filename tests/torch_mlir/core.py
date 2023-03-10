import contextlib
import ctypes
import sys


@contextlib.contextmanager
def dl_open_guard():
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    sys.setdlopenflags(old_flags)


class TestTorchMLIRCore:
    def test_import(self):
        with dl_open_guard():
            # noinspection PyUnresolvedReferences
            from torch_mlir import ir
            from torch_mlir.dialects import torch as torch_dialect
        assert (
            len(torch_dialect.AtenConv2dOp.__bases__) == 1
        ), "failed to import torch dialect extensions; you probably tried to import torch_mlir before pi"

    def test_smoke(self):
        with dl_open_guard():
            # noinspection PyUnresolvedReferences
            from torch_mlir import ir
            from torch_mlir.dialects import torch as torch_dialect

        with ir.Context() as ctx:
            torch_dialect.register_dialect(ctx)
            with ir.Location.unknown() as loc:
                module = ir.Module.create(loc)
                with ir.InsertionPoint.at_block_begin(module.body):
                    n = torch_dialect.ConstantNoneOp()

        assert (
            str(module).replace(" ", "").replace("\n", "")
            == "module{%none=torch.constant.none}"
        )
