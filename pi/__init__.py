# this must be before all of other subpackage imports (the noinspection is so that pycharm doesn't reorder)
# noinspection PyUnresolvedReferences

# from .types_ import *
# from . import torch_wrappers
# from .torch_wrappers import *
#
# from ._tensor import *
#
# from .tensor_helpers import *
# from . import tensor_helpers
#
# import types
#
#
# class TorchWrappers:
#     def __getattr__(self, attr):
#         # prefer the rand, empty, etc in tensors over the one in wrappers
#         if hasattr(tensor_helpers, attr):
#             return getattr(tensor_helpers, attr)
#         if hasattr(torch_wrappers, attr):
#             return getattr(torch_wrappers, attr)
#         else:
#             raise NotImplementedError(f"_torch_wrappers.{attr}")
#
#
# t = TorchWrappers()
#
#
# class FakeModule(types.ModuleType):
#     def __init__(self, name, inner_modules=None):
#         if inner_modules is not None:
#             self.inner_modules = inner_modules
#         else:
#             self.inner_modules = []
#         super(FakeModule, self).__init__(name)
#
#     def __getattr__(self, attr):
#         if attr in self.inner_modules:
#             return t
#         else:
#             return getattr(t, attr)
#
#
# ops = FakeModule("ops", ["aten", "prim", "prims"])
# _C = FakeModule("_C", ["_nn"])
# _VF = FakeModule("_VF")
# special = FakeModule("special")
# linalg = FakeModule("linalg")
#
#
# def manual_seed(*_, **__):
#     return
#
#
# from . import nn as nn
