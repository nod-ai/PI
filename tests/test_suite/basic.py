# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import shark
from shark.compiler.annotations import annotate_args
from tests.framework import TestUtils
from tests.registry import register_test_case


# ==============================================================================


class MmModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1], shark.float32, True),
            ([-1, -1], shark.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return shark.mm(lhs, rhs)


@register_test_case(module_factory=lambda: MmModule())
def MmModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(4, 4), tu.rand(4, 4))


@register_test_case(module_factory=lambda: MmModule())
def MmModule_chained(module, tu: TestUtils):
    res = module.forward(tu.rand(4, 4), tu.rand(4, 4))
    return module.forward(res, res)


# ==============================================================================


class BmmModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1, -1], shark.float32, True),
            ([-1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return shark.bmm(lhs, rhs)


@register_test_case(module_factory=lambda: BmmModule())
def BmmModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(3, 4, 5), tu.rand(3, 5, 4))


# ==============================================================================


# class IsFloatingPointInt(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
# 
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.int32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.is_floating_point(x)
# 
# 
# @register_test_case(module_factory=lambda: IsFloatingPointInt())
# def IsFloatingPointInt_False(module, tu: TestUtils):
#     return module.forward(tu.randint(3, 3, high=100))
# 
# 
# # ==============================================================================
# 
# 
# class IsFloatingPointFloat(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
# 
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.is_floating_point(x)
# 
# 
# @register_test_case(module_factory=lambda: IsFloatingPointFloat())
# def IsFloatingPointFloat_True(module, tu: TestUtils):
#     return module.forward(tu.rand(3))


# # ==============================================================================
#
#
# class ContainsIntList(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args([None])
#     def forward(self):
#         return shark.__contains__([1, 2, 3], 3)
#
#
# @register_test_case(module_factory=lambda: ContainsIntList())
# def ContainsIntList_True(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================
#
#
# class ContainsIntListFalse(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args([None])
#     def forward(self):
#         return shark.__contains__([1, 2, 3], 4)
#
#
# @register_test_case(module_factory=lambda: ContainsIntListFalse())
# def ContainsIntList_False(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================


# A subgraph with multiple mm 
class MmDagModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([4, 4], shark.float32, True),
            ([4, 4], shark.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return shark.mm(lhs, shark.mm(lhs, rhs))


@register_test_case(module_factory=lambda: MmDagModule())
def MmDagModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(4, 4), tu.rand(4, 4))


# ==============================================================================


class MmTanhModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1], shark.float32, True),
            ([-1, -1], shark.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return shark.tanh(self.matmul(lhs, rhs))

    def matmul(self, lhs, rhs):
        return shark.mm(lhs, rhs)


@register_test_case(module_factory=lambda: MmTanhModule())
def MmTanhModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(4, 2), tu.rand(2, 4))


# ==============================================================================


class AddmmModuleFloat(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1], shark.float32, True),
            ([-1, -1], shark.float32, True),
            ([-1, -1], shark.float32, True),
        ]
    )
    def forward(self, M, mat1, mat2):
        return shark.addmm(M, mat1, mat2, beta=3.0, alpha=7.0)


@register_test_case(module_factory=lambda: AddmmModuleFloat())
def AddmmModuleFloat_basic(module, tu: TestUtils):
    return module.forward(tu.rand(4, 4), tu.rand(4, 2), tu.rand(2, 4))


#  ==============================================================================


class AddmmModuleBroadcastable(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([1, -1], shark.float32, True),
            ([-1, -1], shark.float32, True),
            ([-1, -1], shark.float32, True),
        ]
    )
    def forward(self, M, mat1, mat2):
        return shark.addmm(M, mat1, mat2, beta=2.0, alpha=7.0)


@register_test_case(module_factory=lambda: AddmmModuleBroadcastable())
def AddmmModule_broadcastable(module, tu: TestUtils):
    return module.forward(tu.rand(1, 2), tu.rand(3, 2), tu.rand(2, 2))


#  ==============================================================================


class AddmmModuleDifferentRankBroadcastable(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1], shark.float32, True),
            ([-1, -1], shark.float32, True),
            ([-1, -1], shark.float32, True),
        ]
    )
    def forward(self, M, mat1, mat2):
        return shark.addmm(M, mat1, mat2, beta=11.0, alpha=7.0)


@register_test_case(module_factory=lambda: AddmmModuleDifferentRankBroadcastable())
def AddmmModule_differentRankBroadcastable(module, tu: TestUtils):
    return module.forward(tu.rand(3), tu.rand(3, 2), tu.rand(2, 3))


# ==============================================================================


class FlattenStaticModule(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = shark.nn.Flatten(2, 4)

    @annotate_args(
        [
            None,
            ([10, 3, 8, 9, 3, 4], shark.float32, True),
        ]
    )
    def forward(self, x):
        return self.flat(x)


@register_test_case(module_factory=lambda: FlattenStaticModule())
def FlattenStaticModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(10, 3, 8, 9, 3, 4))


# ==============================================================================


# class FlattenRank0Module(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flat = shark.nn.Flatten(-1, -1)
#
#     @annotate_args(
#         [
#             None,
#             ([], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return self.flat(x)
#
#
# @register_test_case(module_factory=lambda: FlattenRank0Module())
# def FlattenRank0Module_basic(module, tu: TestUtils):
#     return module.forward(shark.tensor(4.0))


# ==============================================================================


# class FlattenDynamicModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flat = shark.nn.Flatten(2, 4)
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, 9, 3, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return self.flat(x)
#
#
# @register_test_case(module_factory=lambda: FlattenDynamicModule())
# def FlattenDynamicModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(10, 3, 8, 9, 3, 4))


# ==============================================================================


class ConstantPad2dStaticModule(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad2d = shark.nn.ConstantPad2d((0, 1, 2, 3), -float("inf"))

    @annotate_args(
        [
            None,
            ([1, 1, 20, 20], shark.float32, True),
        ]
    )
    def forward(self, x):
        return self.pad2d(x)


@register_test_case(module_factory=lambda: ConstantPad2dStaticModule())
def ConstantPad2dStaticModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(1, 1, 20, 20, low=-1))


# ==============================================================================


class PadModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, x):
        pad = [0, 1, 2, 3]
        mode = "constant"
        return shark.pad(x, pad, mode, float(1.5))


@register_test_case(module_factory=lambda: PadModule())
def PadModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(1, 1, 20, 20, low=-1))


# ==============================================================================


class PadWithNoneValModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, x):
        pad = [0, 1, 2, 3]
        mode = "constant"
        return shark.pad(x, pad, mode, None)


@register_test_case(module_factory=lambda: PadWithNoneValModule())
def PadWithNoneValModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(1, 1, 20, 20, low=-1))


# # ==============================================================================
#
#
# class ConstantPadNdModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.constant_pad_nd(x, (0, 1), -float("inf"))
#
#
# @register_test_case(module_factory=lambda: ConstantPadNdModule())
# def ConstantPadNdModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))
#
#
# # ==============================================================================
#
#
# class ConstantPadNdStaticModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([1, 1, 20, 20, 4, 4], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.constant_pad_nd(x, (0, 1), -float("inf"))
#
#
# @register_test_case(module_factory=lambda: ConstantPadNdStaticModule())
# def ConstantPadNdStaticModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))
#
#
# # ==============================================================================
#
#
# class ConstantPadNdPartialStaticModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([1, 1, 20, 20, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.constant_pad_nd(x, (0, 1, 2, 3), -float("inf"))
#
#
# @register_test_case(module_factory=lambda: ConstantPadNdPartialStaticModule())
# def ConstantPadNdPartialStaticModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))
#
#
# # ==============================================================================


class TransposeIntModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([3, 4, 2], shark.float32, True),
        ]
    )
    def forward(self, x):
        return shark.transpose(x, 0, 1)


@register_test_case(module_factory=lambda: TransposeIntModule())
def TransposeIntModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(3, 4, 2))


# ==============================================================================


class PermuteModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args([None, ([3, 4, 2], shark.float32, True)])
    def forward(self, x):
        return x.permute(0, 2, 1)


@register_test_case(module_factory=lambda: PermuteModule())
def PermuteModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(3, 4, 2))


# ==============================================================================


class PermuteNegativeIndexModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args([None, ([3, 4, 2], shark.float32, True)])
    def forward(self, x):
        return x.permute(0, -1, 1)


@register_test_case(module_factory=lambda: PermuteNegativeIndexModule())
def PermuteNegativeIndexModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(3, 4, 2))


# ==============================================================================


class Permute0RankModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args([None, ([], shark.float32, True)])
    def forward(self, x):
        return x.permute([])


@register_test_case(module_factory=lambda: Permute0RankModule())
def Permute0RankModule_basic(module, tu: TestUtils):
    return module.forward(shark.tensor(3.0))


# ==============================================================================


class TransposeIntNegDimsModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([3, 4, 2], shark.float32, True),
        ]
    )
    def forward(self, x):
        return shark.transpose(x, -1, -2)


@register_test_case(module_factory=lambda: TransposeIntNegDimsModule())
def TransposeIntNegDimsModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(3, 4, 2))


# ==============================================================================


class TensorsConcatModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1, -1], shark.float32, True),
            ([-1, -1, -1], shark.float32, True),
            ([-1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, x, y, z):
        return shark.cat([x, y, z], 1)


@register_test_case(module_factory=lambda: TensorsConcatModule())
def TensorsConcatModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 2, 4), tu.rand(2, 1, 4), tu.rand(2, 3, 4))


# ==============================================================================


class TensorsConcatNegativeDimModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1, -1], shark.float32, True),
            ([-1, -1, -1], shark.float32, True),
            ([-1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, x, y, z):
        return shark.cat([x, y, z], dim=-2)


@register_test_case(module_factory=lambda: TensorsConcatNegativeDimModule())
def TensorsConcatNegativeDimModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 2, 4), tu.rand(2, 1, 4), tu.rand(2, 3, 4))


# ==============================================================================


class GatherModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1, -1], shark.float32, True),
            ([-1, -1, -1], shark.int64, True),
        ]
    )
    def forward(self, tensor, indices):
        return shark.gather(tensor, 2, indices)


@register_test_case(module_factory=lambda: GatherModule())
def GatherModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 3, 4), shark.tensor([[[1, 2, 3], [1, 2, 3]]]))


# ==============================================================================


class GatherRandomIndexModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1, -1], shark.float32, True),
            ([-1, -1, -1], shark.int64, True),
        ]
    )
    def forward(self, tensor, indices):
        return shark.gather(tensor, 1, indices)


@register_test_case(module_factory=lambda: GatherRandomIndexModule())
def GatherRandomIndexModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 3, 4), tu.randint(2, 3, 4, high=3))


# ==============================================================================


class Gather2DInputModdule(shark.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1], shark.float32, True),
            ([-1, -1], shark.int64, True),
        ]
    )
    def forward(self, tensor, indices):
        return shark.gather(tensor, 1, indices)


@register_test_case(module_factory=lambda: Gather2DInputModdule())
def Gather2DInputModdule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(4, 5), shark.tensor([[1, 2, 3], [4, 3, 2]]))


# ==============================================================================


class GatherStaticModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([2, 3, 4], shark.float32, True),
            ([1, 2, 3], shark.int64, True),
        ]
    )
    def forward(self, tensor, indices):
        return shark.gather(tensor, 2, indices)


@register_test_case(module_factory=lambda: GatherStaticModule())
def GatherStaticModule_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 3, 4), shark.tensor([[[1, 2, 3], [1, 2, 3]]]))


# ==============================================================================


# class AddSizeIntModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, tensor):
#         # This is a workaround for not supporting scalar arguments.
#         # TODO: pass in dim as an argument to the forward method when scalar
#         # arguments are supported.
#         return tensor.add(tensor, alpha=tensor.size(1))
#
#
# @register_test_case(module_factory=lambda: AddSizeIntModule())
# def AddSizeIntModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randn(3, 3))


# ==============================================================================


class AddSizeIntNegDimModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

    @annotate_args(
        [
            None,
            ([-1, -1], shark.float32, True),
        ]
    )
    def forward(self, tensor):
        # This is a workaround for not supporting scalar arguments.
        # TODO: pass in dim as an argument to the forward method when scalar
        # arguments are supported.
        return tensor.add(tensor, alpha=tensor.size(-2))


@register_test_case(module_factory=lambda: AddSizeIntNegDimModule())
def AddSizeIntNegDimModule_basic(module, tu: TestUtils):
    return module.forward(tu.randn(3, 3))


# ==============================================================================


class EmbeddingModuleI64(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = shark.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @annotate_args(
        [
            None,
            ([-1, -1], shark.int64, True),
        ]
    )
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModuleI64())
def EmbeddingModuleI64_basic(module, tu: TestUtils):
    return module.forward(tu.randint(3, 3, high=100))


# ==============================================================================


class EmbeddingModuleI32(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = shark.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @annotate_args(
        [
            None,
            ([-1, -1], shark.int32, True),
        ]
    )
    def forward(self, indices):
        return self.embed.forward(indices)


# TODO(max): this is just a tensor cast
@register_test_case(module_factory=lambda: EmbeddingModuleI32())
def EmbeddingModuleI32_basic(module, tu: TestUtils):
    return module.forward(tu.randint(3, 3, high=100))


# ==============================================================================


class EmbeddingModuleF16(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = shark.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @annotate_args(
        [
            None,
            ([-1, -1], shark.int32, True),
        ]
    )
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModuleF16())
def EmbeddingModuleF16_basic(module, tu: TestUtils):
    return module.forward(tu.randint(3, 3, high=100))


# ==============================================================================


class EmbeddingModuleI32Static(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = shark.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @annotate_args(
        [
            None,
            ([3, 3], shark.int32, True),
        ]
    )
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModuleI32Static())
def EmbeddingModuleI32Static_basic(module, tu: TestUtils):
    return module.forward(tu.randint(3, 3, high=100))


# ==============================================================================


class EmbeddingModule1DIndices(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = shark.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @annotate_args(
        [
            None,
            ([-1], shark.int32, True),
        ]
    )
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModule1DIndices())
def EmbeddingModule1DIndices_basic(module, tu: TestUtils):
    return module.forward(tu.randint(3, high=100))


# ==============================================================================


class SoftmaxIntModule(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = shark.nn.Softmax(2)

    @annotate_args(
        [
            None,
            ([-1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, tensor):
        return self.softmax.forward(tensor)


@register_test_case(module_factory=lambda: SoftmaxIntModule())
def SoftmaxIntModule_basic(module, tu: TestUtils):
    return module.forward(tu.randn(3, 2, 4))


# ==============================================================================


# class _SoftmaxModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, tensor):
#         return shark._softmax(tensor, 0, False)
#
#
# @register_test_case(module_factory=lambda: _SoftmaxModule())
# def _SoftmaxModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randn(3, 2, 4))


# ==============================================================================


class SoftmaxIntNegDimModule(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = shark.nn.Softmax(-2)

    @annotate_args(
        [
            None,
            ([-1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, tensor):
        return self.softmax.forward(tensor)


@register_test_case(module_factory=lambda: SoftmaxIntNegDimModule())
def SoftmaxIntNegDimModule_basic(module, tu: TestUtils):
    return module.forward(tu.randn(3, 2, 4))


# ==============================================================================


class SoftmaxIntArgTypeF64Module(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = shark.nn.Softmax(2)

    @annotate_args(
        [
            None,
            ([-1, -1, -1], shark.float64, True),
        ]
    )
    def forward(self, tensor):
        return self.softmax.forward(tensor)


# TODO(max): this is a cast as well
@register_test_case(module_factory=lambda: SoftmaxIntArgTypeF64Module())
def SoftmaxIntArgTypeF64Module_basic(module, tu: TestUtils):
    return module.forward(tu.randn(3, 2, 4))


# ==============================================================================


# class _LogSoftmaxModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, tensor):
#         return shark._log_softmax(tensor, dim=0, half_to_float=False)
#
#
# @register_test_case(module_factory=lambda: _LogSoftmaxModule())
# def _LogSoftmaxModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randn(3, 2, 4))


# ==============================================================================


# class _LogSoftmaxModuleStable(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.float32, True),
#         ]
#     )
#     def forward(self, tensor):
#         return shark._log_softmax(tensor, dim=0, half_to_float=False)
#
#
# @register_test_case(module_factory=lambda: _LogSoftmaxModuleStable())
# def _LogSoftmaxModuleStable_basic(module, tu: TestUtils):
#     # testing for numerical stability.
#     # Should result in  tensor([-1e9, 0.00]) rather than tensor([-inf, 0.]).
#     a = shark.tensor([0, 1e9])
#     return module.forward(a)


# # ==============================================================================
#
#
# class SoftplusModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.softplus(x)
#
#
# @register_test_case(module_factory=lambda: SoftplusModule())
# def SoftplusModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 3))
#
#
# # ==============================================================================
#
#
# class HardsigmoidModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.hardsigmoid(x)
#
#
# @register_test_case(module_factory=lambda: HardsigmoidModule())
# def HardsigmoidModule_basic(module, tu: TestUtils):
#     return module.forward(shark.tensor([[4.0, -5.0, 3.0], [2.9, -1.5, -3.0]]))
#
#
# # ==============================================================================
#
#
# class HardsigmoidRandomModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.hardsigmoid(x)
#
#
# @register_test_case(module_factory=lambda: HardsigmoidRandomModule())
# def HardsigmoidRandomModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4, low=-10, high=10))
#
#
# # ==============================================================================
#
#
# class BroadcastToModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, 1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.broadcast_to(x, [1, -1, -1, 4])
#
#
# @register_test_case(module_factory=lambda: BroadcastToModule())
# def BroadcastToModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 1, 1))
#
#
# # ==============================================================================
#
#
# class BroadcastToSameRankStaticModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([3, 1, 8], shark.float32, True),
#             ([3, 1, 1], shark.float32, True),
#         ]
#     )
#     def forward(self, x, y):
#         y = shark.broadcast_to(y, [3, 1, 8])
#         return shark.sub(x, y)
#
#
# @register_test_case(module_factory=lambda: BroadcastToSameRankStaticModule())
# def BroadcastToSameRankStaticModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 1, 8), tu.rand(3, 1, 1))
#
#
# # ==============================================================================
#
#
# class BroadcastZeroRankInputStaticModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([3, 1, 8], shark.float32, True),
#             ([], shark.float32, True),
#         ]
#     )
#     def forward(self, x, y):
#         y = shark.broadcast_to(y, [3, 1, 8])
#         return shark.sub(x, y)
#
#
# @register_test_case(module_factory=lambda: BroadcastZeroRankInputStaticModule())
# def BroadcastZeroRankInputStaticModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 1, 8), tu.rand())
#
#
# # ==============================================================================
#
#
# class RollModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([3, -1, 2], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return x.roll([2, -1], [0, 2])
#
#
# @register_test_case(module_factory=lambda: RollModule())
# def RollModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 1, 2))
#
#
# # ==============================================================================
#
#
# class RepeatModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([3, 1, 2], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return x.repeat([2, 1, 3, 4])
#
#
# @register_test_case(module_factory=lambda: RepeatModule())
# def RepeatModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 1, 2))
#
#
# # ==============================================================================
#
#
# class ExpandModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, 1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return x.expand([1, -1, -1, 4])
#
#
# @register_test_case(module_factory=lambda: ExpandModule())
# def ExpandModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 1, 1))
#
#
# # ==============================================================================
#
#
# class ContiguousModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return x.contiguous()
#
#
# @register_test_case(module_factory=lambda: ContiguousModule())
# def ContiguousModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 1))
#
#
# # ==============================================================================
#
#
# class LogSoftmaxIntModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.log_softmax = shark.nn.LogSoftmax(2)
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float64, True),
#         ]
#     )
#     def forward(self, tensor):
#         return self.log_softmax.forward(tensor)
#
#
# @register_test_case(module_factory=lambda: LogSoftmaxIntModule())
# def LogSoftmaxIntModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randn(3, 2, 4).double())
#
#
# # ==============================================================================
#
#
# class NumToTensorIntModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         return shark.prim.NumToTensor(1)
#
#
# @register_test_case(module_factory=lambda: NumToTensorIntModule())
# def NumToTensorIntModule_basic(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================
#
#
# class NumToTensorFloatModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         return shark.prim.NumToTensor(1.0)
#
#
# @register_test_case(module_factory=lambda: NumToTensorFloatModule())
# def NumToTensorFloatModule_basic(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================
#
#
# # This test can be removed once we have one real op returning 3 float32 tensors
# class ReturnThreeTensorFloat32(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#             ([-1, -1], shark.float32, True),
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, a, b, c):
#         return a, b, c
#
#
# @register_test_case(module_factory=lambda: ReturnThreeTensorFloat32())
# def ReturnThreeTensorFloat32_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 3), tu.rand(2, 3), tu.rand(2, 3))
#
#
# # ==============================================================================
#
#
# class AddCMulModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#             ([-1, -1], shark.float32, True),
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input, tensor1, tensor2):
#         return shark.addcmul(input, tensor1, tensor2, value=1.0)
#
#
# @register_test_case(module_factory=lambda: AddCMulModule())
# def AddCMulModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 3), tu.rand(1, 3), tu.rand(1, 3))
#
#
# # ==============================================================================
#
#
# class AddCDivModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#             ([-1, -1], shark.float32, True),
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input, tensor1, tensor2):
#         return shark.addcdiv(input, tensor1, tensor2, value=1.0)
#
#
# @register_test_case(module_factory=lambda: AddCDivModule())
# def AddCDivModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 3), tu.rand(1, 3), tu.rand(1, 3))
#
#
# # ==============================================================================
#
#
# class tensorIntModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         a = 1
#         return shark.tensor(a)
#
#
# @register_test_case(module_factory=lambda: tensorIntModule())
# def TensorIntModule_basic(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================
#
#
# class tensorFloatModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         a = 1.0
#         return shark.tensor(a)
#
#
# @register_test_case(module_factory=lambda: tensorFloatModule())
# def TensorFloatModule_basic(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================
#
#
# class DropoutEvalIntModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.dropout(x, 0.2, train=False)
#
#
# @register_test_case(module_factory=lambda: DropoutEvalIntModule())
# def DropoutEvalIntModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(3, 4, low=5, high=10))
#
#
# # ==============================================================================
#
#
# class DropoutEvalFloatModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.dropout(x, 0.1, train=False)
#
#
# @register_test_case(module_factory=lambda: DropoutEvalFloatModule())
# def DropoutEvalFloatModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4))
#
#
# # ==============================================================================
#
#
# class DropoutTrainModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         res = shark.dropout(x, 0.3, train=True)
#         return shark.mean(res), shark.std(res)
#
#
# @register_test_case(module_factory=lambda: DropoutTrainModule())
# def DropoutTrainModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1024, 1536))
#
#
# # ==============================================================================
#
#
# class NumelModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input):
#         return shark.numel(input)
#
#
# @register_test_case(module_factory=lambda: NumelModule())
# def NumelModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(4, 3, 5))
#
#
# # ==============================================================================
#
#
# class NumelZeroRankModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([], shark.int64, True),
#         ]
#     )
#     def forward(self, input):
#         return shark.numel(input)
#
#
# @register_test_case(module_factory=lambda: NumelZeroRankModule())
# def NumelZeroRankModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(high=10))
#
#
# # ==============================================================================
#
#
# class BoolTensorReturnFalseModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.bool, True),
#         ]
#     )
#     def forward(self, a):
#         return a
#
#
# @register_test_case(module_factory=lambda: BoolTensorReturnFalseModule())
# def BoolTensorReturnFalseModule_basic(module, tu: TestUtils):
#     return module.forward(shark.tensor([0, 0], dtype=shark.bool))
#
#
# # ==============================================================================
#
#
# class BoolTensorReturnTrueModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.bool, True),
#         ]
#     )
#     def forward(self, a):
#         return a
#
#
# @register_test_case(module_factory=lambda: BoolTensorReturnTrueModule())
# def BoolTensorReturnTrueModule_basic(module, tu: TestUtils):
#     return module.forward(shark.tensor([1, 1, 1, 1, 1], dtype=shark.bool))
#
#
# # ==============================================================================
#
#
# class BoolTensorReturnMixedModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.bool, True),
#         ]
#     )
#     def forward(self, a):
#         return a
#
#
# @register_test_case(module_factory=lambda: BoolTensorReturnMixedModule())
# def BoolTensorReturnMixedModule_basic(module, tu: TestUtils):
#     return module.forward(shark.tensor([[1, 0], [0, 1]], dtype=shark.bool))
#
#
# # ==============================================================================
#
#
# class BoolTensorHandleSignless(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.bool, True),
#             ([-1, -1], shark.bool, True),
#         ]
#     )
#     def forward(self, a, b):
#         return a * b
#
#
# @register_test_case(module_factory=lambda: BoolTensorHandleSignless())
# def BoolTensorHandleSignless_basic(module, tu: TestUtils):
#     a = shark.tensor([[1, 1], [1, 1]], dtype=shark.bool)
#     b = shark.tensor([[0, 0], [0, 0]], dtype=shark.bool)
#     return module.forward(a, b)
#
#
# # ==============================================================================
#
#
# class TModuleRank2(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, lhs):
#         return shark.t(lhs)
#
#
# @register_test_case(module_factory=lambda: TModuleRank2())
# def TModuleRank2_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4))
#
#
# # ==============================================================================
#
#
# class TModuleRank1(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.float32, True),
#         ]
#     )
#     def forward(self, lhs):
#         return shark.t(lhs)
#
#
# @register_test_case(module_factory=lambda: TModuleRank1())
# def TModuleRank1_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3))
#
#
# # ==============================================================================
#
#
# class TModuleRank0(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([], shark.float32, True),
#         ]
#     )
#     def forward(self, lhs):
#         return shark.t(lhs)
#
#
# @register_test_case(module_factory=lambda: TModuleRank0())
# def TModuleRank0_basic(module, tu: TestUtils):
#     return module.forward(shark.tensor(7, dtype=shark.float32))
#
#
# # ==============================================================================
#
#
# class TensorLiteralModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.t = shark.randint(-5, 5, (2, 3))
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         return shark.add(self.t, self.t)
#
#
# @register_test_case(module_factory=lambda: TensorLiteralModule())
# def TensorLiteralModule_basic(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================
#
#
# class TensorOpaqueLiteralModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.t = shark.randint(-5, 5, (256, 1024))
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         return shark.add(self.t, self.t)
#
#
# @register_test_case(module_factory=lambda: TensorOpaqueLiteralModule())
# def TensorOpaqueLiteralModule_basic(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================
#
#
# class ReturnTwoTensorF32I64(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#             ([-1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, a, b):
#         return a, b
#
#
# @register_test_case(module_factory=lambda: ReturnTwoTensorF32I64())
# def ReturnTwoTensorF32I64_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 3), tu.randint(2, 3, high=5))
#
#
# # ==============================================================================
#
#
# class IndexTensorModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.float32, True),
#             ([-1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index):
#         return shark.index(x, (index,))
#
#
# @register_test_case(module_factory=lambda: IndexTensorModule())
# def IndexTensorModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(5), tu.randint(2, 3, high=4))
#
#
# # ==============================================================================
#
#
# class IndexTensorModule3dInput(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index):
#         return shark.index(x, (index,))
#
#
# @register_test_case(module_factory=lambda: IndexTensorModule3dInput())
# def IndexTensorModule3dInput_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(5, 4, 3), tu.randint(2, 3, high=3))
#
#
# # ==============================================================================
#
#
# class IndexTensorSelectDimModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, a, ind):
#         return shark.index(a, (None, ind, None))
#
#
# @register_test_case(module_factory=lambda: IndexTensorSelectDimModule())
# def IndexTensorSelectDimModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 4, 6), tu.randint(2, 3, high=3))
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInput(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([3, 3], shark.int64, True),
#             ([3], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2):
#         return shark.index(
#             x,
#             (
#                 index1,
#                 index2,
#             ),
#         )
#
#
# @register_test_case(module_factory=lambda: IndexTensorMultiInput())
# def IndexTensorMultiInput_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(5, 4, 3), tu.randint(3, 3, high=3), tu.randint(3, high=3)
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInputOneDim(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([6, 1], shark.int64, True),
#             ([3], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2):
#         return shark.index(
#             x,
#             (
#                 index1,
#                 index2,
#             ),
#         )
#
#
# @register_test_case(module_factory=lambda: IndexTensorMultiInputOneDim())
# def IndexTensorMultiInputOneDim_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(5, 4, 3), tu.randint(6, 1, high=4), tu.randint(3, high=3)
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInputContiguousOneDimDynamic(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, 1], shark.int64, True),
#             ([-1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2):
#         return shark.index(
#             x,
#             (
#                 None,
#                 index1,
#                 index2,
#             ),
#         )
#
#
# @register_test_case(
#     module_factory=lambda: IndexTensorMultiInputContiguousOneDimDynamic()
# )
# def IndexTensorMultiInputContiguousOneDimDynamic_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(5, 4, 3), tu.randint(6, 1, high=4), tu.randint(3, high=3)
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInputNonContiguousOneDimDynamic(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, 1], shark.int64, True),
#             ([-1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2):
#         return shark.index(
#             x,
#             (
#                 index1,
#                 None,
#                 index2,
#             ),
#         )
#
#
# @register_test_case(
#     module_factory=lambda: IndexTensorMultiInputNonContiguousOneDimDynamic()
# )
# def IndexTensorMultiInputNonContiguousOneDimDynamic_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(5, 4, 3), tu.randint(6, 1, high=4), tu.randint(3, high=3)
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInputNonContiguousDynamic(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, 2], shark.int64, True),
#             ([-1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2):
#         return shark.index(
#             x,
#             (
#                 index2,
#                 None,
#                 index1,
#             ),
#         )
#
#
# @register_test_case(module_factory=lambda: IndexTensorMultiInputNonContiguousDynamic())
# def IndexTensorMultiInputNonContiguousDynamic_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(5, 4, 3), tu.randint(6, 2, high=2), tu.randint(2, high=3)
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInputNonContiguousMultipleStaticDims(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1], shark.float32, True),
#             ([4, 1], shark.int64, True),
#             ([1, 3], shark.int64, True),
#             ([-1, 3], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2, index3):
#         return shark.index(x, (index1, index2, index3))
#
#
# @register_test_case(
#     module_factory=lambda: IndexTensorMultiInputNonContiguousMultipleStaticDims()
# )
# def IndexTensorMultiInputNonContiguousMultipleStaticDims_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(5, 4, 3, 2),
#         tu.randint(4, 1, high=3),
#         tu.randint(1, 3, high=1),
#         tu.randint(4, 3, high=1),
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInputNonContiguous(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1], shark.float32, True),
#             ([4, 2], shark.int64, True),
#             ([4, 2], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2):
#         return shark.index(x, (index1, None, index2))
#
#
# @register_test_case(module_factory=lambda: IndexTensorMultiInputNonContiguous())
# def IndexTensorMultiInputNonContiguous_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(5, 4, 3, 2), tu.randint(4, 2, high=3), tu.randint(4, 2, high=1)
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInputThreeIndexers(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1, -1, -1], shark.float32, True),
#             ([8, 4, 2], shark.int64, True),
#             ([8, 1, 1], shark.int64, True),
#             ([4, 2], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2, index3):
#         return shark.index(x, (None, None, index1, None, index2, index3))
#
#
# @register_test_case(module_factory=lambda: IndexTensorMultiInputThreeIndexers())
# def IndexTensorMultiInputThreeIndexers_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(1, 2, 4, 4, 5, 3),
#         tu.randint(8, 4, 2, high=3),
#         tu.randint(8, 1, 1, high=4),
#         tu.randint(4, 2, high=2),
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorMultiInputContiguousCenter(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1], shark.float32, True),
#             ([2, 2], shark.int64, True),
#             ([2], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2):
#         return shark.index(x, (None, index1, index2, None))
#
#
# @register_test_case(module_factory=lambda: IndexTensorMultiInputContiguousCenter())
# def IndexTensorMultiInputContiguousCenter_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(5, 4, 3, 2), tu.randint(2, 2, high=3), tu.randint(2, high=2)
#     )
#
#
# # ==============================================================================
#
#
# class IndexTensorHackedTwinModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.float32, True),
#             ([-1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index):
#         return shark.index(x, [index])
#
#
# @register_test_case(module_factory=lambda: IndexTensorHackedTwinModule())
# def IndexTensorHackedTwinModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(5), tu.randint(2, 3, high=4))
#
#
# # ==============================================================================
#
#
# class IndexTensorHackedTwinModule3dInput(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index):
#         return shark.index(x, [index])
#
#
# @register_test_case(module_factory=lambda: IndexTensorHackedTwinModule3dInput())
# def IndexTensorHackedTwinModule3dInput_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(5, 4, 3), tu.randint(2, 3, high=3))
#
#
# # ==============================================================================
#
#
# class IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1], shark.float32, True),
#             ([4, 1], shark.int64, True),
#             ([1, 3], shark.int64, True),
#             ([-1, 3], shark.int64, True),
#         ]
#     )
#     def forward(self, x, index1, index2, index3):
#         return shark.index(x, [index1, index2, index3])
#
#
# @register_test_case(
#     module_factory=lambda: IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims()
# )
# def IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims_basic(
#     module, tu: TestUtils
# ):
#     return module.forward(
#         tu.rand(5, 4, 3, 2),
#         tu.randint(4, 1, high=3),
#         tu.randint(1, 3, high=1),
#         tu.randint(4, 3, high=1),
#     )
#
#
# # ==============================================================================
#
#
# class SquareModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.square(x)
#
#
# @register_test_case(module_factory=lambda: SquareModule())
# def SquareModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 3, 4))
#
#
# # ==============================================================================
#
#
# class HardswishModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.hardswish(x)
#
#
# @register_test_case(module_factory=lambda: HardswishModule())
# def HardswishModule_basic(module, tu: TestUtils):
#     return module.forward(shark.tensor([[4.0, -5.0, 3.0], [2.9, -1.5, -3.0]]))
#
#
# # ==============================================================================
#
#
# class HardswishRandomModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.hardswish(x)
#
#
# @register_test_case(module_factory=lambda: HardswishRandomModule())
# def HardswishRandomModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(128, 128, low=-10, high=10))
#
#
# # ==============================================================================
#
#
# class SiluModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.silu(x)
#
#
# @register_test_case(module_factory=lambda: SiluModule())
# def SiluModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(128, 128, low=-10, high=10))
#
#
# # ==============================================================================
#
#
# class HardTanhModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.hardtanh(x, min_val=-2, max_val=2)
#
#
# @register_test_case(module_factory=lambda: HardTanhModule())
# def HardTanhModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(100, 100, low=-5, high=5))
#
#
# # ==============================================================================
#
#
# class HardTanhIntModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.hardtanh(x, min_val=-2, max_val=2)
#
#
# @register_test_case(module_factory=lambda: HardTanhIntModule())
# def HardTanhIntModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(100, 100, low=-5, high=5))
#
#
# # ==============================================================================
#
#
# class BincountModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.int64, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.bincount(x)
#
#
# @register_test_case(module_factory=lambda: BincountModule())
# def BincountModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(1000, high=10))
#
#
# # ==============================================================================
#
#
# class BincountStaticSizeModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([200], shark.int64, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.bincount(x)
#
#
# @register_test_case(module_factory=lambda: BincountStaticSizeModule())
# def BincountStaticSizeModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(200, high=100))
#
#
# # ==============================================================================
#
#
# class BincountMinlengthModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.int64, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.bincount(x, minlength=600)
#
#
# @register_test_case(module_factory=lambda: BincountMinlengthModule())
# def BincountMinlengthModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(20, high=5))
#
#
# # ==============================================================================
#
#
# class ExpandAsFloatModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, 1, 1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x, y):
#         return shark.expand_as(x, y)
#
#
# @register_test_case(module_factory=lambda: ExpandAsFloatModule())
# def ExpandAsFloatModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 1, 1), tu.rand(3, 4, 5))
#
#
# class ExpandAsIntModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([1, 1, 1], shark.int64, True),
#             ([-1, -1, -1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, y):
#         return shark.expand_as(x, y)
#
#
# @register_test_case(module_factory=lambda: ExpandAsIntModule())
# def ExpandAsIntModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(1, 1, 1, high=100), tu.randint(4, 5, 6, high=200))
#
#
# # ==============================================================================
#
#
# class CopyModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x, y):
#         return shark.copy_(x, y)
#
#
# @register_test_case(module_factory=lambda: CopyModule())
# def CopyModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 4))
#
#
# class CopyWithDifferentSizesModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, 4], shark.float32, True),
#             ([-1, -1, 1], shark.float32, True),
#         ]
#     )
#     def forward(self, x, y):
#         return shark.copy_(x, y)
#
#
# @register_test_case(module_factory=lambda: CopyWithDifferentSizesModule())
# def CopyWithDifferentSizesModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 1))
#
#
# class CopyWithDifferentDTypesModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.int64, True),
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x, y):
#         return shark.copy_(x, y)
#
#
# @register_test_case(module_factory=lambda: CopyWithDifferentDTypesModule())
# def CopyWithDifferentDTypesModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(3, 2, 4, high=100), tu.rand(3, 2, 4))
#
#
# class CopyWithDifferentDTypesAndSizesModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, 4], shark.float32, True),
#             ([-1, -1, 1], shark.int64, True),
#         ]
#     )
#     def forward(self, x, y):
#         return shark.copy_(x, y)
#
#
# @register_test_case(module_factory=lambda: CopyWithDifferentDTypesAndSizesModule())
# def CopyWithDifferentDTypesAndSizesModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 2, 4), tu.randint(3, 2, 1, high=1000))
#
#
# # ==============================================================================
#
#
# class ToCopyModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark._to_copy(x)
#
#
# @register_test_case(module_factory=lambda: ToCopyModule())
# def ToCopyModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 2, 4))
#
#
# class ToCopyWithDTypeModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark._to_copy(x, dtype=shark.int64)
#
#
# @register_test_case(module_factory=lambda: ToCopyWithDTypeModule())
# def ToCopyWithDTypeModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 2, 4))
#
#
# class ToCopyWithDTypeFalsePinMemoryModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark._to_copy(x, dtype=shark.int64, pin_memory=False)
#
#
# @register_test_case(module_factory=lambda: ToCopyWithDTypeFalsePinMemoryModule())
# def ToCopyWithDTypeFalsePinMemoryModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 2, 4))
#
#
# class ToCopyBoolDTypeStaticModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([1, 1, 5, 5], shark.uint8, True),
#         ]
#     )
#     def forward(self, x):
#         return shark._to_copy(x, dtype=shark.bool)
#
#
# @register_test_case(module_factory=lambda: ToCopyBoolDTypeStaticModule())
# def ToCopyBoolDTypeStaticModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(1, 1, 5, 5).to(dtype=shark.uint8))
#
#
# # ==============================================================================
#
#
# class FlipModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.flip(x, [1, 2])
#
#
# @register_test_case(module_factory=lambda: FlipModule())
# def FlipModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 2, 4))
#
#
# # ==============================================================================
#
#
# class DetachModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, x):
#         return shark.detach(x)
#
#
# @register_test_case(module_factory=lambda: DetachModule())
# def DetachModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 2, 4))
#
#
# # ==============================================================================
#
#
# class LenStrModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.str = "test"
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         return shark.len(self.str)
#
#
# @register_test_case(module_factory=lambda: LenStrModule())
# def LenStrModule_basic(module, tu: TestUtils):
#     return module.forward()
#
#
# # ==============================================================================
#
#
# class ScalarImplicitFloatModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([], shark.float64, True),
#         ]
#     )
#     def forward(self, x):
#         return float(shark.ScalarImplicit(x))
#
#
# @register_test_case(module_factory=lambda: ScalarImplicitFloatModule())
# def ScalarImplicitFloatModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand().double())
#
#
# class ScalarImplicitIntModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([], shark.int64, True),
#         ]
#     )
#     def forward(self, x):
#         return int(shark.ScalarImplicit(x))
#
#
# @register_test_case(module_factory=lambda: ScalarImplicitIntModule())
# def ScalarImplicitIntModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randint(low=-100, high=100))
#
#
# # ==============================================================================
#
#
# class BaddbmmDynamicModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input, batch1, batch2):
#         return shark.baddbmm(input, batch1, batch2)
#
#
# @register_test_case(module_factory=lambda: BaddbmmDynamicModule())
# def BaddbmmDynamicModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))
#
#
# class BaddbmmStaticModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([5, 2, 7], shark.float32, True),
#             ([5, 2, 9], shark.float32, True),
#             ([5, 9, 7], shark.float32, True),
#         ]
#     )
#     def forward(self, input, batch1, batch2):
#         return shark.baddbmm(input, batch1, batch2)
#
#
# @register_test_case(module_factory=lambda: BaddbmmStaticModule())
# def BaddbmmStaticModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(5, 2, 7), tu.rand(5, 2, 9), tu.rand(5, 9, 7))
#
#
# class BaddbmmDifferentDtypesModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.int64, True),
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input, batch1, batch2):
#         return shark.baddbmm(input, batch1, batch2)
#
#
# @register_test_case(module_factory=lambda: BaddbmmDifferentDtypesModule())
# def BaddbmmDifferentDtypesModule_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.randint(3, 4, 5, high=10), tu.rand(3, 4, 6), tu.rand(3, 6, 5)
#     )
#
#
# class BaddbmmWithAlphaModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input, batch1, batch2):
#         return shark.baddbmm(input, batch1, batch2, alpha=5)
#
#
# @register_test_case(module_factory=lambda: BaddbmmWithAlphaModule())
# def BaddbmmWithAlphaModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))
#
#
# class BaddbmmWithBetaModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input, batch1, batch2):
#         return shark.baddbmm(input, batch1, batch2, beta=0.5)
#
#
# @register_test_case(module_factory=lambda: BaddbmmWithBetaModule())
# def BaddbmmWithBetaModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))
#
#
# class BaddbmmWithAlphaBetaModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input, batch1, batch2):
#         return shark.baddbmm(input, batch1, batch2, beta=6, alpha=2.4)
#
#
# @register_test_case(module_factory=lambda: BaddbmmWithAlphaBetaModule())
# def BaddbmmWithAlphaBetaModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))
#
#
# class BaddbmmBroadcast1DInputModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([1], shark.float32, True),
#             ([5, 2, 9], shark.float32, True),
#             ([5, 9, 7], shark.float32, True),
#         ]
#     )
#     def forward(self, input, batch1, batch2):
#         return shark.baddbmm(input, batch1, batch2)
#
#
# @register_test_case(module_factory=lambda: BaddbmmBroadcast1DInputModule())
# def BaddbmmBroadcast1DInputModule_basic(module, tu: TestUtils):
#     return module.forward(
#         tu.rand(
#             1,
#         ),
#         tu.rand(5, 2, 9),
#         tu.rand(5, 9, 7),
#     )
#
#
# class BaddbmmBroadcast2DInputModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([2, 7], shark.float32, True),
#             ([5, 2, 9], shark.float32, True),
#             ([5, 9, 7], shark.float32, True),
#         ]
#     )
#     def forward(self, input, batch1, batch2):
#         return shark.baddbmm(input, batch1, batch2)
#
#
# @register_test_case(module_factory=lambda: BaddbmmBroadcast2DInputModule())
# def BaddbmmBroadcast2DInputModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 7), tu.rand(5, 2, 9), tu.rand(5, 9, 7))
#
#
# # ==============================================================================
#
#
# class NumpyTRankNStaticModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([3, 4, 5, 6], shark.float32, True),
#         ]
#     )
#     def forward(self, lhs):
#         return shark.numpy_T(lhs)
#
#
# @register_test_case(module_factory=lambda: NumpyTRankNStaticModule())
# def NumpyTRankNStaticModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4, 5, 6))
#
#
# class NumpyTRankNDynamicModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, lhs):
#         return shark.numpy_T(lhs)
#
#
# @register_test_case(module_factory=lambda: NumpyTRankNDynamicModule())
# def NumpyTRankNDynamicModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4, 5, 6, 2))
#
#
# class NumpyTRank2Module(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, lhs):
#         return shark.numpy_T(lhs)
#
#
# @register_test_case(module_factory=lambda: NumpyTRank2Module())
# def NumpyTRank2Module_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3, 4))
#
#
# class NumpyTRank1Module(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1], shark.float32, True),
#         ]
#     )
#     def forward(self, lhs):
#         return shark.numpy_T(lhs)
#
#
# @register_test_case(module_factory=lambda: NumpyTRank1Module())
# def NumpyTRank1Module_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(3))
#
#
# class NumpyTRank0Module(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([], shark.float32, True),
#         ]
#     )
#     def forward(self, lhs):
#         return shark.numpy_T(lhs)
#
#
# @register_test_case(module_factory=lambda: NumpyTRank0Module())
# def NumpyTRank0Module_basic(module, tu: TestUtils):
#     return module.forward(shark.tensor(7, dtype=shark.float32))
#
#
# class AtenEmbeddingBagSumExample(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#             ([-1], shark.int64, True),
#             ([-1], shark.int64, True),
#         ]
#     )
#     def forward(self, weight, indices, offsets):
#         return shark.embedding_bag(
#             weight,
#             indices,
#             offsets,
#             scale_grad_by_freq=False,
#             mode=0,
#             sparse=False,
#             per_sample_weights=None,
#             include_last_offset=False,
#             padding_idx=None,
#         )
#
#
# @register_test_case(module_factory=lambda: AtenEmbeddingBagSumExample())
# def AtenEmbeddingBagSumExample_basic(module, tu: TestUtils):
#     weight = shark.rand(100, 10)
#     indices = shark.LongTensor(
#         [0, 1, 2, 2, 0, 2, 1, 3, 20, 50, 99, 2, 4, 5, 6, 7, 34, 54]
#     )
#     offsets = shark.LongTensor([0, 3, 5, 7, 9, 10, 15])
#     return module.forward(weight, indices, offsets)
#
#
# class Aten_EmbeddingBagExample(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#             ([-1], shark.int64, True),
#             ([-1], shark.int64, True),
#         ]
#     )
#     def forward(self, weight, indices, offsets):
#         return shark._embedding_bag(weight, indices, offsets)
#
#
# @register_test_case(module_factory=lambda: Aten_EmbeddingBagExample())
# def Aten_EmbeddingBagExample_basic(module, tu: TestUtils):
#     weight = shark.rand(100, 10)
#     indices = shark.LongTensor(
#         [0, 1, 2, 2, 0, 2, 1, 3, 20, 50, 99, 2, 4, 5, 6, 7, 34, 54]
#     )
#     offsets = shark.LongTensor([0, 3, 5, 7, 9, 10, 15])
#     return module.forward(weight, indices, offsets)
#
#
# # ==============================================================================
#
#
# class CumsumModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, val):
#         return shark.cumsum(val, 1)
#
#
# @register_test_case(module_factory=lambda: CumsumModule())
# def CumsumModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 7, 4))
#
#
# class CumsumStaticModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([2, 7, 4], shark.float32, True),
#         ]
#     )
#     def forward(self, val):
#         return shark.cumsum(val, 1)
#
#
# @register_test_case(module_factory=lambda: CumsumStaticModule())
# def CumsumStaticModule_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 7, 4))
#
#
# # ==============================================================================
#
#
# class AtenToDeviceModule(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, val):
#         return shark.to(
#             val, device="cpu", dtype=shark.float, non_blocking=False
#         )
#
#
# @register_test_case(module_factory=lambda: AtenToDeviceModule())
# def AtenToDeviceModule_basic(module, tu: TestUtils):
#     return module.forward(tu.randn(2, 4))
#
#
# # ==============================================================================
#
#
# class UpSampleNearest2dBackward(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1], shark.float64, True),
#         ]
#     )
#     def forward(self, input):
#         return shark.upsample_nearest2d_backward(
#             input,
#             output_size=[6, 12],
#             input_size=[1, 1, 2, 3],
#             scales_h=3.0,
#             scales_w=4.0,
#         )
#
#
# @register_test_case(module_factory=lambda: UpSampleNearest2dBackward())
# def UpSampleNearest2dBackward_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 1, 6, 12).to(shark.float64))
#
#
# class UpSampleNearest2dBackwardScalesNone(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, input):
#         return shark.upsample_nearest2d_backward(
#             input,
#             output_size=[4, 8],
#             input_size=[1, 1, 2, 3],
#             scales_h=None,
#             scales_w=None,
#         )
#
#
# @register_test_case(module_factory=lambda: UpSampleNearest2dBackwardScalesNone())
# def UpSampleNearest2dBackwardScalesNone_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 1, 4, 8))
#
#
# # ==============================================================================
#
#
# class SortIntList(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         a = [1, 0, 3, 2]
#         b = [0, 1, 2, 3]
#         a.sort()
#         return a == b
#
#
# @register_test_case(module_factory=lambda: SortIntList())
# def SortIntList_basic(module, tu: TestUtils):
#     return module.forward()
#
#
# class SortIntListReverse(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#         ]
#     )
#     def forward(self):
#         a = [1, 0, 3, 2]
#         b = [3, 2, 1, 0]
#         a.sort(reverse=True)
#         return a == b
#
#
# @register_test_case(module_factory=lambda: SortIntListReverse())
# def SortIntListReverse_basic(module, tu: TestUtils):
#     return module.forward()
