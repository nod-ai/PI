# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import shark
from shark import dtype
from shark.compiler.annotations import annotate_args
from tests.framework import TestUtils
from tests.registry import register_test_case


# ==============================================================================


class Conv2dNoPaddingModule(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = shark.nn.Conv2d(2, 10, 3, bias=False)

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], dtype.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dNoPaddingModule())
def Conv2dNoPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    return module.forward(t)


class Conv2dBiasNoPaddingModule(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = shark.nn.Conv2d(2, 10, 3, bias=True)

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dBiasNoPaddingModule())
def Conv2dBiasNoPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    return module.forward(t)


class Conv2dWithPaddingModule(shark.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = shark.nn.Conv2d(2, 10, 3, bias=False, padding=3)

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], dtype.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dWithPaddingModule())
def Conv2dWithPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    return module.forward(t)


class Conv2dWithPaddingDilationStrideModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = shark.nn.Conv2d(
            in_channels=2,
            out_channels=10,
            kernel_size=3,
            padding=3,
            stride=2,
            dilation=3,
            bias=False,
        )

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], shark.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dWithPaddingDilationStrideModule())
def Conv2dWithPaddingDilationStrideModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    return module.forward(t)


class Conv2dWithPaddingDilationStrideStaticModule(shark.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = shark.nn.Conv2d(
            in_channels=2,
            out_channels=10,
            kernel_size=3,
            padding=3,
            stride=2,
            dilation=3,
            bias=False,
        )

    @annotate_args(
        [
            None,
            ([5, 2, 10, 20], shark.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule()
)
def Conv2dWithPaddingDilationStrideStaticModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    return module.forward(t)


# ==============================================================================


# @register_test_case(module_factory=lambda: UpSampleNearest2d())
# def UpSampleNearest2d_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 1, 6, 12).to(shark.float64))
#
#
# class UpSampleNearest2dSameSize(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, inputVec):
#         return shark._C._nn.upsample_nearest2d(
#             inputVec, output_size=[11, 11], scales_h=None, scales_w=None
#         )
#
#
# @register_test_case(module_factory=lambda: UpSampleNearest2dSameSize())
# def UpSampleNearest2dStaticSize_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(1, 1, 4, 4))
#
#
# class UpSampleNearest2dDiffSize(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args([None, ([-1, -1, -1, -1], shark.float32, True)])
#     def forward(self, inputVec):
#         return shark._C._nn.upsample_nearest2d(
#             inputVec, output_size=[8, 11], scales_h=None, scales_w=None
#         )
#
#
# @register_test_case(module_factory=lambda: UpSampleNearest2dDiffSize())
# def UpSampleNearest2dDynamicSize_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 3, 2, 2))
#
#
# class UpSampleNearest2dDiffFactor(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args([None, ([-1, -1, -1, -1], shark.float32, True)])
#     def forward(self, inputVec):
#         return shark._C._nn.upsample_nearest2d(
#             inputVec, output_size=[6, 10], scales_h=2.3, scales_w=4.7
#         )
#
#
# @register_test_case(module_factory=lambda: UpSampleNearest2dDiffFactor())
# def UpSampleNearest2dDynamicFactor_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 3, 2, 2))
#
#
# class UpSampleNearest2dSameFactor(shark.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @annotate_args(
#         [
#             None,
#             ([-1, -1, -1, -1], shark.float32, True),
#         ]
#     )
#     def forward(self, inputVec):
#         return shark._C._nn.upsample_nearest2d(
#             inputVec, output_size=[8, 8], scales_h=2.0, scales_w=2.0
#         )
#
#
# @register_test_case(module_factory=lambda: UpSampleNearest2dSameFactor())
# def UpSampleNearest2dStaticFactor_basic(module, tu: TestUtils):
#     return module.forward(tu.rand(2, 3, 4, 4))
