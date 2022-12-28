#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from torch_mlir.ir import *
    from torch_mlir.dialects._ods_common import (
        get_default_loc_context,
        get_op_result_or_value,
        get_op_results_or_values,
    )


except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

import re
from typing import Any, Optional, Tuple


class ConstantFloatOp:
    def __init__(self, value: float):
        f64 = F64Type.get()
        # f32 = F32Type.get()
        super().__init__(FloatAttr.get(f64, value))


class ConstantIntOp:
    def __init__(self, value: int):
        i64 = IntegerType.get_signless(64)
        super().__init__(IntegerAttr.get(i64, value))


class ConstantStrOp:
    def __init__(self, value: int):
        super().__init__(StringAttr.get(value))


class ConstantBoolOp:
    def __init__(self, value: bool):
        i1 = IntegerType.get_signless(1)
        super().__init__(IntegerAttr.get(i1, int(value)))


el_type_reg = re.compile(r"!torch\.(.*)")


class PrimListConstructOp:
    def __init__(
        self,
        elements,
        *,
        loc=None,
        ip=None,
    ):
        if len(elements):
            el_type = get_op_result_or_value(elements[0]).type
            el_type_str = el_type_reg.findall(str(el_type))[0]
            res_type = Type.parse(f"!torch.list<{el_type_str}>")
        else:
            res_type = Type.parse(f"!torch.list<int>")
        super().__init__(res_type, elements, loc=loc, ip=ip)


class AtenConv2dOp:
    def __init__(
        self,
        input,
        weight,
        bias: Optional[Any],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        groups: int,
        *,
        loc=None,
        ip=None,
    ):
        from torch_mlir.dialects import torch as torch_dialect

        input = get_op_result_or_value(input)
        # input_size, dtype = parse_sizes_from_tensor_type_str(input)
        weight = get_op_result_or_value(weight)
        # weight_size, _dtype = parse_sizes_from_tensor_type_str(weight)
        if bias is not None:
            bias = get_op_result_or_value(bias)
            # bias_size, _dtype = parse_sizes_from_tensor_type_str(bias)
        else:
            bias = torch_dialect.ConstantNoneOp()
            # bias_size = None

        # result_size = shape_functions.conv2d(input_size, weight_size, bias_size, stride, padding, dilation, groups)
        # TODO(max): implement torch types
        # result_type = Type.parse(f"!torch.vtensor<[{','.join(map(str, result_size))}],{dtype}>")
        result_type = Type.parse("!torch.vtensor")

        if stride[0] == stride[1]:
            stride = torch_dialect.ConstantIntOp(stride[0])
            stride = [stride, stride]
        else:
            stride = list(map(torch_dialect.ConstantIntOp, stride))
        stride = torch_dialect.PrimListConstructOp(stride)

        if padding[0] == padding[1]:
            padding = torch_dialect.ConstantIntOp(padding[0])
            padding = [padding, padding]
        else:
            padding = list(map(torch_dialect.ConstantIntOp, padding))
        padding = torch_dialect.PrimListConstructOp(padding)

        if dilation[0] == dilation[1]:
            dilation = torch_dialect.ConstantIntOp(dilation[0])
            dilation = [dilation, dilation]
        else:
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
        dilation = torch_dialect.PrimListConstructOp(dilation)

        groups = torch_dialect.ConstantIntOp(groups)

        super().__init__(
            result_type,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            loc=loc,
            ip=ip,
        )


class AtenAdd_TensorOp:
    def __init__(
        self,
        lhs,
        rhs,
        *,
        loc=None,
        ip=None,
    ):
        from torch_mlir.dialects import torch as torch_dialect

        # result, self_, other, alpha
        lhs = get_op_result_or_value(lhs)
        rhs = get_op_result_or_value(rhs)
        alpha = torch_dialect.ConstantFloatOp(1.0)
        result_type = Type.parse("!torch.vtensor")
        super().__init__(result_type, lhs, rhs, alpha, loc=loc, ip=ip)


class AtenAddTensorOp:
    def __init__(
        self,
        lhs,
        rhs,
        *,
        alpha=None,
        loc=None,
        ip=None,
    ):
        from torch_mlir.dialects import torch as torch_dialect

        # result, self_, other, alpha
        lhs = get_op_result_or_value(lhs)
        rhs = get_op_result_or_value(rhs)
        if alpha is None:
            alpha = torch_dialect.ConstantFloatOp(1.0)
        result_type = Type.parse("!torch.vtensor")
        super().__init__(result_type, lhs, rhs, alpha, loc=loc, ip=ip)


class AtenMulTensorOp:
    def __init__(
        self,
        lhs,
        rhs,
        *,
        loc=None,
        ip=None,
    ):
        # result, self_, other, alpha
        lhs = get_op_result_or_value(lhs)
        rhs = get_op_result_or_value(rhs)
        result_type = Type.parse("!torch.vtensor")
        super().__init__(result_type, lhs, rhs, loc=loc, ip=ip)


class AtenPadOp:
    def __init__(self, self_, pad, mode, value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)

        if pad[0] == pad[1]:
            pad = torch_dialect.ConstantIntOp(pad[0])
            pad = [pad, pad]
        else:
            pad = list(map(torch_dialect.ConstantIntOp, pad))
        pad = torch_dialect.PrimListConstructOp(pad)
        mode = torch_dialect.ConstantStrOp(mode)
        if value is not None:
            value = torch_dialect.ConstantFloatOp(value)
        else:
            value = torch_dialect.ConstantNoneOp()

        result_type = Type.parse("!torch.vtensor")
        super(AtenPadOp, self).__init__(
            result_type, self_, pad, mode, value, loc=loc, ip=ip
        )


class AtenMmOp:
    def __init__(self, self_, mat2, *, loc=None, ip=None):
        self_ = get_op_result_or_value(self_)
        mat2 = get_op_result_or_value(mat2)
        result_type = Type.parse("!torch.vtensor")
        super(AtenMmOp, self).__init__(result_type, self_, mat2, loc=loc, ip=ip)


class AtenBmmOp:
    def __init__(self, self_, mat2, *, loc=None, ip=None):
        self_ = get_op_result_or_value(self_)
        mat2 = get_op_result_or_value(mat2)
        result_type = Type.parse("!torch.vtensor")
        super(AtenBmmOp, self).__init__(result_type, self_, mat2, loc=loc, ip=ip)


class AtenTanhOp:
    def __init__(self, self_, *, loc=None, ip=None):
        self_ = get_op_result_or_value(self_)
        super(AtenTanhOp, self).__init__(self_.type, self_, loc=loc, ip=ip)


class AtenAddmmOp:
    def __init__(self, self_, mat1, mat2, beta, alpha, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)
        mat1 = get_op_result_or_value(mat1)
        mat2 = get_op_result_or_value(mat2)

        beta = torch_dialect.ConstantFloatOp(beta)
        alpha = torch_dialect.ConstantFloatOp(alpha)

        result_type = Type.parse("!torch.vtensor")
        super(AtenAddmmOp, self).__init__(
            result_type, self_, mat1, mat2, beta, alpha, loc=loc, ip=ip
        )


class AtenFlattenUsingIntsOp:
    def __init__(self, self_, start_dim: int, end_dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)
        start_dim = torch_dialect.ConstantIntOp(start_dim)
        end_dim = torch_dialect.ConstantIntOp(end_dim)
        result_type = Type.parse("!torch.vtensor")
        # result, self_, start_dim, end_dim
        super(AtenFlattenUsingIntsOp, self).__init__(
            result_type, self_, start_dim, end_dim, loc=loc, ip=ip
        )


class AtenTransposeIntOp:
    def __init__(self, self_, dim0, dim1, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)
        dim0 = torch_dialect.ConstantIntOp(dim0)
        dim1 = torch_dialect.ConstantIntOp(dim1)
        result_type = Type.parse("!torch.vtensor")
        super(AtenTransposeIntOp, self).__init__(
            result_type, self_, dim0, dim1, loc=loc, ip=ip
        )


class AtenPermuteOp:
    def __init__(self, self_, dims, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)
        dims = list(map(torch_dialect.ConstantIntOp, dims))
        dims = torch_dialect.PrimListConstructOp(dims)
        result_type = Type.parse("!torch.vtensor")
        super(AtenPermuteOp, self).__init__(result_type, self_, dims, loc=loc, ip=ip)


class AtenCatOp:
    def __init__(self, tensors, dim, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        tensors = get_op_results_or_values(tensors)
        tensors = torch_dialect.PrimListConstructOp(tensors)
        dim = torch_dialect.ConstantIntOp(dim)
        result_type = Type.parse("!torch.vtensor")
        super(AtenCatOp, self).__init__(result_type, tensors, dim, loc=loc, ip=ip)


class AtenGatherOp:
    def __init__(self, self_, dim, index, sparse_grad, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)
        dim = torch_dialect.ConstantIntOp(dim)
        sparse_grad = torch_dialect.ConstantBoolOp(sparse_grad)
        index = get_op_result_or_value(index)
        result_type = Type.parse("!torch.vtensor")
        super(AtenGatherOp, self).__init__(
            result_type, self_, dim, index, sparse_grad, loc=loc, ip=ip
        )


class AtenFill_ScalarOp:
    def __init__(self, self_, value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)
        if isinstance(value, int):
            value = torch_dialect.ConstantIntOp(value)
        elif isinstance(value, bool):
            value = torch_dialect.ConstantBoolOp(value)
        elif isinstance(value, float):
            value = torch_dialect.ConstantFloatOp(value)
        else:
            raise NotImplementedError

        result_type = Type.parse("!torch.vtensor")
        super(AtenFill_ScalarOp, self).__init__(
            result_type, self_, value, loc=loc, ip=ip
        )


class AtenEmbeddingOp:
    def __init__(
        self,
        weight,
        indices,
        padding_idx,
        scale_grad_by_freq,
        sparse,
        *,
        loc=None,
        ip=None,
    ):
        from torch_mlir.dialects import torch as torch_dialect

        weight = get_op_result_or_value(weight)
        indices = get_op_result_or_value(indices)
        padding_idx = torch_dialect.ConstantIntOp(padding_idx)
        scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        sparse = torch_dialect.ConstantBoolOp(sparse)

        result_type = Type.parse("!torch.vtensor")
        super(AtenEmbeddingOp, self).__init__(
            result_type,
            weight,
            indices,
            padding_idx,
            scale_grad_by_freq,
            sparse,
            loc=loc,
            ip=ip,
        )


class AtenEmbeddingBagPaddingIdxOp:
    def __init__(
        self,
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        *,
        loc=None,
        ip=None,
    ):
        from torch_mlir.dialects import torch as torch_dialect

        weight = get_op_result_or_value(weight)
        indices = get_op_result_or_value(indices)
        offsets = get_op_result_or_value(offsets)

        scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        mode = torch_dialect.ConstantIntOp(mode)
        sparse = torch_dialect.ConstantBoolOp(sparse)
        if per_sample_weights is not None:
            per_sample_weights = get_op_result_or_value(per_sample_weights)
        else:
            per_sample_weights = torch_dialect.ConstantNoneOp()
        include_last_offset = torch_dialect.ConstantBoolOp(include_last_offset)
        if padding_idx is not None:
            padding_idx = torch_dialect.ConstantIntOp(padding_idx)
        else:
            padding_idx = torch_dialect.ConstantNoneOp()

        result_type = Type.parse("!torch.vtensor")
        super(AtenEmbeddingBagPaddingIdxOp, self).__init__(
            result_type,
            result_type,
            result_type,
            result_type,
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
            loc=loc,
            ip=ip,
        )


class AtenArangeStartStepOp:
    def __init__(
        self,
        start,
        end,
        step,
        dtype,
        *,
        loc=None,
        ip=None,
    ):
        from torch_mlir.dialects import torch as torch_dialect

        # TODO(max): handle dtype
        start = torch_dialect.ConstantIntOp(start)
        end = torch_dialect.ConstantIntOp(end)
        step = torch_dialect.ConstantIntOp(step)
        none = torch_dialect.ConstantNoneOp().result

        result_type = Type.parse("!torch.vtensor")
        super(AtenArangeStartStepOp, self).__init__(
            result_type,
            start,
            end,
            step,
            none,
            none,
            none,
            none,
            loc=loc,
            ip=ip,
        )


class AtenSoftmaxIntOp:
    def __init__(self, self_, dim, dtype, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)
        # TODO(max): handle dtype
        dim = torch_dialect.ConstantIntOp(dim)
        dtype = torch_dialect.ConstantNoneOp().result

        result_type = Type.parse("!torch.vtensor")
        super(AtenSoftmaxIntOp, self).__init__(
            result_type, self_, dim, dtype, loc=loc, ip=ip
        )


class AtenSizeIntOp:
    def __init__(self, self_, dim, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect

        self_ = get_op_result_or_value(self_)
        dim = torch_dialect.ConstantIntOp(dim)
        super(AtenSizeIntOp, self).__init__(self_, dim, loc=loc, ip=ip)
