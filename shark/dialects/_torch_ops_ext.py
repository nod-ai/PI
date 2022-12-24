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
from typing import Any, List, Optional, Tuple


# !torch.vtensor<[1,2,3],f32>
reg = re.compile(r"!torch.vtensor<\[(.*)\],f32>")


def parse_sizes_from_tensor_type_str(t: OpView) -> List[int]:
    t = get_op_result_or_value(t)
    return list(map(int, reg.findall(str(t.type))[0].split(",")))


class ConstantFloatOp:
    def __init__(self, value: float):
        f64 = F64Type.get()
        super().__init__(FloatAttr.get(f64, value))


class ConstantIntOp:
    def __init__(self, value: int):
        i64 = IntegerType.get_signed(64)
        super().__init__(IntegerAttr.get(i64, value))


el_type_reg = re.compile(r"!torch\.(.*)")


class PrimListConstructOp:
    def __init__(
        self,
        elements,
        *,
        loc=None,
        ip=None,
    ):
        el_type = get_op_result_or_value(elements[0]).type
        el_type_str = el_type_reg.findall(str(el_type))[0]
        res_type = Type.parse(f"!torch.list<{el_type_str}>")
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
        weight = get_op_result_or_value(weight)
        if bias is not None:
            bias = get_op_result_or_value(bias)
        else:
            bias = torch_dialect.ConstantNoneOp()

        # TODO(max): implement torch types
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
