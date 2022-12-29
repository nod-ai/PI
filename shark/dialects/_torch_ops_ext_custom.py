#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    # from shark import Tensor, Number
    from torch_mlir.ir import *
    from torch_mlir.dialects._ods_common import (
        get_default_loc_context,
        get_op_result_or_value,
        get_op_results_or_values,
    )
    from ._torch_ops_ext import *
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

import re
from typing import Any, Optional, Tuple, List, Union


def is_mlir_value(v):
    return isinstance(v, (OpView, Operation, Value, OpResultList))


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


class ConstantNumberOp:
    def __init__(self, value: Union[int, float]):
        if isinstance(value, int):
            i64 = IntegerType.get_signless(64)
            super().__init__(IntegerAttr.get(i64, value))
        elif isinstance(value, float):
            f64 = F64Type.get()
            # f32 = F32Type.get()
            super().__init__(FloatAttr.get(f64, value))
        else:
            raise Exception(f"unknown number type {value}")


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
