import numpy as np
from textwrap import dedent

from pi.mlir.utils import mlir_mod_ctx
from pi.mlir import (
    nn,
    torch_dialect as torch,
    _fp64ElementsAttr,
)
from pi.mlir import F32, F64
from util import check_correct


class TestOverloadCast:
    def test_simple_values(self):
        with mlir_mod_ctx():
            one_int = torch.ConstantIntOp(1)
            two_int = torch.ConstantIntOp(2)
            res = nn.add(one_int, two_int)
            assert str(res) == "Torch_IntValue(%0 = torch.aten.add.int %int1, %int2 : !torch.int, !torch.int -> !torch.int)"

            one_float = torch.ConstantFloatOp(1.0)
            res = nn.add(one_float, two_int)
            assert str(res) == "Torch_FloatValue(%1 = torch.aten.add.float_int %float1.000000e00, %int2 : !torch.float, !torch.int -> !torch.float)"

