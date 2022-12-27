from typing import Tuple

from shark._mlir import _Tensor
from torch_mlir.dialects import torch
from torch_mlir.dialects._ods_common import get_op_result_or_value
from torch_mlir.ir import DenseFPElementsAttr, Attribute, Value as MLIRValue

from shark import mlir_cm
from shark.torch_types import TorchIntType


class Tensor(_Tensor):
    def __init__(self, torch_tensor: MLIRValue):
        torch_tensor = get_op_result_or_value(torch_tensor)
        super(Tensor, self).__init__(torch_tensor._CAPIPtr)

    def __add__(self, other):
        return Tensor(torch.AtenAddTensorOp(self, other))

    def __mul__(self, other):
        return Tensor(torch.AtenMulTensorOp(self, other))


def Empty(sizes: Tuple[int, ...], init_value=0.0, dtype="f32") -> Tensor:
    attr = Attribute.parse(
        f"dense<{init_value}> : tensor<{'x'.join(map(str, sizes))}x{dtype}>"
    )
    dp_attr = DenseFPElementsAttr(attr)
    vt = torch.ValueTensorLiteralOp(dp_attr)
    print(vt.result.type.context)
    t = Tensor(vt)
    return t


if __name__ == "__main__":
    with mlir_cm() as module:
        z = Empty((1, 2, 3))
        tt = TorchIntType(z.type)
        print(tt)
        print(module)
        # sizes = parse_sizes_from_tensor_type_str(z)
        # print(sizes)
