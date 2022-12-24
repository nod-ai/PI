from typing import Tuple

from torch_mlir.dialects import torch
from torch_mlir.ir import (
    DenseFPElementsAttr,
    Attribute,
    Value as MLIRValue
)

from shark import mlir_cm


def Tensor(sizes: Tuple[int, ...], init_value=0.0, dtype="f32") -> MLIRValue:
    attr = Attribute.parse(
        f"dense<{init_value}> : tensor<{'x'.join(map(str, sizes))}x{dtype}>"
    )
    dp_attr = DenseFPElementsAttr(attr)
    return torch.ValueTensorLiteralOp(dp_attr).result


if __name__ == "__main__":
    with mlir_cm() as module:
        z = Tensor((1, 2, 3))
        print(module)
        # sizes = parse_sizes_from_tensor_type_str(z)
        # print(sizes)
