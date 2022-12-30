import shark

from torch_mlir.dialects import torch
from torch_mlir.ir import IntegerType
import numpy as np

from shark.mlir_utils import mlir_cm

if __name__ == "__main__":
    with mlir_cm() as module:
        z = shark.empty((1, 2, 3))
        t = torch.ConstantIntOp(1).result
        print(t.type)
        i = IntegerType.get_unsigned(32)
        print(i)
        print(module)
        vt = shark.from_numpy(np.random.rand(10, 10))
        z = vt + vt
        print(module)
        # sizes = parse_sizes_from_tensor_type_str(z)
        # print(sizes)
