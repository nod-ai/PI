from shark import mlir_cm, empty, from_numpy
from torch_mlir.dialects import torch
from torch_mlir.ir import IntegerType
import numpy as np

from shark.types_ import Torch_ValueTensorType, Torch_IntType

if __name__ == "__main__":
    with mlir_cm() as module:
        # z = empty((1, 2, 3))
        # tv = Torch_ValueTensorType(z)
        # print(tv)
        t = torch.ConstantIntOp(1).result
        print(t.type)
        # i = IntegerType.get_unsigned(32)
        # print(i)
        tt = Torch_IntType(t)
        print(tt)
        # print(module)
        # vt = from_numpy(np.random.rand(10, 10))
        # z = vt + vt
        print(module)
        # sizes = parse_sizes_from_tensor_type_str(z)
        # print(sizes)
