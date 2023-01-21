import pi
from torch_mlir.dialects import torch
from torch_mlir.ir import IntegerType
import numpy as np

from pi.mlir_utils import mlir_cm
from pi.types_ import TorchInt, TorchValue, TorchListOfTorchTensorType

if __name__ == "__main__":
    with mlir_cm() as module:
        t = TorchValue(torch.ConstantIntOp(1).result)
        tt = TorchValue(torch.ConstantIntOp(2).result)
        print(t)
        print(t.type)
        l = torch.PrimListConstructOp(
            TorchListOfTorchTensorType(dtype=t.type), [t, tt]
        ).result
        print(l)
        ll = TorchValue(l)
        print(ll)
        print(ll.type)
