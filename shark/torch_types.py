from shark._mlir import _TorchIntType
from torch_mlir.ir import Type as MLIRType


class TorchIntType(_TorchIntType):
    def __init__(self, type: MLIRType):
        super(TorchIntType, self).__init__(type._CAPIPtr)
