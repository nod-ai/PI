import pi
from pi import nn
from pi.mlir.compile import pipile
from pi.mlir.utils import TensorPlaceholder


class RamirosModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)

    def forward(self, arg0, arg1):
        new_zeros = pi.ops.aten.new_zeros(arg0, [5, 4], pin_memory=False)
        slice_3 = pi.ops.aten.slice(new_zeros, 0, 0, 5)
        slice_4 = pi.ops.aten.slice(slice_3, 1, 1, 5)
        copy_ = pi.ops.aten.copy_(slice_4, arg1)
        return copy_


test_module = RamirosModel()
x = TensorPlaceholder((1, 3, 32, 32), pi.float32)
mlir_module = pipile(test_module, example_args=(x, x))
print(mlir_module)
