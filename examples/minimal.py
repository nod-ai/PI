import pi
from pi import nn
from pi.mlir.utils import pipile, lower_pi_to_linalg
from pi.utils.annotations import annotate_args, TensorPlaceholder


class MyConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], pi.float32, True),
        ]
    )
    def forward(self, x):
        y = self.conv(x)
        z = y + y
        w = z * z
        return w


test_module = MyConv2d()
x = TensorPlaceholder((1, 3, 32, 32), pi.float32)
mlir_module = pipile(test_module, example_args=(x,))
print(mlir_module)
mlir_module = lower_pi_to_linalg(mlir_module)
print(mlir_module)
