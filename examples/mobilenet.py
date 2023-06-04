import pi
from pi import nn
from pi.mlir.compile import pipile
from pi.mlir.utils import annotate_args, TensorPlaceholder
from pi.models.mobilenetv3 import mobilenet_v3_large


class MyMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobile = mobilenet_v3_large()

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], pi.float32, True),
        ]
    )
    def forward(self, x):
        y = self.mobile(x)
        return y


test_module = MyMobileNet()
x = TensorPlaceholder((1, 3, 64, 64), pi.float32)
mlir_module = pipile(test_module, example_args=(x,))
print(mlir_module.operation.get_asm(large_elements_limit=10))
