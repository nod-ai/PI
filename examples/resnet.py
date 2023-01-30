import pi
from pi import nn
from pi.mlir.utils import pipile
from pi.utils.annotations import annotate_args
from pi.models.resnet import resnet18


class MyResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18()

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], pi.float32, True),
        ]
    )
    def forward(self, x):
        y = self.resnet(x)
        return y


test_module = MyResNet18()
x = pi.randn((1, 3, 64, 64))
mlir_module = pipile(test_module, example_args=(x,))
print(mlir_module)
