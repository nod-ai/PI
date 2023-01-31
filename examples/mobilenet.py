import pi
from pi import nn
from pi.mlir.utils import pipile
from pi.utils.annotations import annotate_args
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
x = pi.randn((1, 3, 64, 64))
mlir_module = pipile(test_module, example_args=(x,))
print(mlir_module)
