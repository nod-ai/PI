import pi
from pi import nn
from pi.mlir.utils import pipile
from pi.utils.annotations import annotate_args
from pi.models.unet import UNet2DConditionModel


class MyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel()

    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], pi.float32, True),
        ]
    )
    def forward(self, x):
        y = self.resnet(x)
        return y


test_module = MyUNet()
x = pi.randn((1, 3, 64, 64))
mlir_module = pipile(test_module, example_args=(x,))
print(mlir_module)
