import pi
from pi import nn
from pi.mlir.compile import pipile
from pi.mlir.utils import TensorPlaceholder
from pi.models.resnet import resnet18


class MyResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18()

    def forward(self, x):
        y = self.resnet(x)
        return y


test_module = MyResNet18()
x = TensorPlaceholder((1, 3, 64, 64), pi.float32)
mlir_module = pipile(test_module, example_args=(x,))
print(mlir_module.operation.get_asm(large_elements_limit=10))
