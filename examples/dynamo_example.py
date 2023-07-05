import numpy as np

from pi import nn
import pi
from pi.mlir.compile import pipile
from pi.utils import TensorPlaceholder, mlir_mod_ctx, tensor_op, Torch_IntValue
from pi.utils.ast_rewrite import rewrite_ast_callback
from pi.utils.dynamo import dynamo


def foo():
    x = Torch_IntValue(5)
    y = int(x)
    return y


with mlir_mod_ctx(), dynamo(rewrite_ast_callback):
    z = foo()

print(z)


class MyConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        y = self.conv(x)
        z = y + y
        w = z * z
        return w


test_module = MyConv2d()

with mlir_mod_ctx() as module:
    t = tensor_op(np.random.randn(1, 3, 32, 32))
    with dynamo(rewrite_ast_callback):
        z = test_module(t)
print(module)


class MyIntModule(nn.Module):
    def forward(self, x):
        y = int(x)
        return y


test_module = MyIntModule()
t = TensorPlaceholder((1,), dtype_=pi.int64)
module = pipile(test_module, example_args=(t,))
print(module)
