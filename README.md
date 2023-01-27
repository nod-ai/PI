- [PI](#PI)
- [Installing](#installing)
- [Minimal example](#minimal-example)
- [Moderately interesting example](#moderately-interesting-example)
- [Torch-MLIR](#torch-mlir)

<p align="center">
    <img width="598" alt="image" src="https://user-images.githubusercontent.com/5657668/205545845-544fe701-79d5-43c1-beec-09763f22cc85.png">
</p>

# PI

Early days of a Python frontend for MLIR.

# Installing

Just 

```shell
pip install - requirements.txt 
pip install . --no-build-isolation
```

and you're good to go.

# Torch-MLIR

Preliminary support for the `torch-mlir` dialect is available:

```python
class MyConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        y = self.conv(x)
        z = y + y
        w = z * z
        return w
```

lowers to

```mlir
module {
  func.func private @simple_conv2d() -> !torch.vtensor {
    %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<1x3x32x32xf32>) : !torch.vtensor<[1,3,32,32],f32>
    %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.vtensor<[1],f32>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<1x3x3x3xf32>) : !torch.vtensor<[1,3,3,3],f32>
    %int1 = torch.constant.int 1
    %int1_0 = torch.constant.int 1
    %3 = torch.prim.ListConstruct %int1, %int1_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0 = torch.constant.int 0
    %int0_1 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0, %int0_1 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2 = torch.constant.int 1
    %int1_3 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int1_2, %int1_3 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_4 = torch.constant.int 1
    %6 = torch.aten.conv2d %0, %2, %1, %3, %4, %5, %int1_4 : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[1,3,3,3],f32>, !torch.vtensor<[1],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor
    %7 = "torch.constant.number"() {value = 1 : i64} : () -> !torch.number
    %8 = torch.aten.add.Tensor %6, %6, %7 : !torch.vtensor, !torch.vtensor, !torch.number -> !torch.vtensor
    %9 = torch.aten.mul.Tensor %8, %8 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
    return %9 : !torch.vtensor
  }
}
```

This is very rough right now; to get a rough idea of the current status check the [latest tests](https://github.com/nod-ai/PI/actions?query=workflow%3ATest++).

Currently, we're passing 561 out of 770 of Torch-MLIR's test-suite (`torch-mlir==20230104.708`).

# Build Wheel

```shell
pip install - requirements.txt 
pip wheel . --no-build-isolation --wheel-dir wheelhouse
```