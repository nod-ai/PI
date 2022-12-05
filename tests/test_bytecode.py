from shark._mlir_libs._mlir.ir import F32Type
from shark.dialects import linalg
from shark.compiler.byte_code import mlir_compile


@mlir_compile
def build_mlir_module():
    f32 = F32Type.get()
    for i in range(10):
        tens = linalg.InitTensorOp([3, 4], f32)
    return tens


tens, module = build_mlir_module()
print(module)
