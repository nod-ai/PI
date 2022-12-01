from shark.compiler.compiler import mlir_compile

import test_kernel

print(mlir_compile(test_kernel.test_kernel))

