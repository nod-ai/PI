import test_numpy
from shark.compiler.compiler import mlir_compile

mlir_module = mlir_compile(test_numpy)
