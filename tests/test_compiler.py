from shark.compiler.compiler import mlir_compile, mlir_bytecode_xpython_compile, mlir_bytecode_pytype_compile
from test_numpy import test_mat_mul

mlir_module = mlir_bytecode_pytype_compile(test_mat_mul)
print(mlir_module)
