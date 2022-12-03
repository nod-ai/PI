from shark.compiler.compiler import mlir_compile

# from test_double_for import test_double_for
from test_numpy import test_double_for

# from test_single_for import  test_single_for

# print(mlir_compile(test_kernel.test_single_for))

print(mlir_compile(test_double_for))
