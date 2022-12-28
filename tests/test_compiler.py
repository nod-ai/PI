import os.path

from shark.compiler.compiler import mlir_trace
# TODO(max): need to figure out how to reload the module so that the bytecode gets run through pycc


mlir_module = mlir_trace(os.path.abspath("simple_kernels.py"))
print(mlir_module)

mlir_module = mlir_trace(os.path.abspath("nn_module.py"))
print(mlir_module)
