from shark.compiler.compiler import mlir_trace
# TODO(max): need to figure out how to reload the module so that the bytecode gets run through pycc
# import matmul

mlir_module = mlir_trace("matmul.py")
print(mlir_module)
