TORCH_MLIR_INSTALL_DIR=/Users/mlevental/dev_projects/PI/torch_mlir_install/torch_mlir_install

$TORCH_MLIR_INSTALL_DIR/bin/llvm-tblgen -dump-json \
  -I $TORCH_MLIR_INSTALL_DIR/include \
  $TORCH_MLIR_INSTALL_DIR/include/torch-mlir/Dialect/Torch/IR/TorchOps.td \
  -o torch.json
