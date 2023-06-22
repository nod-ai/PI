TORCH_MLIR_INSTALL_DIR=$PWD/../../torch_mlir_install/torch_mlir_install

$TORCH_MLIR_INSTALL_DIR/bin/llvm-tblgen -dump-json \
  -I $TORCH_MLIR_INSTALL_DIR/include \
  $TORCH_MLIR_INSTALL_DIR/include/torch-mlir/Dialect/Torch/IR/TorchOps.td \
  -o torch.json

PYTORCH_INSTALL_DIR=$(python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent)")
cp ${PYTORCH_INSTALL_DIR}/_C/__init__.pyi .