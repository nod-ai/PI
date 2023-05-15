
// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
py::object where(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &condition, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::where(condition, self, other);
}
