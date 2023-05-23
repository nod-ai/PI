
// __abs__(self) -> Tensor
// aten::abs : (Tensor) -> (Tensor)
py::object __abs__(const mlir::torch::PyAnyTorchTensorValue &self) {
  return mlir::torch::abs(self);
}

// __div__(self, other Any) -> Tensor
// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __div__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::div(self, other);
}

// __eq__(self, other Any) -> Tensor
// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __eq__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::eq(self, other);
}

// __ge__(self, other Any) -> Tensor
// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __ge__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::ge(self, other);
}

// __gt__(self, other Any) -> Tensor
// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __gt__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::gt(self, other);
}

// __le__(self, other Any) -> Tensor
// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __le__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::le(self, other);
}

// __lt__(self, other Any) -> Tensor
// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __lt__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::lt(self, other);
}

// __matmul__(self, other Any) -> Tensor
// aten::matmul : (Tensor, Tensor) -> (Tensor)
py::object __matmul__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::matmul(self, other);
}

// __mul__(self, other Any) -> Tensor
// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __mul__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::mul(self, other);
}

// __ne__(self, other Any) -> Tensor
// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __ne__(const mlir::torch::PyAnyTorchTensorValue &self, const mlir::torch::PyAnyTorchTensorValue &other) {
  return mlir::torch::ne(self, other);
}

// __neg__(self) -> Tensor
// aten::neg : (Tensor) -> (Tensor)
py::object __neg__(const mlir::torch::PyAnyTorchTensorValue &self) {
  return mlir::torch::neg(self);
}
