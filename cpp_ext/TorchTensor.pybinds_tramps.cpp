
// __abs__(self) -> Tensor
// aten::abs : (Tensor) -> (Tensor)
PyAnyTorchTensorValue __abs__(const PyAnyTorchTensorValue &self) {
  return abs(self);
}

// __div__(self, other Any) -> Tensor
// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __div__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return div(self, other);
}

// __div__(self, other Any) -> Tensor
// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __div__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return div(self, other);
}

// __eq__(self, other Any) -> Tensor
// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __eq__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return eq(self, other);
}

// __eq__(self, other Any) -> Tensor
// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __eq__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return eq(self, other);
}

// __ge__(self, other Any) -> Tensor
// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __ge__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return ge(self, other);
}

// __ge__(self, other Any) -> Tensor
// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __ge__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return ge(self, other);
}

// __gt__(self, other Any) -> Tensor
// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __gt__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return gt(self, other);
}

// __gt__(self, other Any) -> Tensor
// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __gt__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return gt(self, other);
}

// __le__(self, other Any) -> Tensor
// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __le__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return le(self, other);
}

// __le__(self, other Any) -> Tensor
// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __le__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return le(self, other);
}

// __lt__(self, other Any) -> Tensor
// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __lt__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return lt(self, other);
}

// __lt__(self, other Any) -> Tensor
// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __lt__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return lt(self, other);
}

// __matmul__(self, other Any) -> Tensor
// aten::matmul : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __matmul__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return matmul(self, other);
}

// __mul__(self, other Any) -> Tensor
// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __mul__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return mul(self, other);
}

// __mul__(self, other Any) -> Tensor
// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __mul__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return mul(self, other);
}

// __ne__(self, other Any) -> Tensor
// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __ne__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return ne(self, other);
}

// __ne__(self, other Any) -> Tensor
// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __ne__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return ne(self, other);
}

// __neg__(self) -> Tensor
// aten::neg : (Tensor) -> (Tensor)
PyAnyTorchTensorValue __neg__(const PyAnyTorchTensorValue &self) {
  return neg(self);
}
