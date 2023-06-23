
// __abs__(self) -> Tensor
// aten::abs : (Tensor) -> (Tensor)
PyAnyTorchTensorValue __abs__(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  return abs(self, loc, ip);
}

// __div__(self, other: Any) -> Tensor
// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __div__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return div(self, other, loc, ip);
}

// __div__(self, other: Any) -> Tensor
// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __div__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return div(self, other, loc, ip);
}

// __eq__(self, other: Any) -> Tensor
// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __eq__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return eq(self, other, loc, ip);
}

// __eq__(self, other: Any) -> Tensor
// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __eq__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return eq(self, other, loc, ip);
}

// __ge__(self, other: Any) -> Tensor
// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __ge__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return ge(self, other, loc, ip);
}

// __ge__(self, other: Any) -> Tensor
// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __ge__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return ge(self, other, loc, ip);
}

// __gt__(self, other: Any) -> Tensor
// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __gt__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return gt(self, other, loc, ip);
}

// __gt__(self, other: Any) -> Tensor
// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __gt__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return gt(self, other, loc, ip);
}

// __le__(self, other: Any) -> Tensor
// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __le__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return le(self, other, loc, ip);
}

// __le__(self, other: Any) -> Tensor
// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __le__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return le(self, other, loc, ip);
}

// __lt__(self, other: Any) -> Tensor
// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __lt__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return lt(self, other, loc, ip);
}

// __lt__(self, other: Any) -> Tensor
// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __lt__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return lt(self, other, loc, ip);
}

// __matmul__(self, other: Any) -> Tensor
// aten::matmul : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __matmul__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return matmul(self, other, loc, ip);
}

// __mul__(self, other: Any) -> Tensor
// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __mul__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return mul(self, other, loc, ip);
}

// __mul__(self, other: Any) -> Tensor
// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __mul__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return mul(self, other, loc, ip);
}

// __ne__(self, other: Any) -> Tensor
// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue __ne__(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return ne(self, other, loc, ip);
}

// __ne__(self, other: Any) -> Tensor
// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __ne__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  return ne(self, other, loc, ip);
}

// __neg__(self) -> Tensor
// aten::neg : (Tensor) -> (Tensor)
PyAnyTorchTensorValue __neg__(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  return neg(self, loc, ip);
}
