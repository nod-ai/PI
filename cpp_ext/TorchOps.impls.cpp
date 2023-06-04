
// aten::abs : (Tensor) -> (Tensor)
PyAnyTorchTensorValue abs(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.abs").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::abs_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue abs_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.abs_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue adaptive_avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size) {
  return PyGlobals::get().lookupOperationClass("torch.aten.adaptive_avg_pool2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, output_size).cast<PyAnyTorchTensorValue>();
}

// aten::add.float_int : (float, int) -> (float)
PyTorch_FloatValue add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add.float_int").value()(a, b).cast<PyTorch_FloatValue>();
}

// aten::add.int : (int, int) -> (int)
PyTorch_IntValue add(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add.int").value()(a, b).cast<PyTorch_IntValue>();
}

// aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue add(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::add.str : (str, str) -> (str)
PyTorch_StringValue add(const PyTorch_StringValue &a, const PyTorch_StringValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add.str").value()(a, b).cast<PyTorch_StringValue>();
}

// aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue add(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::add_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue add_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::add_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue add_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcdiv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.addcdiv").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, tensor1, tensor2, value).cast<PyAnyTorchTensorValue>();
}

// aten::addcdiv_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcdiv_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.addcdiv_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, tensor1, tensor2, value).cast<PyAnyTorchTensorValue>();
}

// aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.addcmul").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, tensor1, tensor2, value).cast<PyAnyTorchTensorValue>();
}

// aten::addcmul_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcmul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.addcmul_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, tensor1, tensor2, value).cast<PyAnyTorchTensorValue>();
}

// aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue addmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat1, const PyAnyTorchTensorValue &mat2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.addmm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mat1, mat2, beta, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::alias_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue alias_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.alias_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::all.bool : (bool[]) -> (bool)
PyTorch_BoolValue all(const PyAnyTorchListOfTorchBoolValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.all.bool").value()(self).cast<PyTorch_BoolValue>();
}

// aten::all : (Tensor) -> (Tensor)
PyAnyTorchTensorValue all(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.all").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::amax : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue amax(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.amax").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::any.bool : (bool[]) -> (bool)
PyTorch_BoolValue any(const PyAnyTorchListOfTorchBoolValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.any.bool").value()(self).cast<PyTorch_BoolValue>();
}

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue any(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.any.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::any : (Tensor) -> (Tensor)
PyAnyTorchTensorValue any(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.any").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.arange").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), end, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.arange.start").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), start, end, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::arange.start_out : (Scalar, Scalar, Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchTensorValue &out) {
  return PyGlobals::get().lookupOperationClass("torch.aten.arange.start_out").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), start, end, step, out).cast<PyAnyTorchTensorValue>();
}

// aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.arange.start_step").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), start, end, step, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
PyAnyTorchTensorValue argmax(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.argmax").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
PyAnyTorchTensorValue as_strided_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset) {
  return PyGlobals::get().lookupOperationClass("torch.aten.as_strided_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride, storage_offset).cast<PyAnyTorchTensorValue>();
}

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
PyAnyTorchTensorValue as_strided_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset) {
  return PyGlobals::get().lookupOperationClass("torch.aten.as_strided_scatter").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, size, stride, storage_offset).cast<PyAnyTorchTensorValue>();
}

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue atan2(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.atan2").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue atan2_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.atan2_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::atan : (Tensor) -> (Tensor)
PyAnyTorchTensorValue atan(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.atan").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::atan_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue atan_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.atan_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override) {
  return PyGlobals::get().lookupOperationClass("torch.aten.avg_pool2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override).cast<PyAnyTorchTensorValue>();
}

// aten::baddbmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue baddbmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.baddbmm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, batch1, batch2, beta, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::baddbmm_ : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue baddbmm_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.baddbmm_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, batch1, batch2, beta, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)
PyAnyTorchTensorValue batch_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enabled) {
  return PyGlobals::get().lookupOperationClass("torch.aten.batch_norm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled).cast<PyAnyTorchTensorValue>();
}

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, generator).cast<PyAnyTorchTensorValue>();
}

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli.p").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator).cast<PyAnyTorchTensorValue>();
}

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator).cast<PyAnyTorchTensorValue>();
}

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli_.float").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator).cast<PyAnyTorchTensorValue>();
}

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator).cast<PyAnyTorchTensorValue>();
}

// aten::bincount : (Tensor, Tensor?, int) -> (Tensor)
PyAnyTorchTensorValue bincount(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &weights, const PyTorch_IntValue &minlength) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bincount").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, weights, minlength).cast<PyAnyTorchTensorValue>();
}

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_and.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_and_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::bitwise_not : (Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_not(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_not").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::bitwise_not_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_not_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_not_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_or.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_or_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_xor.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_xor_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::bmm : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bmm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mat2).cast<PyAnyTorchTensorValue>();
}

// aten::Bool.float : (float) -> (bool)
PyTorch_BoolValue Bool(const PyTorch_FloatValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Bool.float").value()(a).cast<PyTorch_BoolValue>();
}

// aten::Bool.int : (int) -> (bool)
PyTorch_BoolValue Bool(const PyTorch_IntValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Bool.int").value()(a).cast<PyTorch_BoolValue>();
}

// aten::Bool.Tensor : (Tensor) -> (bool)
PyTorch_BoolValue Bool(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Bool.Tensor").value()(a).cast<PyTorch_BoolValue>();
}

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue broadcast_to(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  return PyGlobals::get().lookupOperationClass("torch.aten.broadcast_to").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size).cast<PyAnyTorchTensorValue>();
}

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue bucketize(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bucketize.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, boundaries, out_int32, right).cast<PyAnyTorchTensorValue>();
}

// aten::cat : (Tensor[], int) -> (Tensor)
PyAnyTorchTensorValue cat(const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cat").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), tensors, dim).cast<PyAnyTorchTensorValue>();
}

// aten::ceil.float : (float) -> (int)
PyTorch_IntValue ceil(const PyTorch_FloatValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ceil.float").value()(a).cast<PyTorch_IntValue>();
}

// aten::ceil : (Tensor) -> (Tensor)
PyAnyTorchTensorValue ceil(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ceil").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::ceil_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue ceil_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ceil_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::clamp_max : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_max(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clamp_max").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, max).cast<PyAnyTorchTensorValue>();
}

// aten::clamp_max_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_max_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clamp_max_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, max).cast<PyAnyTorchTensorValue>();
}

// aten::clamp_min : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_min(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clamp_min").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, min).cast<PyAnyTorchTensorValue>();
}

// aten::clamp_min_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_min_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clamp_min_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, min).cast<PyAnyTorchTensorValue>();
}

// aten::clamp : (Tensor, Scalar?, Scalar?) -> (Tensor)
PyAnyTorchTensorValue clamp(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clamp").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, min, max).cast<PyAnyTorchTensorValue>();
}

// aten::clamp.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
PyAnyTorchTensorValue clamp(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clamp.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, min, max).cast<PyAnyTorchTensorValue>();
}

// aten::clamp_ : (Tensor, Scalar?, Scalar?) -> (Tensor)
PyAnyTorchTensorValue clamp_(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clamp_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, min, max).cast<PyAnyTorchTensorValue>();
}

// aten::clamp_.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
PyAnyTorchTensorValue clamp_(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clamp_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, min, max).cast<PyAnyTorchTensorValue>();
}

// aten::clone : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue clone(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clone").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)
PyAnyTorchTensorValue constant_pad_nd(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.constant_pad_nd").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, pad__, value).cast<PyAnyTorchTensorValue>();
}

// aten::contiguous : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue contiguous(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.contiguous").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)
PyAnyTorchTensorValue conv2d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_IntValue &groups) {
  return PyGlobals::get().lookupOperationClass("torch.aten.conv2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias, stride, padding, dilation, groups).cast<PyAnyTorchTensorValue>();
}

// aten::conv_transpose1d : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose1d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) {
  return PyGlobals::get().lookupOperationClass("torch.aten.conv_transpose1d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias, stride, padding, output_padding, groups, dilation).cast<PyAnyTorchTensorValue>();
}

// aten::conv_transpose2d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose2d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) {
  return PyGlobals::get().lookupOperationClass("torch.aten.conv_transpose2d.input").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias, stride, padding, output_padding, groups, dilation).cast<PyAnyTorchTensorValue>();
}

// aten::conv_transpose3d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose3d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) {
  return PyGlobals::get().lookupOperationClass("torch.aten.conv_transpose3d.input").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias, stride, padding, output_padding, groups, dilation).cast<PyAnyTorchTensorValue>();
}

// aten::convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)
PyAnyTorchTensorValue convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups) {
  return PyGlobals::get().lookupOperationClass("torch.aten.convolution").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups).cast<PyAnyTorchTensorValue>();
}

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue copy(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking) {
  return PyGlobals::get().lookupOperationClass("torch.aten.copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, non_blocking).cast<PyAnyTorchTensorValue>();
}

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue copy_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking) {
  return PyGlobals::get().lookupOperationClass("torch.aten.copy_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, non_blocking).cast<PyAnyTorchTensorValue>();
}

// aten::cos : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cos(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cos").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::cos_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cos_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cos_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::cpu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cpu(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cpu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::cross_entropy_loss : (Tensor, Tensor, Tensor?, int, int, float) -> (Tensor)
PyAnyTorchTensorValue cross_entropy_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyTorch_FloatValue &label_smoothing) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cross_entropy_loss").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, target, weight, reduction, ignore_index, label_smoothing).cast<PyAnyTorchTensorValue>();
}

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue cumsum(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cumsum").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
void Delete(const PyTorch_DictValue &self, const PyTorch_StringValue &key) {
  PyGlobals::get().lookupOperationClass("torch.aten.Delete.Dict_str").value()(self, key);
}

// aten::detach_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue detach_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.detach_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::detach : (Tensor) -> (Tensor)
PyAnyTorchTensorValue detach(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.detach").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue diagonal_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2) {
  return PyGlobals::get().lookupOperationClass("torch.aten.diagonal_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, offset, dim1, dim2).cast<PyAnyTorchTensorValue>();
}

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue diagonal_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2) {
  return PyGlobals::get().lookupOperationClass("torch.aten.diagonal_scatter").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, offset, dim1, dim2).cast<PyAnyTorchTensorValue>();
}

// aten::dim : (Tensor) -> (int)
PyTorch_IntValue dim(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.dim").value()(self).cast<PyTorch_IntValue>();
}

// aten::div.float : (float, float) -> (float)
PyTorch_FloatValue div(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.float").value()(a, b).cast<PyTorch_FloatValue>();
}

// aten::div.int : (int, int) -> (float)
PyTorch_FloatValue div(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.int").value()(a, b).cast<PyTorch_FloatValue>();
}

// aten::div : (Scalar, Scalar) -> (float)
PyTorch_FloatValue div(const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div").value()(a, b).cast<PyTorch_FloatValue>();
}

// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.Tensor_mode").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, rounding_mode).cast<PyAnyTorchTensorValue>();
}

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::div_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div_.Tensor_mode").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, rounding_mode).cast<PyAnyTorchTensorValue>();
}

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::dropout : (Tensor, float, bool) -> (Tensor)
PyAnyTorchTensorValue dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train) {
  return PyGlobals::get().lookupOperationClass("torch.aten.dropout").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, p, train).cast<PyAnyTorchTensorValue>();
}

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
PyAnyTorchTensorValue dropout_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train) {
  return PyGlobals::get().lookupOperationClass("torch.aten.dropout_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, train).cast<PyAnyTorchTensorValue>();
}

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
PyAnyTorchTensorValue embedding_dense_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq) {
  return PyGlobals::get().lookupOperationClass("torch.aten.embedding_dense_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, indices, num_weights, padding_idx, scale_grad_by_freq).cast<PyAnyTorchTensorValue>();
}

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
PyAnyTorchTensorValue embedding(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse) {
  return PyGlobals::get().lookupOperationClass("torch.aten.embedding").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), weight, indices, padding_idx, scale_grad_by_freq, sparse).cast<PyAnyTorchTensorValue>();
}

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue empty_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.empty_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue empty(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.empty.memory_format").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::eq.device : (Device, Device) -> (bool)
PyTorch_BoolValue eq(const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.device").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::eq.float : (float, float) -> (bool)
PyTorch_BoolValue eq(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.float").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::eq.int_list : (int[], int[]) -> (bool)
PyTorch_BoolValue eq(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.int_list").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::eq.int : (int, int) -> (bool)
PyTorch_BoolValue eq(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue eq(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::eq.str : (str, str) -> (bool)
PyTorch_BoolValue eq(const PyTorch_StringValue &a, const PyTorch_StringValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.str").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue eq(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::eq_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::erf : (Tensor) -> (Tensor)
PyAnyTorchTensorValue erf(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.erf").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::erf_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue erf_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.erf_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::exp : (Tensor) -> (Tensor)
PyAnyTorchTensorValue exp(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.exp").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::exp_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue exp_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.exp_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue expand_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expand_as").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue expand_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expand_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, implicit).cast<PyAnyTorchTensorValue>();
}

// aten::expand : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue expand(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expand").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, implicit).cast<PyAnyTorchTensorValue>();
}

// aten::expm1 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue expm1(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expm1").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::expm1_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue expm1_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expm1_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
PyAnyTorchTensorValue fft_fft(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fft_fft").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, n, dim, norm).cast<PyAnyTorchTensorValue>();
}

// aten::fill.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fill(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fill.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, value).cast<PyAnyTorchTensorValue>();
}

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fill.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, value).cast<PyAnyTorchTensorValue>();
}

// aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fill_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, value).cast<PyAnyTorchTensorValue>();
}

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fill_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, value).cast<PyAnyTorchTensorValue>();
}

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue flatten(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.flatten.using_ints").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, start_dim, end_dim).cast<PyAnyTorchTensorValue>();
}

// aten::flip : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue flip(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  return PyGlobals::get().lookupOperationClass("torch.aten.flip").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims).cast<PyAnyTorchTensorValue>();
}

// aten::FloatImplicit : (Tensor) -> (float)
PyTorch_FloatValue FloatImplicit(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.FloatImplicit").value()(a).cast<PyTorch_FloatValue>();
}

// aten::Float.Scalar : (Scalar) -> (float)
PyTorch_FloatValue Float(const PyAnyTorchScalarValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Float.Scalar").value()(a).cast<PyTorch_FloatValue>();
}

// aten::Float.str : (str) -> (float)
PyTorch_FloatValue Float(const PyTorch_StringValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Float.str").value()(a).cast<PyTorch_FloatValue>();
}

// aten::Float.Tensor : (Tensor) -> (float)
PyTorch_FloatValue Float(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Float.Tensor").value()(a).cast<PyTorch_FloatValue>();
}

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floor_divide").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floor_divide.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::floor : (Tensor) -> (Tensor)
PyAnyTorchTensorValue floor(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::floor_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue floor_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floor_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::floordiv.int : (int, int) -> (int)
PyTorch_IntValue floordiv(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floordiv.int").value()(a, b).cast<PyTorch_IntValue>();
}

// aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fmod(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fmod.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::fmod_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fmod_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fmod_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue frobenius_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.frobenius_norm.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::full_like : (Tensor, Scalar, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue full_like(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.full_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, fill_value, dtype, layout, device, pin_memory, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::full : (int[], Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue full(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.full").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, fill_value, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue gather(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gather").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, sparse_grad).cast<PyAnyTorchTensorValue>();
}

// aten::ge.float_int : (float, int) -> (bool)
PyTorch_BoolValue ge(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.float_int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::ge.float : (float, float) -> (bool)
PyTorch_BoolValue ge(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.float").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::ge.int : (int, int) -> (bool)
PyTorch_BoolValue ge(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ge(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ge(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::ge_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
PyAnyTorchTensorValue gelu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gelu_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, approximate).cast<PyAnyTorchTensorValue>();
}

// aten::gelu : (Tensor, str) -> (Tensor)
PyAnyTorchTensorValue gelu(const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gelu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, approximate).cast<PyAnyTorchTensorValue>();
}

// aten::gt.float_int : (float, int) -> (bool)
PyTorch_BoolValue gt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.float_int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::gt.float : (float, float) -> (bool)
PyTorch_BoolValue gt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.float").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::gt.int : (int, int) -> (bool)
PyTorch_BoolValue gt(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue gt(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue gt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::gt_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::hardsigmoid : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardsigmoid(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardsigmoid").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardsigmoid_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardsigmoid_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::hardswish : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardswish(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardswish").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::hardswish_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardswish_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardswish_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::hardtanh_backward : (Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardtanh_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, min_val, max_val).cast<PyAnyTorchTensorValue>();
}

// aten::hardtanh : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardtanh").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, min_val, max_val).cast<PyAnyTorchTensorValue>();
}

// aten::hardtanh_ : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardtanh_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, min_val, max_val).cast<PyAnyTorchTensorValue>();
}

// aten::imag : (Tensor) -> (Tensor)
PyAnyTorchTensorValue imag(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.imag").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::index_put.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate) {
  return PyGlobals::get().lookupOperationClass("torch.aten.index_put.hacked_twin").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, indices, values, accumulate).cast<PyAnyTorchTensorValue>();
}

// aten::index_put_.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate) {
  return PyGlobals::get().lookupOperationClass("torch.aten.index_put_.hacked_twin").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, indices, values, accumulate).cast<PyAnyTorchTensorValue>();
}

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue index_select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index) {
  return PyGlobals::get().lookupOperationClass("torch.aten.index_select").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index).cast<PyAnyTorchTensorValue>();
}

// aten::index.Tensor_hacked_twin : (Tensor, Tensor[]) -> (Tensor)
PyAnyTorchTensorValue index(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices) {
  return PyGlobals::get().lookupOperationClass("torch.aten.index.Tensor_hacked_twin").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, indices).cast<PyAnyTorchTensorValue>();
}

// aten::Int.bool : (bool) -> (int)
PyTorch_IntValue Int(const PyTorch_BoolValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Int.bool").value()(a).cast<PyTorch_IntValue>();
}

// aten::Int.float : (float) -> (int)
PyTorch_IntValue Int(const PyTorch_FloatValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Int.float").value()(a).cast<PyTorch_IntValue>();
}

// aten::IntImplicit : (Tensor) -> (int)
PyTorch_IntValue IntImplicit(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.IntImplicit").value()(a).cast<PyTorch_IntValue>();
}

// aten::Int.Scalar : (Scalar) -> (int)
PyTorch_IntValue Int(const PyAnyTorchScalarValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Int.Scalar").value()(a).cast<PyTorch_IntValue>();
}

// aten::Int.Tensor : (Tensor) -> (int)
PyTorch_IntValue Int(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Int.Tensor").value()(a).cast<PyTorch_IntValue>();
}

// aten::is_floating_point : (Tensor) -> (bool)
PyTorch_BoolValue is_floating_point(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.is_floating_point").value()(self).cast<PyTorch_BoolValue>();
}

// aten::join : (str, str[]) -> (str)
PyTorch_StringValue join(const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values) {
  return PyGlobals::get().lookupOperationClass("torch.aten.join").value()(self, values).cast<PyTorch_StringValue>();
}

// aten::keys.str : (Dict(str, t)) -> (str[])
PyAnyTorchListOfTorchStringValue keys(const PyTorch_DictValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.keys.str").value()(PyAnyTorchListOfTorchStringType(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchListOfTorchStringValue>();
}

// aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)
PyAnyTorchTensorValue layer_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enable) {
  return PyGlobals::get().lookupOperationClass("torch.aten.layer_norm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, normalized_shape, weight, bias, eps, cudnn_enable).cast<PyAnyTorchTensorValue>();
}

// aten::le.int : (int, int) -> (bool)
PyTorch_BoolValue le(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.le.int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue le(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.le.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue le(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.le.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::le_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue le_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.le_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue le_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.le_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::leaky_relu_backward : (Tensor, Tensor, Scalar, bool) -> (Tensor)
PyAnyTorchTensorValue leaky_relu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, const PyTorch_BoolValue &self_is_result) {
  return PyGlobals::get().lookupOperationClass("torch.aten.leaky_relu_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, negative_slope, self_is_result).cast<PyAnyTorchTensorValue>();
}

// aten::leaky_relu : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue leaky_relu(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope) {
  return PyGlobals::get().lookupOperationClass("torch.aten.leaky_relu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, negative_slope).cast<PyAnyTorchTensorValue>();
}

// aten::leaky_relu_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue leaky_relu_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope) {
  return PyGlobals::get().lookupOperationClass("torch.aten.leaky_relu_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, negative_slope).cast<PyAnyTorchTensorValue>();
}

// aten::len.str : (str) -> (int)
PyTorch_IntValue len(const PyTorch_StringValue &s) {
  return PyGlobals::get().lookupOperationClass("torch.aten.len.str").value()(s).cast<PyTorch_IntValue>();
}

// aten::len.t : (t[]) -> (int)
PyTorch_IntValue len(const PyAnyTorchListValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.len.t").value()(a).cast<PyTorch_IntValue>();
}

// aten::len.Tensor : (Tensor) -> (int)
PyTorch_IntValue len(const PyAnyTorchTensorValue &t) {
  return PyGlobals::get().lookupOperationClass("torch.aten.len.Tensor").value()(t).cast<PyTorch_IntValue>();
}

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lerp(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lerp.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, end, weight).cast<PyAnyTorchTensorValue>();
}

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lerp_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lerp_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, end, weight).cast<PyAnyTorchTensorValue>();
}

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue lift_fresh_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lift_fresh_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::linalg_vector_norm : (Tensor, Scalar, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue linalg_vector_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &ord, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.linalg_vector_norm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, ord, dim, keepdim, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias) {
  return PyGlobals::get().lookupOperationClass("torch.aten.linear").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias).cast<PyAnyTorchTensorValue>();
}

// aten::log1p : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log1p(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log1p").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::log1p_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log1p_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log1p_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::log2 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log2(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log2").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::log2_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log2_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log2_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::log.int : (int) -> (float)
PyTorch_FloatValue log(const PyTorch_IntValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log.int").value()(a).cast<PyTorch_FloatValue>();
}

// aten::log : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log_softmax.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::log_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_and").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_and_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::logical_not : (Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_not(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_not").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::logical_not_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_not_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_not_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_or").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_or_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_xor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_xor_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue logsumexp(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logsumexp").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::lt.float_int : (float, int) -> (bool)
PyTorch_BoolValue lt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.float_int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::lt.float : (float, float) -> (bool)
PyTorch_BoolValue lt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.float").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::lt.int : (int, int) -> (bool)
PyTorch_BoolValue lt(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue lt(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::lt_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.masked_fill.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask, value).cast<PyAnyTorchTensorValue>();
}

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.masked_fill.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask, value).cast<PyAnyTorchTensorValue>();
}

// aten::masked_fill_.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.masked_fill_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask, value).cast<PyAnyTorchTensorValue>();
}

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.masked_fill_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask, value).cast<PyAnyTorchTensorValue>();
}

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_select(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask) {
  return PyGlobals::get().lookupOperationClass("torch.aten.masked_select").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask).cast<PyAnyTorchTensorValue>();
}

// aten::matmul : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue matmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.matmul").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::max : (Tensor) -> (Tensor)
PyAnyTorchTensorValue max(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.max").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
PyAnyTorchTensorValue max_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode) {
  return PyGlobals::get().lookupOperationClass("torch.aten.max_pool2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, kernel_size, stride, padding, dilation, ceil_mode).cast<PyAnyTorchTensorValue>();
}

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
PyAnyTorchTensorValue max_pool2d_with_indices_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices) {
  return PyGlobals::get().lookupOperationClass("torch.aten.max_pool2d_with_indices_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices).cast<PyAnyTorchTensorValue>();
}

// aten::maximum : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue maximum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.maximum").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::mean.dim : (Tensor, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mean.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::mean : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mean").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::minimum : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue minimum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.minimum").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::mish : (Tensor) -> (Tensor)
PyAnyTorchTensorValue mish(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mish").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::mm : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mat2).cast<PyAnyTorchTensorValue>();
}

// aten::movedim.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue movedim(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &source, const PyTorch_IntValue &destination) {
  return PyGlobals::get().lookupOperationClass("torch.aten.movedim.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, source, destination).cast<PyAnyTorchTensorValue>();
}

// aten::mse_loss_backward : (Tensor, Tensor, Tensor, int) -> (Tensor)
PyAnyTorchTensorValue mse_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mse_loss_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, target, reduction).cast<PyAnyTorchTensorValue>();
}

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
PyAnyTorchTensorValue mse_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mse_loss").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, target, reduction).cast<PyAnyTorchTensorValue>();
}

// aten::mul.float : (float, float) -> (float)
PyTorch_FloatValue mul(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul.float").value()(a, b).cast<PyTorch_FloatValue>();
}

// aten::mul.int : (int, int) -> (int)
PyTorch_IntValue mul(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul.int").value()(a, b).cast<PyTorch_IntValue>();
}

// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue mul(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::mul_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::mv : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mv").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, vec).cast<PyAnyTorchTensorValue>();
}

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue narrow(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length) {
  return PyGlobals::get().lookupOperationClass("torch.aten.narrow").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, length).cast<PyAnyTorchTensorValue>();
}

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
PyAnyTorchTensorValue native_dropout_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale) {
  return PyGlobals::get().lookupOperationClass("torch.aten.native_dropout_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, mask, scale).cast<PyAnyTorchTensorValue>();
}

// aten::ne.bool : (bool, bool) -> (bool)
PyTorch_BoolValue ne(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.bool").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::ne.float_int : (float, int) -> (bool)
PyTorch_BoolValue ne(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.float_int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::ne.int_list : (int[], int[]) -> (bool)
PyTorch_BoolValue ne(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.int_list").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::ne.int : (int, int) -> (bool)
PyTorch_BoolValue ne(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.int").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ne(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ne(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::ne_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::neg.float : (float) -> (float)
PyTorch_FloatValue neg(const PyTorch_FloatValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.neg.float").value()(a).cast<PyTorch_FloatValue>();
}

// aten::neg.int : (int) -> (int)
PyTorch_IntValue neg(const PyTorch_IntValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.neg.int").value()(a).cast<PyTorch_IntValue>();
}

// aten::neg : (Tensor) -> (Tensor)
PyAnyTorchTensorValue neg(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.neg").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::neg_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue neg_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.neg_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_empty(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.new_empty").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_empty_strided(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.new_empty_strided").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_ones(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.new_ones").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_zeros(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.new_zeros").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::nll_loss2d_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue nll_loss2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight) {
  return PyGlobals::get().lookupOperationClass("torch.aten.nll_loss2d_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, target, weight, reduction, ignore_index, total_weight).cast<PyAnyTorchTensorValue>();
}

// aten::nll_loss_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue nll_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight) {
  return PyGlobals::get().lookupOperationClass("torch.aten.nll_loss_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, target, weight, reduction, ignore_index, total_weight).cast<PyAnyTorchTensorValue>();
}

// aten::norm.ScalarOpt_dim : (Tensor, Scalar?, int[], bool) -> (Tensor)
PyAnyTorchTensorValue norm(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &p, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.norm.ScalarOpt_dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, dim, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::numel : (Tensor) -> (int)
PyTorch_IntValue numel(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.numel").value()(self).cast<PyTorch_IntValue>();
}

// aten::numpy_T : (Tensor) -> (Tensor)
PyAnyTorchTensorValue numpy_T(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.numpy_T").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::one_hot : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue one_hot(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &num_classes) {
  return PyGlobals::get().lookupOperationClass("torch.aten.one_hot").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, num_classes).cast<PyAnyTorchTensorValue>();
}

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue ones_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ones_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue ones(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ones").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
PyAnyTorchTensorValue pad(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.pad").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, pad__, mode, value).cast<PyAnyTorchTensorValue>();
}

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue permute_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  return PyGlobals::get().lookupOperationClass("torch.aten.permute_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims).cast<PyAnyTorchTensorValue>();
}

// aten::permute : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue permute(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  return PyGlobals::get().lookupOperationClass("torch.aten.permute").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims).cast<PyAnyTorchTensorValue>();
}

// aten::pow.int_float : (int, float) -> (float)
PyTorch_FloatValue pow(const PyTorch_IntValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.pow.int_float").value()(a, b).cast<PyTorch_FloatValue>();
}

// aten::pow.Scalar : (Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &exponent) {
  return PyGlobals::get().lookupOperationClass("torch.aten.pow.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, exponent).cast<PyAnyTorchTensorValue>();
}

// aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &exponent) {
  return PyGlobals::get().lookupOperationClass("torch.aten.pow.Tensor_Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, exponent).cast<PyAnyTorchTensorValue>();
}

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent) {
  return PyGlobals::get().lookupOperationClass("torch.aten.pow.Tensor_Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, exponent).cast<PyAnyTorchTensorValue>();
}

// aten::prelu : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue prelu(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight) {
  return PyGlobals::get().lookupOperationClass("torch.aten.prelu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, weight).cast<PyAnyTorchTensorValue>();
}

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue rand_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.rand_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randint(const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randint.low").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), low, high, size, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::randint : (int, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randint(const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randint").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), high, size, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randn.generator").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, generator, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue randn_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randn_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randn").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::real : (Tensor) -> (Tensor)
PyAnyTorchTensorValue real(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.real").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::reciprocal : (Tensor) -> (Tensor)
PyAnyTorchTensorValue reciprocal(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.reciprocal").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::reciprocal_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue reciprocal_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.reciprocal_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::relu6 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu6(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.relu6").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::relu6_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu6_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.relu6_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::relu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.relu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::relu_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.relu_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::remainder.int : (int, int) -> (int)
PyTorch_IntValue remainder(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.remainder.int").value()(a, b).cast<PyTorch_IntValue>();
}

// aten::remainder.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue remainder(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.remainder.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::repeat : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue repeat(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats) {
  return PyGlobals::get().lookupOperationClass("torch.aten.repeat").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, repeats).cast<PyAnyTorchTensorValue>();
}

// aten::reshape : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue reshape(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape) {
  return PyGlobals::get().lookupOperationClass("torch.aten.reshape").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, shape).cast<PyAnyTorchTensorValue>();
}

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
PyAnyTorchTensorValue resize_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.resize_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue roll(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims) {
  return PyGlobals::get().lookupOperationClass("torch.aten.roll").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, shifts, dims).cast<PyAnyTorchTensorValue>();
}

// aten::round : (Tensor) -> (Tensor)
PyAnyTorchTensorValue round(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.round").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::round_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue round_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.round_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::rsqrt : (Tensor) -> (Tensor)
PyAnyTorchTensorValue rsqrt(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.rsqrt").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::rsqrt_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue rsqrt_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.rsqrt_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue rsub(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.rsub.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::scaled_dot_product_attention : (Tensor, Tensor, Tensor, Tensor?, float, bool, float?) -> (Tensor)
PyAnyTorchTensorValue scaled_dot_product_attention(const PyAnyTorchTensorValue &query, const PyAnyTorchTensorValue &key, const PyAnyTorchTensorValue &value, const PyAnyTorchOptionalTensorValue &attn_mask, const PyTorch_FloatValue &dropout_p, const PyTorch_BoolValue &is_causal, const PyAnyTorchOptionalFloatValue &scale) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scaled_dot_product_attention").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), query, key, value, attn_mask, dropout_p, is_causal, scale).cast<PyAnyTorchTensorValue>();
}

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_add(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter_add").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src).cast<PyAnyTorchTensorValue>();
}

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_add_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter_add_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src).cast<PyAnyTorchTensorValue>();
}

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
PyAnyTorchTensorValue scatter_reduce(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter_reduce.two").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src, reduce, include_self).cast<PyAnyTorchTensorValue>();
}

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
PyAnyTorchTensorValue scatter_reduce_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter_reduce_.two").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src, reduce, include_self).cast<PyAnyTorchTensorValue>();
}

// aten::scatter.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter.src").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src).cast<PyAnyTorchTensorValue>();
}

// aten::scatter.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter.value").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, value).cast<PyAnyTorchTensorValue>();
}

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  return PyGlobals::get().lookupOperationClass("torch.aten.select_copy.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index).cast<PyAnyTorchTensorValue>();
}

// aten::select.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  return PyGlobals::get().lookupOperationClass("torch.aten.select.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index).cast<PyAnyTorchTensorValue>();
}

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  return PyGlobals::get().lookupOperationClass("torch.aten.select_scatter").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, dim, index).cast<PyAnyTorchTensorValue>();
}

// aten::sigmoid : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sigmoid(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sigmoid").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::sigmoid_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sigmoid_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sigmoid_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::silu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue silu(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.silu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::silu_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue silu_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.silu_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::sin : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sin(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sin").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::sin_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sin_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sin_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::size.int : (Tensor, int) -> (int)
PyTorch_IntValue size(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.size.int").value()(self, dim).cast<PyTorch_IntValue>();
}

// aten::size : (Tensor) -> (int[])
PyAnyTorchListOfTorchIntValue size(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.size").value()(PyAnyTorchListOfTorchIntType(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchListOfTorchIntValue>();
}

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.slice_copy.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, end, step).cast<PyAnyTorchTensorValue>();
}

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.slice_scatter").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, dim, start, end, step).cast<PyAnyTorchTensorValue>();
}

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.slice.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, end, step).cast<PyAnyTorchTensorValue>();
}

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.softmax.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::softplus : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue softplus(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &threshold__) {
  return PyGlobals::get().lookupOperationClass("torch.aten.softplus").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, beta, threshold__).cast<PyAnyTorchTensorValue>();
}

// aten::sort.int : (int[], bool) -> ()
void sort(const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse) {
  PyGlobals::get().lookupOperationClass("torch.aten.sort.int").value()(self, reverse);
}

// aten::sqrt.int : (int) -> (float)
PyTorch_FloatValue sqrt(const PyTorch_IntValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sqrt.int").value()(a).cast<PyTorch_FloatValue>();
}

// aten::sqrt : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sqrt(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sqrt").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::sqrt_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sqrt_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sqrt_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::square : (Tensor) -> (Tensor)
PyAnyTorchTensorValue square(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.square").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::square_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue square_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.square_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue squeeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.squeeze_copy.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim).cast<PyAnyTorchTensorValue>();
}

// aten::squeeze_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue squeeze_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.squeeze_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.squeeze.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim).cast<PyAnyTorchTensorValue>();
}

// aten::squeeze : (Tensor) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.squeeze").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::stack : (Tensor[], int) -> (Tensor)
PyAnyTorchTensorValue stack(const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.stack").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), tensors, dim).cast<PyAnyTorchTensorValue>();
}

// aten::std.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.std.correction").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, correction, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::std.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.std.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, unbiased, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::std : (Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased) {
  return PyGlobals::get().lookupOperationClass("torch.aten.std").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, unbiased).cast<PyAnyTorchTensorValue>();
}

// aten::sub.float : (float, float) -> (float)
PyTorch_FloatValue sub(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sub.float").value()(a, b).cast<PyTorch_FloatValue>();
}

// aten::sub.int : (int, int) -> (int)
PyTorch_IntValue sub(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sub.int").value()(a, b).cast<PyTorch_IntValue>();
}

// aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sub.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sub.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::sub_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sub_.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::sub_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sub_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, alpha).cast<PyAnyTorchTensorValue>();
}

// aten::sum.dim_IntList : (Tensor, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sum.dim_IntList").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::sum : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sum").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::t_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue t_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.t_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::t : (Tensor) -> (Tensor)
PyAnyTorchTensorValue t(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.t").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tanh_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output).cast<PyAnyTorchTensorValue>();
}

// aten::tanh : (Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tanh").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::tanh_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tanh_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tensor.bool").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad).cast<PyAnyTorchTensorValue>();
}

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tensor.float").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad).cast<PyAnyTorchTensorValue>();
}

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tensor.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad).cast<PyAnyTorchTensorValue>();
}

// aten::tensor : (t[], int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyAnyTorchListValue &data, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), data, dtype, device, requires_grad).cast<PyAnyTorchTensorValue>();
}

// aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__) {
  return PyGlobals::get().lookupOperationClass("torch.aten.threshold_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, threshold__).cast<PyAnyTorchTensorValue>();
}

// aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.threshold").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, threshold__, value).cast<PyAnyTorchTensorValue>();
}

// aten::threshold_ : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.threshold_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, threshold__, value).cast<PyAnyTorchTensorValue>();
}

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.device").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, device, dtype, non_blocking, copy, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.dtype_layout").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.dtype").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, non_blocking, copy, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.other").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, non_blocking, copy, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.prim_Device").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, device, dtype, non_blocking, copy).cast<PyAnyTorchTensorValue>();
}

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue transpose_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1) {
  return PyGlobals::get().lookupOperationClass("torch.aten.transpose_copy.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim0, dim1).cast<PyAnyTorchTensorValue>();
}

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue transpose(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1) {
  return PyGlobals::get().lookupOperationClass("torch.aten.transpose.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim0, dim1).cast<PyAnyTorchTensorValue>();
}

// aten::triu : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue triu(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal) {
  return PyGlobals::get().lookupOperationClass("torch.aten.triu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, diagonal).cast<PyAnyTorchTensorValue>();
}

// aten::triu_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue triu_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal) {
  return PyGlobals::get().lookupOperationClass("torch.aten.triu_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, diagonal).cast<PyAnyTorchTensorValue>();
}

// aten::type_as : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue type_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.type_as").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue unfold_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.unfold_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dimension, size, step).cast<PyAnyTorchTensorValue>();
}

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue uniform(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.uniform").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, from, to, generator).cast<PyAnyTorchTensorValue>();
}

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue uniform_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.uniform_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, from, to, generator).cast<PyAnyTorchTensorValue>();
}

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.unsqueeze_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim).cast<PyAnyTorchTensorValue>();
}

// aten::unsqueeze : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.unsqueeze").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim).cast<PyAnyTorchTensorValue>();
}

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.unsqueeze_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim).cast<PyAnyTorchTensorValue>();
}

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
PyAnyTorchTensorValue upsample_nearest2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w) {
  return PyGlobals::get().lookupOperationClass("torch.aten.upsample_nearest2d_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output_size, input_size, scales_h, scales_w).cast<PyAnyTorchTensorValue>();
}

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
PyAnyTorchTensorValue upsample_nearest2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w) {
  return PyGlobals::get().lookupOperationClass("torch.aten.upsample_nearest2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, output_size, scales_h, scales_w).cast<PyAnyTorchTensorValue>();
}

// aten::var.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.var.correction").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, correction, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::var.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.var.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, unbiased, keepdim).cast<PyAnyTorchTensorValue>();
}

// aten::var : (Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased) {
  return PyGlobals::get().lookupOperationClass("torch.aten.var").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, unbiased).cast<PyAnyTorchTensorValue>();
}

// aten::view_as_complex : (Tensor) -> (Tensor)
PyAnyTorchTensorValue view_as_complex(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.view_as_complex").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue view_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.view_copy.dtype").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype).cast<PyAnyTorchTensorValue>();
}

// aten::view_copy : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue view_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  return PyGlobals::get().lookupOperationClass("torch.aten.view_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size).cast<PyAnyTorchTensorValue>();
}

// aten::view : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  return PyGlobals::get().lookupOperationClass("torch.aten.view").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size).cast<PyAnyTorchTensorValue>();
}

// aten::where.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.where.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), condition, self, other).cast<PyAnyTorchTensorValue>();
}

// aten::where.ScalarOther : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.where.ScalarOther").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), condition, self, other).cast<PyAnyTorchTensorValue>();
}

// aten::where.ScalarSelf : (Tensor, Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.where.ScalarSelf").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), condition, self, other).cast<PyAnyTorchTensorValue>();
}

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.where.self").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), condition, self, other).cast<PyAnyTorchTensorValue>();
}

// aten::zero : (Tensor) -> (Tensor)
PyAnyTorchTensorValue zero(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.zero").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::zero_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue zero_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.zero_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue zeros_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.zeros_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue zeros(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.zeros").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory).cast<PyAnyTorchTensorValue>();
}

// aten::_convolution.deprecated : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled) {
  return PyGlobals::get().lookupOperationClass("torch.aten._convolution.deprecated").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled).cast<PyAnyTorchTensorValue>();
}

// aten::_convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, const PyTorch_BoolValue &allow_tf32) {
  return PyGlobals::get().lookupOperationClass("torch.aten._convolution").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32).cast<PyAnyTorchTensorValue>();
}

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue _log_softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten._log_softmax_backward_data").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output, dim, input_dtype).cast<PyAnyTorchTensorValue>();
}

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue _log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float) {
  return PyGlobals::get().lookupOperationClass("torch.aten._log_softmax").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, half_to_float).cast<PyAnyTorchTensorValue>();
}

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue _reshape_alias_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride) {
  return PyGlobals::get().lookupOperationClass("torch.aten._reshape_alias_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride).cast<PyAnyTorchTensorValue>();
}

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue _reshape_alias(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride) {
  return PyGlobals::get().lookupOperationClass("torch.aten._reshape_alias").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride).cast<PyAnyTorchTensorValue>();
}

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
PyAnyTorchTensorValue _shape_as_tensor(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten._shape_as_tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self).cast<PyAnyTorchTensorValue>();
}

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue _softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten._softmax_backward_data").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output, dim, input_dtype).cast<PyAnyTorchTensorValue>();
}

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue _softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float) {
  return PyGlobals::get().lookupOperationClass("torch.aten._softmax").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, half_to_float).cast<PyAnyTorchTensorValue>();
}

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue _to_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten._to_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, non_blocking, memory_format).cast<PyAnyTorchTensorValue>();
}

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue _unsafe_view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  return PyGlobals::get().lookupOperationClass("torch.aten._unsafe_view").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size).cast<PyAnyTorchTensorValue>();
}

// aten::__and__.bool : (bool, bool) -> (bool)
PyTorch_BoolValue __and__(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__and__.bool").value()(a, b).cast<PyTorch_BoolValue>();
}

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __and__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__and__.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other).cast<PyAnyTorchTensorValue>();
}

// aten::__contains__.int_list : (int[], int) -> (bool)
PyTorch_BoolValue __contains__(const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__contains__.int_list").value()(l, item).cast<PyTorch_BoolValue>();
}

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
PyTorch_BoolValue __contains__(const PyTorch_DictValue &dict, const PyTorch_StringValue &key) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__contains__.str").value()(dict, key).cast<PyTorch_BoolValue>();
}

// aten::__derive_index : (int, int, int) -> (int)
PyTorch_IntValue __derive_index(const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__derive_index").value()(index, start, step).cast<PyTorch_IntValue>();
}

// aten::__not__ : (bool) -> (bool)
PyTorch_BoolValue __not__(const PyTorch_BoolValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__not__").value()(self).cast<PyTorch_BoolValue>();
}

// aten::__range_length : (int, int, int) -> (int)
PyTorch_IntValue __range_length(const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__range_length").value()(lo, hi, step).cast<PyTorch_IntValue>();
}

// prim::device : (Tensor) -> (Device)
PyTorch_DeviceValue device(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.prim.device").value()(a).cast<PyTorch_DeviceValue>();
}

// prim::dtype : (Tensor) -> (int)
PyTorch_IntValue dtype(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.prim.dtype").value()(a).cast<PyTorch_IntValue>();
}

// prim::layout : (Tensor) -> (int)
PyTorch_IntValue layout(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.prim.layout").value()(a).cast<PyTorch_IntValue>();
}

// prim::max.int : (int, int) -> (int)
PyTorch_IntValue max(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.prim.max.int").value()(a, b).cast<PyTorch_IntValue>();
}

// prim::max.self_int : (int[]) -> (int)
PyTorch_IntValue max(const PyAnyTorchListOfTorchIntValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.prim.max.self_int").value()(self).cast<PyTorch_IntValue>();
}

// prim::min.int : (int, int) -> (int)
PyTorch_IntValue min(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.prim.min.int").value()(a, b).cast<PyTorch_IntValue>();
}

// prim::min.self_int : (int[]) -> (int)
PyTorch_IntValue min(const PyAnyTorchListOfTorchIntValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.prim.min.self_int").value()(self).cast<PyTorch_IntValue>();
}

// prim::NumToTensor.Scalar : (Scalar) -> (Tensor)
PyAnyTorchTensorValue NumToTensor(const PyAnyTorchScalarValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.prim.NumToTensor.Scalar").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), a).cast<PyAnyTorchTensorValue>();
}

// prim::RaiseException : (str, str?) -> ()
void RaiseException(const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls) {
  PyGlobals::get().lookupOperationClass("torch.prim.RaiseException").value()(msg, cls);
}

// prims::convert_element_type : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue convert_element_type(const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.prims.convert_element_type").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), a, dtype).cast<PyAnyTorchTensorValue>();
}

// prims::squeeze : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &a, const PyAnyTorchListOfTorchIntValue &dimensions) {
  return PyGlobals::get().lookupOperationClass("torch.prims.squeeze").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), a, dimensions).cast<PyAnyTorchTensorValue>();
}

// prims::var : (Tensor, int[]?, float, int?) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &inp, const PyAnyTorchOptionalListOfTorchIntValue &dims, const PyTorch_FloatValue &correction, const PyAnyTorchOptionalIntValue &output_dtype) {
  return PyGlobals::get().lookupOperationClass("torch.prims.var").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), inp, dims, correction, output_dtype).cast<PyAnyTorchTensorValue>();
}

// prims::view_of : (Tensor) -> (Tensor)
PyAnyTorchTensorValue view_of(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.prims.view_of").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), a).cast<PyAnyTorchTensorValue>();
}

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i) {
  return PyGlobals::get().lookupOperationClass("torch.quantized.linear").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), X, W_prepack, Y_scale_i, Y_zero_point_i).cast<PyAnyTorchTensorValue>();
}
