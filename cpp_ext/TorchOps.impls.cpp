
// aten::Bool.Tensor : (Tensor) -> (bool)
py::object Bool(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Bool.Tensor").value()(a);
}

// aten::Bool.float : (float) -> (bool)
py::object Bool(const PyTorch_FloatValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Bool.float").value()(a);
}

// aten::Bool.int : (int) -> (bool)
py::object Bool(const PyTorch_IntValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Bool.int").value()(a);
}

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
py::object Delete(const PyTorch_DictValue &self, const PyTorch_StringValue &key) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Delete.Dict_str").value()(self, key);
}

// aten::Float.Tensor : (Tensor) -> (float)
py::object Float(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Float.Tensor").value()(a);
}

// aten::Float.str : (str) -> (float)
py::object Float(const PyTorch_StringValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Float.str").value()(a);
}

// aten::FloatImplicit : (Tensor) -> (float)
py::object FloatImplicit(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.FloatImplicit").value()(a);
}

// aten::Int.Tensor : (Tensor) -> (int)
py::object Int(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Int.Tensor").value()(a);
}

// aten::Int.float : (float) -> (int)
py::object Int(const PyTorch_FloatValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Int.float").value()(a);
}

// aten::Int.bool : (bool) -> (int)
py::object Int(const PyTorch_BoolValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.Int.bool").value()(a);
}

// aten::IntImplicit : (Tensor) -> (int)
py::object IntImplicit(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.IntImplicit").value()(a);
}

// prim::RaiseException : (str, str?) -> ()
py::object RaiseException(const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls) {
  return PyGlobals::get().lookupOperationClass("torch.prim.RaiseException").value()(msg, cls);
}

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __and__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__and__.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::__and__.bool : (bool, bool) -> (bool)
py::object __and__(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__and__.bool").value()(a, b);
}

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
py::object __contains__(const PyTorch_DictValue &dict, const PyTorch_StringValue &key) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__contains__.str").value()(dict, key);
}

// aten::__contains__.int_list : (int[], int) -> (bool)
py::object __contains__(const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__contains__.int_list").value()(l, item);
}

// aten::__derive_index : (int, int, int) -> (int)
py::object __derive_index(const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__derive_index").value()(index, start, step);
}

// aten::__not__ : (bool) -> (bool)
py::object __not__(const PyTorch_BoolValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__not__").value()(self);
}

// aten::__range_length : (int, int, int) -> (int)
py::object __range_length(const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.__range_length").value()(lo, hi, step);
}

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
py::object _log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float) {
  return PyGlobals::get().lookupOperationClass("torch.aten._log_softmax").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, half_to_float);
}

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
py::object _log_softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten._log_softmax_backward_data").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output, dim, input_dtype);
}

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
py::object _reshape_alias(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride) {
  return PyGlobals::get().lookupOperationClass("torch.aten._reshape_alias").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride);
}

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
py::object _reshape_alias_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride) {
  return PyGlobals::get().lookupOperationClass("torch.aten._reshape_alias_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride);
}

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
py::object _shape_as_tensor(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten._shape_as_tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
py::object _softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float) {
  return PyGlobals::get().lookupOperationClass("torch.aten._softmax").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, half_to_float);
}

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
py::object _softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten._softmax_backward_data").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output, dim, input_dtype);
}

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
py::object _to_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten._to_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
py::object _unsafe_view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  return PyGlobals::get().lookupOperationClass("torch.aten._unsafe_view").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size);
}

// aten::abs : (Tensor) -> (Tensor)
py::object abs(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.abs").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::abs_ : (Tensor) -> (Tensor)
py::object abs_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.abs_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
py::object adaptive_avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size) {
  return PyGlobals::get().lookupOperationClass("torch.aten.adaptive_avg_pool2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, output_size);
}

// aten::add.str : (str, str) -> (str)
py::object add(const PyTorch_StringValue &a, const PyTorch_StringValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add.str").value()(a, b);
}

// aten::add.int : (int, int) -> (int)
py::object add(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add.int").value()(a, b);
}

// aten::add.float_int : (float, int) -> (float)
py::object add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.add.float_int").value()(a, b);
}

// aten::alias_copy : (Tensor) -> (Tensor)
py::object alias_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.alias_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::all : (Tensor) -> (Tensor)
py::object all(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.all").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::all.bool : (bool[]) -> (bool)
py::object all(const PyAnyTorchListOfTorchBoolValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.all.bool").value()(self);
}

// aten::amax : (Tensor, int[], bool) -> (Tensor)
py::object amax(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.amax").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::any : (Tensor) -> (Tensor)
py::object any(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.any").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
py::object any(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.any.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::any.bool : (bool[]) -> (bool)
py::object any(const PyAnyTorchListOfTorchBoolValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.any.bool").value()(self);
}

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
py::object argmax(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.argmax").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
py::object as_strided_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset) {
  return PyGlobals::get().lookupOperationClass("torch.aten.as_strided_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride, storage_offset);
}

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
py::object as_strided_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset) {
  return PyGlobals::get().lookupOperationClass("torch.aten.as_strided_scatter").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, size, stride, storage_offset);
}

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
py::object atan2(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.atan2").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
py::object atan2_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.atan2_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
py::object avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override) {
  return PyGlobals::get().lookupOperationClass("torch.aten.avg_pool2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, generator);
}

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli.p").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator);
}

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator);
}

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
py::object bernoulli_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli_.float").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator);
}

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
py::object bernoulli_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bernoulli_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator);
}

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_and.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_and_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_not : (Tensor) -> (Tensor)
py::object bitwise_not(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_not").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::bitwise_not_ : (Tensor) -> (Tensor)
py::object bitwise_not_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_not_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_or.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_or_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_xor.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bitwise_xor_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bmm : (Tensor, Tensor) -> (Tensor)
py::object bmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bmm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mat2);
}

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
py::object broadcast_to(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  return PyGlobals::get().lookupOperationClass("torch.aten.broadcast_to").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size);
}

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
py::object bucketize(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right) {
  return PyGlobals::get().lookupOperationClass("torch.aten.bucketize.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, boundaries, out_int32, right);
}

// aten::ceil : (Tensor) -> (Tensor)
py::object ceil(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ceil").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::ceil.float : (float) -> (int)
py::object ceil(const PyTorch_FloatValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ceil.float").value()(a);
}

// aten::ceil_ : (Tensor) -> (Tensor)
py::object ceil_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ceil_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::clone : (Tensor, int?) -> (Tensor)
py::object clone(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.clone").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, memory_format);
}

// aten::contiguous : (Tensor, int) -> (Tensor)
py::object contiguous(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.contiguous").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, memory_format);
}

// prims::convert_element_type : (Tensor, int) -> (Tensor)
py::object convert_element_type(const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.prims.convert_element_type").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), a, dtype);
}

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
py::object copy(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking) {
  return PyGlobals::get().lookupOperationClass("torch.aten.copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, non_blocking);
}

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
py::object copy_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking) {
  return PyGlobals::get().lookupOperationClass("torch.aten.copy_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, non_blocking);
}

// aten::cos : (Tensor) -> (Tensor)
py::object cos(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cos").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::cos_ : (Tensor) -> (Tensor)
py::object cos_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cos_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::cpu : (Tensor) -> (Tensor)
py::object cpu(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cpu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
py::object cumsum(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.cumsum").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype);
}

// aten::detach : (Tensor) -> (Tensor)
py::object detach(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.detach").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::detach_copy : (Tensor) -> (Tensor)
py::object detach_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.detach_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// prim::device : (Tensor) -> (Device)
py::object device(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.prim.device").value()(a);
}

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
py::object diagonal_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2) {
  return PyGlobals::get().lookupOperationClass("torch.aten.diagonal_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, offset, dim1, dim2);
}

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
py::object diagonal_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2) {
  return PyGlobals::get().lookupOperationClass("torch.aten.diagonal_scatter").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, offset, dim1, dim2);
}

// aten::dim : (Tensor) -> (int)
py::object dim(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.dim").value()(self);
}

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
py::object div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
py::object div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.Tensor_mode").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, rounding_mode);
}

// aten::div.int : (int, int) -> (float)
py::object div(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.int").value()(a, b);
}

// aten::div.float : (float, float) -> (float)
py::object div(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div.float").value()(a, b);
}

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
py::object div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode) {
  return PyGlobals::get().lookupOperationClass("torch.aten.div_.Tensor_mode").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, rounding_mode);
}

// aten::dropout : (Tensor, float, bool) -> (Tensor)
py::object dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train) {
  return PyGlobals::get().lookupOperationClass("torch.aten.dropout").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, p, train);
}

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
py::object dropout_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train) {
  return PyGlobals::get().lookupOperationClass("torch.aten.dropout_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, train);
}

// prim::dtype : (Tensor) -> (int)
py::object dtype(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.prim.dtype").value()(a);
}

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
py::object embedding(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse) {
  return PyGlobals::get().lookupOperationClass("torch.aten.embedding").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
py::object embedding_dense_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq) {
  return PyGlobals::get().lookupOperationClass("torch.aten.embedding_dense_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
py::object empty(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.empty.memory_format").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory, memory_format);
}

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object empty_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.empty_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
py::object eq(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::eq.int_list : (int[], int[]) -> (bool)
py::object eq(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.int_list").value()(a, b);
}

// aten::eq.str : (str, str) -> (bool)
py::object eq(const PyTorch_StringValue &a, const PyTorch_StringValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.str").value()(a, b);
}

// aten::eq.int : (int, int) -> (bool)
py::object eq(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.int").value()(a, b);
}

// aten::eq.float : (float, float) -> (bool)
py::object eq(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.float").value()(a, b);
}

// aten::eq.device : (Device, Device) -> (bool)
py::object eq(const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq.device").value()(a, b);
}

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.eq_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::erf : (Tensor) -> (Tensor)
py::object erf(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.erf").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::erf_ : (Tensor) -> (Tensor)
py::object erf_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.erf_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::exp : (Tensor) -> (Tensor)
py::object exp(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.exp").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::exp_ : (Tensor) -> (Tensor)
py::object exp_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.exp_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::expand : (Tensor, int[], bool) -> (Tensor)
py::object expand(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expand").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, implicit);
}

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
py::object expand_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expand_as").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
py::object expand_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expand_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, implicit);
}

// aten::expm1 : (Tensor) -> (Tensor)
py::object expm1(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expm1").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::expm1_ : (Tensor) -> (Tensor)
py::object expm1_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.expm1_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
py::object fft_fft(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fft_fft").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, n, dim, norm);
}

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
py::object fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fill.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, value);
}

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.fill_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, value);
}

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
py::object flatten(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.flatten.using_ints").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, start_dim, end_dim);
}

// aten::flip : (Tensor, int[]) -> (Tensor)
py::object flip(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  return PyGlobals::get().lookupOperationClass("torch.aten.flip").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims);
}

// aten::floor : (Tensor) -> (Tensor)
py::object floor(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::floor_ : (Tensor) -> (Tensor)
py::object floor_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floor_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
py::object floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floor_divide").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::floordiv.int : (int, int) -> (int)
py::object floordiv(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.floordiv.int").value()(a, b);
}

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
py::object frobenius_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.frobenius_norm.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
py::object gather(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gather").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, sparse_grad);
}

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ge(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::ge.int : (int, int) -> (bool)
py::object ge(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.int").value()(a, b);
}

// aten::ge.float : (float, float) -> (bool)
py::object ge(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.float").value()(a, b);
}

// aten::ge.float_int : (float, int) -> (bool)
py::object ge(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge.float_int").value()(a, b);
}

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ge_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::gelu : (Tensor, str) -> (Tensor)
py::object gelu(const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gelu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, approximate);
}

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
py::object gelu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gelu_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, approximate);
}

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
py::object gt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::gt.int : (int, int) -> (bool)
py::object gt(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.int").value()(a, b);
}

// aten::gt.float : (float, float) -> (bool)
py::object gt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.float").value()(a, b);
}

// aten::gt.float_int : (float, int) -> (bool)
py::object gt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt.float_int").value()(a, b);
}

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.gt_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::hardsigmoid : (Tensor) -> (Tensor)
py::object hardsigmoid(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardsigmoid").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
py::object hardsigmoid_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardsigmoid_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::hardswish : (Tensor) -> (Tensor)
py::object hardswish(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardswish").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::hardswish_ : (Tensor) -> (Tensor)
py::object hardswish_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.hardswish_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
py::object index_select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index) {
  return PyGlobals::get().lookupOperationClass("torch.aten.index_select").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index);
}

// aten::is_floating_point : (Tensor) -> (bool)
py::object is_floating_point(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.is_floating_point").value()(self);
}

// aten::join : (str, str[]) -> (str)
py::object join(const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values) {
  return PyGlobals::get().lookupOperationClass("torch.aten.join").value()(self, values);
}

// aten::keys.str : (Dict(str, t)) -> (str[])
py::object keys(const PyTorch_DictValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.keys.str").value()(PyAnyTorchListOfTorchStringType(DefaultingPyMlirContext::resolve()), self);
}

// prim::layout : (Tensor) -> (int)
py::object layout(const PyAnyTorchTensorValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.prim.layout").value()(a);
}

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
py::object le(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.le.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::le.int : (int, int) -> (bool)
py::object le(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.le.int").value()(a, b);
}

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object le_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.le_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::len.Tensor : (Tensor) -> (int)
py::object len(const PyAnyTorchTensorValue &t) {
  return PyGlobals::get().lookupOperationClass("torch.aten.len.Tensor").value()(t);
}

// aten::len.str : (str) -> (int)
py::object len(const PyTorch_StringValue &s) {
  return PyGlobals::get().lookupOperationClass("torch.aten.len.str").value()(s);
}

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object lerp(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lerp.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, end, weight);
}

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object lerp_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lerp_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, end, weight);
}

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
py::object lift_fresh_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lift_fresh_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
py::object linear(const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i) {
  return PyGlobals::get().lookupOperationClass("torch.quantized.linear").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), X, W_prepack, Y_scale_i, Y_zero_point_i);
}

// aten::log : (Tensor) -> (Tensor)
py::object log(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log.int : (int) -> (float)
py::object log(const PyTorch_IntValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log.int").value()(a);
}

// aten::log1p : (Tensor) -> (Tensor)
py::object log1p(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log1p").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log1p_ : (Tensor) -> (Tensor)
py::object log1p_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log1p_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log2 : (Tensor) -> (Tensor)
py::object log2(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log2").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log2_ : (Tensor) -> (Tensor)
py::object log2_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log2_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log_ : (Tensor) -> (Tensor)
py::object log_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
py::object log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.log_softmax.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype);
}

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
py::object logical_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_and").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
py::object logical_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_and_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_not : (Tensor) -> (Tensor)
py::object logical_not(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_not").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::logical_not_ : (Tensor) -> (Tensor)
py::object logical_not_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_not_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
py::object logical_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_or").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
py::object logical_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_or_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
py::object logical_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_xor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
py::object logical_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logical_xor_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
py::object logsumexp(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.logsumexp").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
py::object lt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::lt.int : (int, int) -> (bool)
py::object lt(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.int").value()(a, b);
}

// aten::lt.float : (float, float) -> (bool)
py::object lt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.float").value()(a, b);
}

// aten::lt.float_int : (float, int) -> (bool)
py::object lt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt.float_int").value()(a, b);
}

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.lt_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.masked_fill.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask, value);
}

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.masked_fill_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask, value);
}

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
py::object masked_select(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask) {
  return PyGlobals::get().lookupOperationClass("torch.aten.masked_select").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask);
}

// aten::matmul : (Tensor, Tensor) -> (Tensor)
py::object matmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.matmul").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::max : (Tensor) -> (Tensor)
py::object max(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.max").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// prim::max.self_int : (int[]) -> (int)
py::object max(const PyAnyTorchListOfTorchIntValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.prim.max.self_int").value()(self);
}

// prim::max.int : (int, int) -> (int)
py::object max(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.prim.max.int").value()(a, b);
}

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
py::object max_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode) {
  return PyGlobals::get().lookupOperationClass("torch.aten.max_pool2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
py::object max_pool2d_with_indices_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices) {
  return PyGlobals::get().lookupOperationClass("torch.aten.max_pool2d_with_indices_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

// aten::maximum : (Tensor, Tensor) -> (Tensor)
py::object maximum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.maximum").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::mean : (Tensor, int?) -> (Tensor)
py::object mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mean").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype);
}

// prim::min.self_int : (int[]) -> (int)
py::object min(const PyAnyTorchListOfTorchIntValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.prim.min.self_int").value()(self);
}

// prim::min.int : (int, int) -> (int)
py::object min(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.prim.min.int").value()(a, b);
}

// aten::minimum : (Tensor, Tensor) -> (Tensor)
py::object minimum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.minimum").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::mish : (Tensor) -> (Tensor)
py::object mish(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mish").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::mm : (Tensor, Tensor) -> (Tensor)
py::object mm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mm").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mat2);
}

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
py::object mse_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mse_loss").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, target, reduction);
}

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
py::object mul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::mul.int : (int, int) -> (int)
py::object mul(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul.int").value()(a, b);
}

// aten::mul.float : (float, float) -> (float)
py::object mul(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul.float").value()(a, b);
}

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mul_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::mv : (Tensor, Tensor) -> (Tensor)
py::object mv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec) {
  return PyGlobals::get().lookupOperationClass("torch.aten.mv").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, vec);
}

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
py::object narrow(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length) {
  return PyGlobals::get().lookupOperationClass("torch.aten.narrow").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, length);
}

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
py::object native_dropout_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale) {
  return PyGlobals::get().lookupOperationClass("torch.aten.native_dropout_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, mask, scale);
}

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ne(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::ne.int_list : (int[], int[]) -> (bool)
py::object ne(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.int_list").value()(a, b);
}

// aten::ne.int : (int, int) -> (bool)
py::object ne(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.int").value()(a, b);
}

// aten::ne.float_int : (float, int) -> (bool)
py::object ne(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.float_int").value()(a, b);
}

// aten::ne.bool : (bool, bool) -> (bool)
py::object ne(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne.bool").value()(a, b);
}

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ne_.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::neg : (Tensor) -> (Tensor)
py::object neg(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.neg").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::neg.int : (int) -> (int)
py::object neg(const PyTorch_IntValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.neg.int").value()(a);
}

// aten::neg.float : (float) -> (float)
py::object neg(const PyTorch_FloatValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.neg.float").value()(a);
}

// aten::neg_ : (Tensor) -> (Tensor)
py::object neg_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.neg_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_empty(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.new_empty").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory);
}

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_empty_strided(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.new_empty_strided").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride, dtype, layout, device, pin_memory);
}

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_ones(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.new_ones").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory);
}

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_zeros(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.new_zeros").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory);
}

// aten::numel : (Tensor) -> (int)
py::object numel(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.numel").value()(self);
}

// aten::numpy_T : (Tensor) -> (Tensor)
py::object numpy_T(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.numpy_T").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object ones(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ones").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory);
}

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object ones_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.ones_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
py::object pad(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value) {
  return PyGlobals::get().lookupOperationClass("torch.aten.pad").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, pad, mode, value);
}

// aten::permute : (Tensor, int[]) -> (Tensor)
py::object permute(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  return PyGlobals::get().lookupOperationClass("torch.aten.permute").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims);
}

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
py::object permute_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  return PyGlobals::get().lookupOperationClass("torch.aten.permute_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims);
}

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
py::object pow(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent) {
  return PyGlobals::get().lookupOperationClass("torch.aten.pow.Tensor_Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, exponent);
}

// aten::pow.int_float : (int, float) -> (float)
py::object pow(const PyTorch_IntValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.pow.int_float").value()(a, b);
}

// aten::prelu : (Tensor, Tensor) -> (Tensor)
py::object prelu(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight) {
  return PyGlobals::get().lookupOperationClass("torch.aten.prelu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, weight);
}

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object rand_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.rand_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object randint(const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randint.low").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), low, high, size, dtype, layout, device, pin_memory);
}

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randn").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory);
}

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
py::object randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randn.generator").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, generator, dtype, layout, device, pin_memory);
}

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object randn_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.randn_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}

// aten::reciprocal : (Tensor) -> (Tensor)
py::object reciprocal(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.reciprocal").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::reciprocal_ : (Tensor) -> (Tensor)
py::object reciprocal_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.reciprocal_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::relu : (Tensor) -> (Tensor)
py::object relu(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.relu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::relu6 : (Tensor) -> (Tensor)
py::object relu6(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.relu6").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::relu6_ : (Tensor) -> (Tensor)
py::object relu6_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.relu6_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::relu_ : (Tensor) -> (Tensor)
py::object relu_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.relu_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::remainder.int : (int, int) -> (int)
py::object remainder(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.remainder.int").value()(a, b);
}

// aten::repeat : (Tensor, int[]) -> (Tensor)
py::object repeat(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats) {
  return PyGlobals::get().lookupOperationClass("torch.aten.repeat").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, repeats);
}

// aten::reshape : (Tensor, int[]) -> (Tensor)
py::object reshape(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape) {
  return PyGlobals::get().lookupOperationClass("torch.aten.reshape").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, shape);
}

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
py::object resize_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.resize_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, memory_format);
}

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
py::object roll(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims) {
  return PyGlobals::get().lookupOperationClass("torch.aten.roll").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, shifts, dims);
}

// aten::round : (Tensor) -> (Tensor)
py::object round(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.round").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::round_ : (Tensor) -> (Tensor)
py::object round_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.round_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::rsqrt : (Tensor) -> (Tensor)
py::object rsqrt(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.rsqrt").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::rsqrt_ : (Tensor) -> (Tensor)
py::object rsqrt_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.rsqrt_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
py::object scatter_add(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter_add").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src);
}

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
py::object scatter_add_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter_add_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src);
}

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
py::object scatter_reduce(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter_reduce.two").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src, reduce, include_self);
}

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
py::object scatter_reduce_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.scatter_reduce_.two").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src, reduce, include_self);
}

// aten::select.int : (Tensor, int, int) -> (Tensor)
py::object select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  return PyGlobals::get().lookupOperationClass("torch.aten.select.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index);
}

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
py::object select_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  return PyGlobals::get().lookupOperationClass("torch.aten.select_copy.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index);
}

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
py::object select_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  return PyGlobals::get().lookupOperationClass("torch.aten.select_scatter").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, dim, index);
}

// aten::sigmoid : (Tensor) -> (Tensor)
py::object sigmoid(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sigmoid").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::sigmoid_ : (Tensor) -> (Tensor)
py::object sigmoid_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sigmoid_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::silu : (Tensor) -> (Tensor)
py::object silu(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.silu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::silu_ : (Tensor) -> (Tensor)
py::object silu_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.silu_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::sin : (Tensor) -> (Tensor)
py::object sin(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sin").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::sin_ : (Tensor) -> (Tensor)
py::object sin_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sin_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::size : (Tensor) -> (int[])
py::object size(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.size").value()(PyAnyTorchListOfTorchIntType(DefaultingPyMlirContext::resolve()), self);
}

// aten::size.int : (Tensor, int) -> (int)
py::object size(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.size.int").value()(self, dim);
}

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
py::object slice(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.slice.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, end, step);
}

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
py::object slice_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.slice_copy.Tensor").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, end, step);
}

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
py::object slice_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.slice_scatter").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, dim, start, end, step);
}

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
py::object softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.softmax.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype);
}

// aten::sort.int : (int[], bool) -> ()
py::object sort(const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sort.int").value()(self, reverse);
}

// aten::sqrt : (Tensor) -> (Tensor)
py::object sqrt(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sqrt").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::sqrt.int : (int) -> (float)
py::object sqrt(const PyTorch_IntValue &a) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sqrt.int").value()(a);
}

// aten::sqrt_ : (Tensor) -> (Tensor)
py::object sqrt_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sqrt_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::square : (Tensor) -> (Tensor)
py::object square(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.square").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::square_ : (Tensor) -> (Tensor)
py::object square_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.square_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
py::object squeeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.squeeze.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::squeeze : (Tensor) -> (Tensor)
py::object squeeze(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.squeeze").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::squeeze_copy : (Tensor) -> (Tensor)
py::object squeeze_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.squeeze_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
py::object squeeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.squeeze_copy.dim").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::std : (Tensor, bool) -> (Tensor)
py::object std(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased) {
  return PyGlobals::get().lookupOperationClass("torch.aten.std").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, unbiased);
}

// aten::sub.int : (int, int) -> (int)
py::object sub(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sub.int").value()(a, b);
}

// aten::sub.float : (float, float) -> (float)
py::object sub(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sub.float").value()(a, b);
}

// aten::sum : (Tensor, int?) -> (Tensor)
py::object sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.sum").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype);
}

// aten::t : (Tensor) -> (Tensor)
py::object t(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.t").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::t_copy : (Tensor) -> (Tensor)
py::object t_copy(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.t_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::tanh : (Tensor) -> (Tensor)
py::object tanh(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tanh").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::tanh_ : (Tensor) -> (Tensor)
py::object tanh_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tanh_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
py::object tanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tanh_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output);
}

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tensor.bool").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad);
}

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tensor.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad);
}

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  return PyGlobals::get().lookupOperationClass("torch.aten.tensor.float").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad);
}

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.dtype").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, non_blocking, copy, memory_format);
}

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.dtype_layout").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
}

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.other").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, non_blocking, copy, memory_format);
}

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.prim_Device").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, device, dtype, non_blocking, copy);
}

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.to.device").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, device, dtype, non_blocking, copy, memory_format);
}

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
py::object transpose(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1) {
  return PyGlobals::get().lookupOperationClass("torch.aten.transpose.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim0, dim1);
}

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
py::object transpose_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1) {
  return PyGlobals::get().lookupOperationClass("torch.aten.transpose_copy.int").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim0, dim1);
}

// aten::triu : (Tensor, int) -> (Tensor)
py::object triu(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal) {
  return PyGlobals::get().lookupOperationClass("torch.aten.triu").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, diagonal);
}

// aten::triu_ : (Tensor, int) -> (Tensor)
py::object triu_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal) {
  return PyGlobals::get().lookupOperationClass("torch.aten.triu_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, diagonal);
}

// aten::type_as : (Tensor, Tensor) -> (Tensor)
py::object type_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.type_as").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
py::object unfold_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step) {
  return PyGlobals::get().lookupOperationClass("torch.aten.unfold_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dimension, size, step);
}

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
py::object uniform(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.uniform").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, from, to, generator);
}

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
py::object uniform_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator) {
  return PyGlobals::get().lookupOperationClass("torch.aten.uniform_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, from, to, generator);
}

// aten::unsqueeze : (Tensor, int) -> (Tensor)
py::object unsqueeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.unsqueeze").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
py::object unsqueeze_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.unsqueeze_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
py::object unsqueeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  return PyGlobals::get().lookupOperationClass("torch.aten.unsqueeze_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
py::object upsample_nearest2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w) {
  return PyGlobals::get().lookupOperationClass("torch.aten.upsample_nearest2d").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, output_size, scales_h, scales_w);
}

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
py::object upsample_nearest2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w) {
  return PyGlobals::get().lookupOperationClass("torch.aten.upsample_nearest2d_backward").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output_size, input_size, scales_h, scales_w);
}

// aten::var : (Tensor, bool) -> (Tensor)
py::object var(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased) {
  return PyGlobals::get().lookupOperationClass("torch.aten.var").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, unbiased);
}

// aten::view : (Tensor, int[]) -> (Tensor)
py::object view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  return PyGlobals::get().lookupOperationClass("torch.aten.view").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size);
}

// aten::view_copy : (Tensor, int[]) -> (Tensor)
py::object view_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  return PyGlobals::get().lookupOperationClass("torch.aten.view_copy").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size);
}

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
py::object view_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype) {
  return PyGlobals::get().lookupOperationClass("torch.aten.view_copy.dtype").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype);
}

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
py::object where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  return PyGlobals::get().lookupOperationClass("torch.aten.where.self").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), condition, self, other);
}

// aten::zero : (Tensor) -> (Tensor)
py::object zero(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.zero").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::zero_ : (Tensor) -> (Tensor)
py::object zero_(const PyAnyTorchTensorValue &self) {
  return PyGlobals::get().lookupOperationClass("torch.aten.zero_").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object zeros(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  return PyGlobals::get().lookupOperationClass("torch.aten.zeros").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory);
}

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object zeros_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  return PyGlobals::get().lookupOperationClass("torch.aten.zeros_like").value()(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}