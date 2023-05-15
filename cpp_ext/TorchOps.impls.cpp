
// aten::Bool.Tensor : (Tensor) -> (bool)
py::object Bool(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBoolTensorOp")(a);
}

// aten::Bool.float : (float) -> (bool)
py::object Bool(const PyTorch_FloatValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBoolFloatOp")(a);
}

// aten::Bool.int : (int) -> (bool)
py::object Bool(const PyTorch_IntValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBoolIntOp")(a);
}

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
py::object Delete(const PyTorch_DictValue &self, const PyTorch_StringValue &key) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDeleteDictStrOp")(self, key);
}

// aten::Float.Tensor : (Tensor) -> (float)
py::object Float(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFloatTensorOp")(a);
}

// aten::Float.str : (str) -> (float)
py::object Float(const PyTorch_StringValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFloatStrOp")(a);
}

// aten::FloatImplicit : (Tensor) -> (float)
py::object FloatImplicit(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFloatImplicitOp")(a);
}

// aten::Int.Tensor : (Tensor) -> (int)
py::object Int(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenIntTensorOp")(a);
}

// aten::Int.float : (float) -> (int)
py::object Int(const PyTorch_FloatValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenIntFloatOp")(a);
}

// aten::Int.bool : (bool) -> (int)
py::object Int(const PyTorch_BoolValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenIntBoolOp")(a);
}

// aten::IntImplicit : (Tensor) -> (int)
py::object IntImplicit(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenIntImplicitOp")(a);
}

// prim::RaiseException : (str, str?) -> ()
py::object RaiseException(const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimRaiseExceptionOp")(msg, cls);
}

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __and__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten__And__TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::__and__.bool : (bool, bool) -> (bool)
py::object __and__(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten__And__BoolOp")(a, b);
}

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
py::object __contains__(const PyTorch_DictValue &dict, const PyTorch_StringValue &key) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten__Contains__StrOp")(dict, key);
}

// aten::__contains__.int_list : (int[], int) -> (bool)
py::object __contains__(const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten__Contains__IntListOp")(l, item);
}

// aten::__derive_index : (int, int, int) -> (int)
py::object __derive_index(const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten__DeriveIndexOp")(index, start, step);
}

// aten::__not__ : (bool) -> (bool)
py::object __not__(const PyTorch_BoolValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten__Not__Op")(self);
}

// aten::__range_length : (int, int, int) -> (int)
py::object __range_length(const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten__RangeLengthOp")(lo, hi, step);
}

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
py::object _log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_LogSoftmaxOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, half_to_float);
}

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
py::object _log_softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_LogSoftmaxBackwardDataOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output, dim, input_dtype);
}

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
py::object _reshape_alias(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_ReshapeAliasOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride);
}

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
py::object _reshape_alias_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_ReshapeAliasCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride);
}

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
py::object _shape_as_tensor(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_ShapeAsTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
py::object _softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_SoftmaxOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, half_to_float);
}

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
py::object _softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_SoftmaxBackwardDataOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output, dim, input_dtype);
}

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
py::object _to_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_ToCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
py::object _unsafe_view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("Aten_UnsafeViewOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size);
}

// aten::abs : (Tensor) -> (Tensor)
py::object abs(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAbsOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::abs_ : (Tensor) -> (Tensor)
py::object abs_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAbs_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
py::object adaptive_avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAdaptiveAvgPool2dOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, output_size);
}

// aten::add.str : (str, str) -> (str)
py::object add(const PyTorch_StringValue &a, const PyTorch_StringValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAddStrOp")(a, b);
}

// aten::add.int : (int, int) -> (int)
py::object add(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAddIntOp")(a, b);
}

// aten::add.float_int : (float, int) -> (float)
py::object add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAddFloatIntOp")(a, b);
}

// aten::alias_copy : (Tensor) -> (Tensor)
py::object alias_copy(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAliasCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::all : (Tensor) -> (Tensor)
py::object all(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAllOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::all.bool : (bool[]) -> (bool)
py::object all(const PyAnyTorchListOfTorchBoolValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAllBoolOp")(self);
}

// aten::amax : (Tensor, int[], bool) -> (Tensor)
py::object amax(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAmaxOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::any : (Tensor) -> (Tensor)
py::object any(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAnyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
py::object any(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAnyDimOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::any.bool : (bool[]) -> (bool)
py::object any(const PyAnyTorchListOfTorchBoolValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAnyBoolOp")(self);
}

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
py::object argmax(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenArgmaxOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
py::object as_strided_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAsStridedCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride, storage_offset);
}

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
py::object as_strided_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAsStridedScatterOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, size, stride, storage_offset);
}

// aten::atan : (Tensor) -> (Tensor)
py::object atan(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAtanOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
py::object atan2(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAtan2Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
py::object atan2_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAtan2_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::atan_ : (Tensor) -> (Tensor)
py::object atan_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAtan_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
py::object avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAvgPool2dOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBernoulliOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, generator);
}

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBernoulliPOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator);
}

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBernoulliTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator);
}

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
py::object bernoulli_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBernoulli_FloatOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator);
}

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
py::object bernoulli_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBernoulli_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, generator);
}

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBitwiseAndTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBitwiseAnd_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_not : (Tensor) -> (Tensor)
py::object bitwise_not(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBitwiseNotOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::bitwise_not_ : (Tensor) -> (Tensor)
py::object bitwise_not_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBitwiseNot_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBitwiseOrTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBitwiseOr_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBitwiseXorTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBitwiseXor_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::bmm : (Tensor, Tensor) -> (Tensor)
py::object bmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBmmOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mat2);
}

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
py::object broadcast_to(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBroadcastToOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size);
}

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
py::object bucketize(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenBucketizeTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, boundaries, out_int32, right);
}

// aten::ceil : (Tensor) -> (Tensor)
py::object ceil(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCeilOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::ceil.float : (float) -> (int)
py::object ceil(const PyTorch_FloatValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCeilFloatOp")(a);
}

// aten::ceil_ : (Tensor) -> (Tensor)
py::object ceil_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCeil_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::clone : (Tensor, int?) -> (Tensor)
py::object clone(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCloneOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, memory_format);
}

// aten::contiguous : (Tensor, int) -> (Tensor)
py::object contiguous(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenContiguousOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, memory_format);
}

// prims::convert_element_type : (Tensor, int) -> (Tensor)
py::object convert_element_type(const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimsConvertElementTypeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), a, dtype);
}

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
py::object copy(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, non_blocking);
}

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
py::object copy_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCopy_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, non_blocking);
}

// aten::cos : (Tensor) -> (Tensor)
py::object cos(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCosOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::cos_ : (Tensor) -> (Tensor)
py::object cos_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCos_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::cpu : (Tensor) -> (Tensor)
py::object cpu(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCpuOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
py::object cumsum(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenCumsumOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype);
}

// aten::detach : (Tensor) -> (Tensor)
py::object detach(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDetachOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::detach_copy : (Tensor) -> (Tensor)
py::object detach_copy(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDetachCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// prim::device : (Tensor) -> (Device)
py::object device(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimDeviceOp")(a);
}

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
py::object diagonal_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDiagonalCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, offset, dim1, dim2);
}

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
py::object diagonal_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDiagonalScatterOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, offset, dim1, dim2);
}

// aten::dim : (Tensor) -> (int)
py::object dim(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDimOp")(self);
}

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
py::object div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDivTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
py::object div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDivTensorModeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, rounding_mode);
}

// aten::div.int : (int, int) -> (float)
py::object div(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDivIntOp")(a, b);
}

// aten::div.float : (float, float) -> (float)
py::object div(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDivFloatOp")(a, b);
}

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDiv_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
py::object div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDiv_TensorModeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, rounding_mode);
}

// aten::dropout : (Tensor, float, bool) -> (Tensor)
py::object dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDropoutOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), input, p, train);
}

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
py::object dropout_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenDropout_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, p, train);
}

// prim::dtype : (Tensor) -> (int)
py::object dtype(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimDtypeOp")(a);
}

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
py::object embedding(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEmbeddingOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
py::object embedding_dense_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEmbeddingDenseBackwardOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
py::object empty(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEmptyMemoryFormatOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory, memory_format);
}

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object empty_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEmptyLikeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
py::object eq(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEqTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::eq.int_list : (int[], int[]) -> (bool)
py::object eq(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEqIntListOp")(a, b);
}

// aten::eq.str : (str, str) -> (bool)
py::object eq(const PyTorch_StringValue &a, const PyTorch_StringValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEqStrOp")(a, b);
}

// aten::eq.int : (int, int) -> (bool)
py::object eq(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEqIntOp")(a, b);
}

// aten::eq.float : (float, float) -> (bool)
py::object eq(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEqFloatOp")(a, b);
}

// aten::eq.device : (Device, Device) -> (bool)
py::object eq(const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEqDeviceOp")(a, b);
}

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenEq_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::erf : (Tensor) -> (Tensor)
py::object erf(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenErfOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::erf_ : (Tensor) -> (Tensor)
py::object erf_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenErf_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::exp : (Tensor) -> (Tensor)
py::object exp(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenExpOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::exp_ : (Tensor) -> (Tensor)
py::object exp_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenExp_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::expand : (Tensor, int[], bool) -> (Tensor)
py::object expand(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenExpandOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, implicit);
}

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
py::object expand_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenExpandAsOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
py::object expand_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenExpandCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, implicit);
}

// aten::expm1 : (Tensor) -> (Tensor)
py::object expm1(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenExpm1Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::expm1_ : (Tensor) -> (Tensor)
py::object expm1_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenExpm1_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
py::object fft_fft(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFftFftOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, n, dim, norm);
}

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
py::object fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFillTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, value);
}

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFill_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, value);
}

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
py::object flatten(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFlattenUsingIntsOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, start_dim, end_dim);
}

// aten::flip : (Tensor, int[]) -> (Tensor)
py::object flip(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFlipOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims);
}

// aten::floor : (Tensor) -> (Tensor)
py::object floor(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFloorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::floor_ : (Tensor) -> (Tensor)
py::object floor_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFloor_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
py::object floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFloorDivideOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::floordiv.int : (int, int) -> (int)
py::object floordiv(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFloordivIntOp")(a, b);
}

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
py::object frobenius_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenFrobeniusNormDimOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
py::object gather(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGatherOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, sparse_grad);
}

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ge(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGeTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::ge.int : (int, int) -> (bool)
py::object ge(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGeIntOp")(a, b);
}

// aten::ge.float : (float, float) -> (bool)
py::object ge(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGeFloatOp")(a, b);
}

// aten::ge.float_int : (float, int) -> (bool)
py::object ge(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGeFloatIntOp")(a, b);
}

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGe_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::gelu : (Tensor, str) -> (Tensor)
py::object gelu(const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGeluOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, approximate);
}

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
py::object gelu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGeluBackwardOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, approximate);
}

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
py::object gt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGtTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::gt.int : (int, int) -> (bool)
py::object gt(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGtIntOp")(a, b);
}

// aten::gt.float : (float, float) -> (bool)
py::object gt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGtFloatOp")(a, b);
}

// aten::gt.float_int : (float, int) -> (bool)
py::object gt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGtFloatIntOp")(a, b);
}

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenGt_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::hardsigmoid : (Tensor) -> (Tensor)
py::object hardsigmoid(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenHardsigmoidOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
py::object hardsigmoid_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenHardsigmoid_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::hardswish : (Tensor) -> (Tensor)
py::object hardswish(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenHardswishOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::hardswish_ : (Tensor) -> (Tensor)
py::object hardswish_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenHardswish_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::imag : (Tensor) -> (Tensor)
py::object imag(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenImagOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
py::object index_select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenIndexSelectOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index);
}

// aten::is_floating_point : (Tensor) -> (bool)
py::object is_floating_point(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenIsFloatingPointOp")(self);
}

// aten::join : (str, str[]) -> (str)
py::object join(const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenJoinOp")(self, values);
}

// aten::keys.str : (Dict(str, t)) -> (str[])
py::object keys(const PyTorch_DictValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenKeysStrOp")(PyAnyTorchListOfTorchStringType(DefaultingPyMlirContext::resolve()), self);
}

// prim::layout : (Tensor) -> (int)
py::object layout(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimLayoutOp")(a);
}

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
py::object le(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLeTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::le.int : (int, int) -> (bool)
py::object le(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLeIntOp")(a, b);
}

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object le_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLe_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::len.Tensor : (Tensor) -> (int)
py::object len(const PyAnyTorchTensorValue &t) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLenTensorOp")(t);
}

// aten::len.str : (str) -> (int)
py::object len(const PyTorch_StringValue &s) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLenStrOp")(s);
}

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object lerp(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLerpTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, end, weight);
}

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object lerp_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLerp_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, end, weight);
}

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
py::object lift_fresh_copy(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLiftFreshCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
py::object linear(const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("QuantizedLinearOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), X, W_prepack, Y_scale_i, Y_zero_point_i);
}

// aten::log : (Tensor) -> (Tensor)
py::object log(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log.int : (int) -> (float)
py::object log(const PyTorch_IntValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogIntOp")(a);
}

// aten::log1p : (Tensor) -> (Tensor)
py::object log1p(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLog1pOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log1p_ : (Tensor) -> (Tensor)
py::object log1p_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLog1p_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log2 : (Tensor) -> (Tensor)
py::object log2(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLog2Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log2_ : (Tensor) -> (Tensor)
py::object log2_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLog2_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log_ : (Tensor) -> (Tensor)
py::object log_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLog_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
py::object log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogSoftmaxIntOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype);
}

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
py::object logical_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogicalAndOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
py::object logical_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogicalAnd_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_not : (Tensor) -> (Tensor)
py::object logical_not(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogicalNotOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::logical_not_ : (Tensor) -> (Tensor)
py::object logical_not_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogicalNot_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
py::object logical_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogicalOrOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
py::object logical_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogicalOr_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
py::object logical_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogicalXorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
py::object logical_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogicalXor_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
py::object logsumexp(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLogsumexpOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, keepdim);
}

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
py::object lt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLtTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::lt.int : (int, int) -> (bool)
py::object lt(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLtIntOp")(a, b);
}

// aten::lt.float : (float, float) -> (bool)
py::object lt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLtFloatOp")(a, b);
}

// aten::lt.float_int : (float, int) -> (bool)
py::object lt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLtFloatIntOp")(a, b);
}

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenLt_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMaskedFillTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask, value);
}

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMaskedFill_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask, value);
}

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
py::object masked_select(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMaskedSelectOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mask);
}

// aten::matmul : (Tensor, Tensor) -> (Tensor)
py::object matmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMatmulOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::max : (Tensor) -> (Tensor)
py::object max(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMaxOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// prim::max.self_int : (int[]) -> (int)
py::object max(const PyAnyTorchListOfTorchIntValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimMaxSelfIntOp")(self);
}

// prim::max.int : (int, int) -> (int)
py::object max(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimMaxIntOp")(a, b);
}

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
py::object max_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMaxPool2dOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
py::object max_pool2d_with_indices_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMaxPool2dWithIndicesBackwardOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

// aten::maximum : (Tensor, Tensor) -> (Tensor)
py::object maximum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMaximumOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::mean : (Tensor, int?) -> (Tensor)
py::object mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMeanOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype);
}

// prim::min.self_int : (int[]) -> (int)
py::object min(const PyAnyTorchListOfTorchIntValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimMinSelfIntOp")(self);
}

// prim::min.int : (int, int) -> (int)
py::object min(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimMinIntOp")(a, b);
}

// aten::minimum : (Tensor, Tensor) -> (Tensor)
py::object minimum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMinimumOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::mish : (Tensor) -> (Tensor)
py::object mish(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMishOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::mm : (Tensor, Tensor) -> (Tensor)
py::object mm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMmOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, mat2);
}

// aten::movedim.int : (Tensor, int, int) -> (Tensor)
py::object movedim(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &source, const PyTorch_IntValue &destination) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMovedimIntOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, source, destination);
}

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
py::object mse_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMseLossOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, target, reduction);
}

// aten::mse_loss_backward : (Tensor, Tensor, Tensor, int) -> (Tensor)
py::object mse_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMseLossBackwardOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, self, target, reduction);
}

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
py::object mul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMulTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::mul.int : (int, int) -> (int)
py::object mul(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMulIntOp")(a, b);
}

// aten::mul.float : (float, float) -> (float)
py::object mul(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMulFloatOp")(a, b);
}

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMul_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::mv : (Tensor, Tensor) -> (Tensor)
py::object mv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenMvOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, vec);
}

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
py::object narrow(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNarrowOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, length);
}

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
py::object native_dropout_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNativeDropoutBackwardOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, mask, scale);
}

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ne(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNeTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::ne.int_list : (int[], int[]) -> (bool)
py::object ne(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNeIntListOp")(a, b);
}

// aten::ne.int : (int, int) -> (bool)
py::object ne(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNeIntOp")(a, b);
}

// aten::ne.float_int : (float, int) -> (bool)
py::object ne(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNeFloatIntOp")(a, b);
}

// aten::ne.bool : (bool, bool) -> (bool)
py::object ne(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNeBoolOp")(a, b);
}

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNe_TensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::neg : (Tensor) -> (Tensor)
py::object neg(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNegOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::neg.int : (int) -> (int)
py::object neg(const PyTorch_IntValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNegIntOp")(a);
}

// aten::neg.float : (float) -> (float)
py::object neg(const PyTorch_FloatValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNegFloatOp")(a);
}

// aten::neg_ : (Tensor) -> (Tensor)
py::object neg_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNeg_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_empty(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNewEmptyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory);
}

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_empty_strided(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNewEmptyStridedOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, stride, dtype, layout, device, pin_memory);
}

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_ones(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNewOnesOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory);
}

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_zeros(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNewZerosOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, dtype, layout, device, pin_memory);
}

// aten::numel : (Tensor) -> (int)
py::object numel(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNumelOp")(self);
}

// aten::numpy_T : (Tensor) -> (Tensor)
py::object numpy_T(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenNumpyTOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::one_hot : (Tensor, int) -> (Tensor)
py::object one_hot(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &num_classes) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenOneHotOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, num_classes);
}

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object ones(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenOnesOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory);
}

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object ones_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenOnesLikeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
py::object pad(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenPadOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, pad, mode, value);
}

// aten::permute : (Tensor, int[]) -> (Tensor)
py::object permute(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenPermuteOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims);
}

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
py::object permute_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenPermuteCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dims);
}

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
py::object pow(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenPowTensorTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, exponent);
}

// aten::pow.int_float : (int, float) -> (float)
py::object pow(const PyTorch_IntValue &a, const PyTorch_FloatValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenPowIntFloatOp")(a, b);
}

// aten::prelu : (Tensor, Tensor) -> (Tensor)
py::object prelu(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenPreluOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, weight);
}

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object rand_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRandLikeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object randint(const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRandintLowOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), low, high, size, dtype, layout, device, pin_memory);
}

// aten::randint : (int, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object randint(const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRandintOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), high, size, dtype, layout, device, pin_memory);
}

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRandnOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory);
}

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
py::object randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRandnGeneratorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, generator, dtype, layout, device, pin_memory);
}

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object randn_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRandnLikeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}

// aten::real : (Tensor) -> (Tensor)
py::object real(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRealOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::reciprocal : (Tensor) -> (Tensor)
py::object reciprocal(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenReciprocalOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::reciprocal_ : (Tensor) -> (Tensor)
py::object reciprocal_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenReciprocal_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::relu : (Tensor) -> (Tensor)
py::object relu(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenReluOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::relu6 : (Tensor) -> (Tensor)
py::object relu6(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRelu6Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::relu6_ : (Tensor) -> (Tensor)
py::object relu6_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRelu6_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::relu_ : (Tensor) -> (Tensor)
py::object relu_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRelu_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::remainder.int : (int, int) -> (int)
py::object remainder(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRemainderIntOp")(a, b);
}

// aten::repeat : (Tensor, int[]) -> (Tensor)
py::object repeat(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRepeatOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, repeats);
}

// aten::reshape : (Tensor, int[]) -> (Tensor)
py::object reshape(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenReshapeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, shape);
}

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
py::object resize_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenResize_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size, memory_format);
}

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
py::object roll(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRollOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, shifts, dims);
}

// aten::round : (Tensor) -> (Tensor)
py::object round(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRoundOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::round_ : (Tensor) -> (Tensor)
py::object round_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRound_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::rsqrt : (Tensor) -> (Tensor)
py::object rsqrt(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRsqrtOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::rsqrt_ : (Tensor) -> (Tensor)
py::object rsqrt_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenRsqrt_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::scatter.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
py::object scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenScatterSrcOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src);
}

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
py::object scatter_add(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenScatterAddOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src);
}

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
py::object scatter_add_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenScatterAdd_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src);
}

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
py::object scatter_reduce(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenScatterReduceTwoOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src, reduce, include_self);
}

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
py::object scatter_reduce_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenScatterReduce_TwoOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index, src, reduce, include_self);
}

// aten::select.int : (Tensor, int, int) -> (Tensor)
py::object select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSelectIntOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index);
}

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
py::object select_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSelectCopyIntOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, index);
}

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
py::object select_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSelectScatterOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, dim, index);
}

// aten::sigmoid : (Tensor) -> (Tensor)
py::object sigmoid(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSigmoidOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::sigmoid_ : (Tensor) -> (Tensor)
py::object sigmoid_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSigmoid_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::silu : (Tensor) -> (Tensor)
py::object silu(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSiluOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::silu_ : (Tensor) -> (Tensor)
py::object silu_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSilu_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::sin : (Tensor) -> (Tensor)
py::object sin(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSinOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::sin_ : (Tensor) -> (Tensor)
py::object sin_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSin_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::size : (Tensor) -> (int[])
py::object size(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSizeOp")(PyAnyTorchListOfTorchIntType(DefaultingPyMlirContext::resolve()), self);
}

// aten::size.int : (Tensor, int) -> (int)
py::object size(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSizeIntOp")(self, dim);
}

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
py::object slice(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSliceTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, end, step);
}

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
py::object slice_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSliceCopyTensorOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, start, end, step);
}

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
py::object slice_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSliceScatterOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, src, dim, start, end, step);
}

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
py::object softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSoftmaxIntOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim, dtype);
}

// aten::sort.int : (int[], bool) -> ()
py::object sort(const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSortIntOp")(self, reverse);
}

// aten::sqrt : (Tensor) -> (Tensor)
py::object sqrt(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSqrtOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::sqrt.int : (int) -> (float)
py::object sqrt(const PyTorch_IntValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSqrtIntOp")(a);
}

// aten::sqrt_ : (Tensor) -> (Tensor)
py::object sqrt_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSqrt_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::square : (Tensor) -> (Tensor)
py::object square(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSquareOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::square_ : (Tensor) -> (Tensor)
py::object square_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSquare_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
py::object squeeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSqueezeDimOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::squeeze : (Tensor) -> (Tensor)
py::object squeeze(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSqueezeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// prims::squeeze : (Tensor, int[]) -> (Tensor)
py::object squeeze(const PyAnyTorchTensorValue &a, const PyAnyTorchListOfTorchIntValue &dimensions) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimsSqueezeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), a, dimensions);
}

// aten::squeeze_copy : (Tensor) -> (Tensor)
py::object squeeze_copy(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSqueezeCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
py::object squeeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSqueezeCopyDimOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::std : (Tensor, bool) -> (Tensor)
py::object std(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenStdOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, unbiased);
}

// aten::sub.int : (int, int) -> (int)
py::object sub(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSubIntOp")(a, b);
}

// aten::sub.float : (float, float) -> (float)
py::object sub(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSubFloatOp")(a, b);
}

// aten::sum : (Tensor, int?) -> (Tensor)
py::object sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenSumOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype);
}

// aten::t : (Tensor) -> (Tensor)
py::object t(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::t_copy : (Tensor) -> (Tensor)
py::object t_copy(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::tanh : (Tensor) -> (Tensor)
py::object tanh(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTanhOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::tanh_ : (Tensor) -> (Tensor)
py::object tanh_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTanh_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
py::object tanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTanhBackwardOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output);
}

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTensorBoolOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad);
}

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTensorIntOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad);
}

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTensorFloatOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), t, dtype, device, requires_grad);
}

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenToDtypeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, non_blocking, copy, memory_format);
}

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenToDtypeLayoutOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
}

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenToOtherOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other, non_blocking, copy, memory_format);
}

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenToPrimDeviceOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, device, dtype, non_blocking, copy);
}

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenToDeviceOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, device, dtype, non_blocking, copy, memory_format);
}

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
py::object transpose(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTransposeIntOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim0, dim1);
}

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
py::object transpose_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTransposeCopyIntOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim0, dim1);
}

// aten::triu : (Tensor, int) -> (Tensor)
py::object triu(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTriuOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, diagonal);
}

// aten::triu_ : (Tensor, int) -> (Tensor)
py::object triu_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTriu_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, diagonal);
}

// aten::type_as : (Tensor, Tensor) -> (Tensor)
py::object type_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenTypeAsOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, other);
}

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
py::object unfold_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenUnfoldCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dimension, size, step);
}

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
py::object uniform(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenUniformOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, from, to, generator);
}

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
py::object uniform_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenUniform_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, from, to, generator);
}

// aten::unsqueeze : (Tensor, int) -> (Tensor)
py::object unsqueeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenUnsqueezeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
py::object unsqueeze_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenUnsqueeze_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
py::object unsqueeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenUnsqueezeCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dim);
}

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
py::object upsample_nearest2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenUpsampleNearest2dOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, output_size, scales_h, scales_w);
}

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
py::object upsample_nearest2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenUpsampleNearest2dBackwardOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), grad_output, output_size, input_size, scales_h, scales_w);
}

// aten::var : (Tensor, bool) -> (Tensor)
py::object var(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenVarOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, unbiased);
}

// aten::view : (Tensor, int[]) -> (Tensor)
py::object view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenViewOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size);
}

// aten::view_as_complex : (Tensor) -> (Tensor)
py::object view_as_complex(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenViewAsComplexOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::view_copy : (Tensor, int[]) -> (Tensor)
py::object view_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenViewCopyOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, size);
}

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
py::object view_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenViewCopyDtypeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype);
}

// prims::view_of : (Tensor) -> (Tensor)
py::object view_of(const PyAnyTorchTensorValue &a) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("PrimsViewOfOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), a);
}

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
py::object where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenWhereSelfOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), condition, self, other);
}

// aten::zero : (Tensor) -> (Tensor)
py::object zero(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenZeroOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::zero_ : (Tensor) -> (Tensor)
py::object zero_(const PyAnyTorchTensorValue &self) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenZero_Op")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self);
}

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object zeros(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenZerosOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), size, dtype, layout, device, pin_memory);
}

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object zeros_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) {
  auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenZerosLikeOp")(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), self, dtype, layout, device, pin_memory, memory_format);
}
