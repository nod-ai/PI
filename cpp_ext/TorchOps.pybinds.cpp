
// aten::Bool.Tensor : (Tensor) -> (bool)
m.def("Bool", [](const PyAnyTorchTensorValue &a) { return Bool(a); }, "a"_a);

// aten::Bool.float : (float) -> (bool)
m.def("Bool", [](const PyTorch_FloatValue &a) { return Bool(a); }, "a"_a);

// aten::Bool.int : (int) -> (bool)
m.def("Bool", [](const PyTorch_IntValue &a) { return Bool(a); }, "a"_a);

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
m.def("Delete", [](const PyTorch_DictValue &self, const PyTorch_StringValue &key) { return Delete(self, key); }, "self"_a, "key"_a);

// aten::Float.Tensor : (Tensor) -> (float)
m.def("Float", [](const PyAnyTorchTensorValue &a) { return Float(a); }, "a"_a);

// aten::Float.Scalar : (Scalar) -> (float)
m.def("Float", [](const PyAnyTorchScalarValue &a) { return Float(a); }, "a"_a);

// aten::Float.str : (str) -> (float)
m.def("Float", [](const PyTorch_StringValue &a) { return Float(a); }, "a"_a);

// aten::FloatImplicit : (Tensor) -> (float)
m.def("FloatImplicit", [](const PyAnyTorchTensorValue &a) { return FloatImplicit(a); }, "a"_a);

// aten::Int.Tensor : (Tensor) -> (int)
m.def("Int", [](const PyAnyTorchTensorValue &a) { return Int(a); }, "a"_a);

// aten::Int.float : (float) -> (int)
m.def("Int", [](const PyTorch_FloatValue &a) { return Int(a); }, "a"_a);

// aten::Int.Scalar : (Scalar) -> (int)
m.def("Int", [](const PyAnyTorchScalarValue &a) { return Int(a); }, "a"_a);

// aten::Int.bool : (bool) -> (int)
m.def("Int", [](const PyTorch_BoolValue &a) { return Int(a); }, "a"_a);

// aten::IntImplicit : (Tensor) -> (int)
m.def("IntImplicit", [](const PyAnyTorchTensorValue &a) { return IntImplicit(a); }, "a"_a);

// prim::NumToTensor.Scalar : (Scalar) -> (Tensor)
m.def("NumToTensor", [](const PyAnyTorchScalarValue &a) { return NumToTensor(a); }, "a"_a);

// prim::RaiseException : (str, str?) -> ()
m.def("RaiseException", [](const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls) { return RaiseException(msg, cls); }, "msg"_a, "cls"_a = py::none());

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("__and__", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return __and__(self, other); }, "self"_a, "other"_a);

// aten::__and__.bool : (bool, bool) -> (bool)
m.def("__and__", [](const PyTorch_BoolValue &a, const PyTorch_BoolValue &b) { return __and__(a, b); }, "a"_a, "b"_a);

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
m.def("__contains__", [](const PyTorch_DictValue &dict, const PyTorch_StringValue &key) { return __contains__(dict, key); }, "dict"_a, "key"_a);

// aten::__contains__.int_list : (int[], int) -> (bool)
m.def("__contains__", [](const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item) { return __contains__(l, item); }, "l"_a, "item"_a);

// aten::__derive_index : (int, int, int) -> (int)
m.def("__derive_index", [](const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step) { return __derive_index(index, start, step); }, "index"_a, "start"_a, "step"_a);

// aten::__not__ : (bool) -> (bool)
m.def("__not__", [](const PyTorch_BoolValue &self) { return __not__(self); }, "self"_a);

// aten::__range_length : (int, int, int) -> (int)
m.def("__range_length", [](const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step) { return __range_length(lo, hi, step); }, "lo"_a, "hi"_a, "step"_a);

// aten::_convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool, bool) -> (Tensor)
m.def("_convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, const PyTorch_BoolValue &allow_tf32) { return _convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a, "benchmark"_a, "deterministic"_a, "cudnn_enabled"_a, "allow_tf32"_a);

// aten::_convolution.deprecated : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool) -> (Tensor)
m.def("_convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled) { return _convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a, "benchmark"_a, "deterministic"_a, "cudnn_enabled"_a);

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
m.def("_log_softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float) { return _log_softmax(self, dim, half_to_float); }, "self"_a, "dim"_a, "half_to_float"_a);

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
m.def("_log_softmax_backward_data", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype) { return _log_softmax_backward_data(grad_output, output, dim, input_dtype); }, "grad_output"_a, "output"_a, "dim"_a, "input_dtype"_a);

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
m.def("_reshape_alias", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride) { return _reshape_alias(self, size, stride); }, "self"_a, "size"_a, "stride"_a);

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
m.def("_reshape_alias_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride) { return _reshape_alias_copy(self, size, stride); }, "self"_a, "size"_a, "stride"_a);

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
m.def("_shape_as_tensor", [](const PyAnyTorchTensorValue &self) { return _shape_as_tensor(self); }, "self"_a);

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
m.def("_softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float) { return _softmax(self, dim, half_to_float); }, "self"_a, "dim"_a, "half_to_float"_a);

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
m.def("_softmax_backward_data", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype) { return _softmax_backward_data(grad_output, output, dim, input_dtype); }, "grad_output"_a, "output"_a, "dim"_a, "input_dtype"_a);

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
m.def("_to_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format) { return _to_copy(self, dtype, layout, device, pin_memory, non_blocking, memory_format); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "non_blocking"_a = false, "memory_format"_a = py::none());

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
m.def("_unsafe_view", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) { return _unsafe_view(self, size); }, "self"_a, "size"_a);

// aten::abs : (Tensor) -> (Tensor)
m.def("abs", [](const PyAnyTorchTensorValue &self) { return abs(self); }, "self"_a);

// aten::abs_ : (Tensor) -> (Tensor)
m.def("abs_", [](const PyAnyTorchTensorValue &self) { return abs_(self); }, "self"_a);

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
m.def("adaptive_avg_pool2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size) { return adaptive_avg_pool2d(self, output_size); }, "self"_a, "output_size"_a);

// aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("add", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha) { return add(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("add", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) { return add(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::add.str : (str, str) -> (str)
m.def("add", [](const PyTorch_StringValue &a, const PyTorch_StringValue &b) { return add(a, b); }, "a"_a, "b"_a);

// aten::add.int : (int, int) -> (int)
m.def("add", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return add(a, b); }, "a"_a, "b"_a);

// aten::add.float_int : (float, int) -> (float)
m.def("add", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b) { return add(a, b); }, "a"_a, "b"_a);

// aten::add_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("add_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha) { return add_(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::add_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("add_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) { return add_(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcdiv", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value) { return addcdiv(self, tensor1, tensor2, value); }, "self"_a, "tensor1"_a, "tensor2"_a, "value"_a = 1);

// aten::addcdiv_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcdiv_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value) { return addcdiv_(self, tensor1, tensor2, value); }, "self"_a, "tensor1"_a, "tensor2"_a, "value"_a = 1);

// aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcmul", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value) { return addcmul(self, tensor1, tensor2, value); }, "self"_a, "tensor1"_a, "tensor2"_a, "value"_a = 1);

// aten::addcmul_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcmul_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value) { return addcmul_(self, tensor1, tensor2, value); }, "self"_a, "tensor1"_a, "tensor2"_a, "value"_a = 1);

// aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("addmm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat1, const PyAnyTorchTensorValue &mat2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha) { return addmm(self, mat1, mat2, beta, alpha); }, "self"_a, "mat1"_a, "mat2"_a, "beta"_a = 1, "alpha"_a = 1);

// aten::alias_copy : (Tensor) -> (Tensor)
m.def("alias_copy", [](const PyAnyTorchTensorValue &self) { return alias_copy(self); }, "self"_a);

// aten::all : (Tensor) -> (Tensor)
m.def("all", [](const PyAnyTorchTensorValue &self) { return all(self); }, "self"_a);

// aten::all.bool : (bool[]) -> (bool)
m.def("all", [](const PyAnyTorchListOfTorchBoolValue &self) { return all(self); }, "self"_a);

// aten::amax : (Tensor, int[], bool) -> (Tensor)
m.def("amax", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) { return amax(self, dim, keepdim); }, "self"_a, "dim"_a = std::vector<int>{}, "keepdim"_a = false);

// aten::any : (Tensor) -> (Tensor)
m.def("any", [](const PyAnyTorchTensorValue &self) { return any(self); }, "self"_a);

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
m.def("any", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim) { return any(self, dim, keepdim); }, "self"_a, "dim"_a, "keepdim"_a = false);

// aten::any.bool : (bool[]) -> (bool)
m.def("any", [](const PyAnyTorchListOfTorchBoolValue &self) { return any(self); }, "self"_a);

// aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return arange(end, dtype, layout, device, pin_memory); }, "end"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return arange(start, end, dtype, layout, device, pin_memory); }, "start"_a, "end"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return arange(start, end, step, dtype, layout, device, pin_memory); }, "start"_a, "end"_a, "step"_a = 1, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::arange.start_out : (Scalar, Scalar, Scalar, Tensor) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchTensorValue &out) { return arange(start, end, step, out); }, "start"_a, "end"_a, "step"_a = 1, "out"_a);

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
m.def("argmax", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim) { return argmax(self, dim, keepdim); }, "self"_a, "dim"_a = py::none(), py::kw_only(), "keepdim"_a = false);

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
m.def("as_strided_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset) { return as_strided_copy(self, size, stride, storage_offset); }, "self"_a, "size"_a, "stride"_a, "storage_offset"_a = py::none());

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
m.def("as_strided_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset) { return as_strided_scatter(self, src, size, stride, storage_offset); }, "self"_a, "src"_a, "size"_a, "stride"_a, "storage_offset"_a = py::none());

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
m.def("atan2", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return atan2(self, other); }, "self"_a, "other"_a);

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
m.def("atan2_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return atan2_(self, other); }, "self"_a, "other"_a);

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
m.def("avg_pool2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override) { return avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override); }, "self"_a, "kernel_size"_a, "stride"_a = std::vector<int>{}, "padding"_a = std::vector<int>{0, 0}, "ceil_mode"_a = false, "count_include_pad"_a = true, "divisor_override"_a = py::none());

// aten::baddbmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("baddbmm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha) { return baddbmm(self, batch1, batch2, beta, alpha); }, "self"_a, "batch1"_a, "batch2"_a, "beta"_a = 1, "alpha"_a = 1);

// aten::baddbmm_ : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("baddbmm_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha) { return baddbmm_(self, batch1, batch2, beta, alpha); }, "self"_a, "batch1"_a, "batch2"_a, "beta"_a = 1, "alpha"_a = 1);

// aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)
m.def("batch_norm", [](const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enabled) { return batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled); }, "input"_a, "weight"_a = py::none(), "bias"_a = py::none(), "running_mean"_a = py::none(), "running_var"_a = py::none(), py::kw_only(), "training"_a, "momentum"_a, "eps"_a, "cudnn_enabled"_a);

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator) { return bernoulli(self, generator); }, "self"_a, "generator"_a = py::none());

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator) { return bernoulli(self, p, generator); }, "self"_a, "p"_a, "generator"_a = py::none());

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator) { return bernoulli(self, p, generator); }, "self"_a, "p"_a, "generator"_a = py::none());

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
m.def("bernoulli_", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator) { return bernoulli_(self, p, generator); }, "self"_a, "p"_a = 0.5, "generator"_a = py::none());

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
m.def("bernoulli_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator) { return bernoulli_(self, p, generator); }, "self"_a, "p"_a, "generator"_a = py::none());

// aten::bincount : (Tensor, Tensor?, int) -> (Tensor)
m.def("bincount", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &weights, const PyTorch_IntValue &minlength) { return bincount(self, weights, minlength); }, "self"_a, "weights"_a = py::none(), py::kw_only(), "minlength"_a = 0);

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_and", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return bitwise_and(self, other); }, "self"_a, "other"_a);

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_and_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return bitwise_and_(self, other); }, "self"_a, "other"_a);

// aten::bitwise_not : (Tensor) -> (Tensor)
m.def("bitwise_not", [](const PyAnyTorchTensorValue &self) { return bitwise_not(self); }, "self"_a);

// aten::bitwise_not_ : (Tensor) -> (Tensor)
m.def("bitwise_not_", [](const PyAnyTorchTensorValue &self) { return bitwise_not_(self); }, "self"_a);

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_or", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return bitwise_or(self, other); }, "self"_a, "other"_a);

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_or_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return bitwise_or_(self, other); }, "self"_a, "other"_a);

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_xor", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return bitwise_xor(self, other); }, "self"_a, "other"_a);

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_xor_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return bitwise_xor_(self, other); }, "self"_a, "other"_a);

// aten::bmm : (Tensor, Tensor) -> (Tensor)
m.def("bmm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2) { return bmm(self, mat2); }, "self"_a, "mat2"_a);

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
m.def("broadcast_to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) { return broadcast_to(self, size); }, "self"_a, "size"_a);

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
m.def("bucketize", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right) { return bucketize(self, boundaries, out_int32, right); }, "self"_a, "boundaries"_a, "out_int32"_a = false, "right"_a = false);

// aten::cat : (Tensor[], int) -> (Tensor)
m.def("cat", [](const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim) { return cat(tensors, dim); }, "tensors"_a, "dim"_a = 0);

// aten::ceil : (Tensor) -> (Tensor)
m.def("ceil", [](const PyAnyTorchTensorValue &self) { return ceil(self); }, "self"_a);

// aten::ceil.float : (float) -> (int)
m.def("ceil", [](const PyTorch_FloatValue &a) { return ceil(a); }, "a"_a);

// aten::ceil_ : (Tensor) -> (Tensor)
m.def("ceil_", [](const PyAnyTorchTensorValue &self) { return ceil_(self); }, "self"_a);

// aten::clamp : (Tensor, Scalar?, Scalar?) -> (Tensor)
m.def("clamp", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max) { return clamp(self, min, max); }, "self"_a, "min"_a = py::none(), "max"_a = py::none());

// aten::clamp.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
m.def("clamp", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max) { return clamp(self, min, max); }, "self"_a, "min"_a = py::none(), "max"_a = py::none());

// aten::clamp_ : (Tensor, Scalar?, Scalar?) -> (Tensor)
m.def("clamp_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max) { return clamp_(self, min, max); }, "self"_a, "min"_a = py::none(), "max"_a = py::none());

// aten::clamp_.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
m.def("clamp_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max) { return clamp_(self, min, max); }, "self"_a, "min"_a = py::none(), "max"_a = py::none());

// aten::clamp_max : (Tensor, Scalar) -> (Tensor)
m.def("clamp_max", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max) { return clamp_max(self, max); }, "self"_a, "max"_a);

// aten::clamp_max_ : (Tensor, Scalar) -> (Tensor)
m.def("clamp_max_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max) { return clamp_max_(self, max); }, "self"_a, "max"_a);

// aten::clamp_min : (Tensor, Scalar) -> (Tensor)
m.def("clamp_min", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min) { return clamp_min(self, min); }, "self"_a, "min"_a);

// aten::clamp_min_ : (Tensor, Scalar) -> (Tensor)
m.def("clamp_min_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min) { return clamp_min_(self, min); }, "self"_a, "min"_a);

// aten::clone : (Tensor, int?) -> (Tensor)
m.def("clone", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format) { return clone(self, memory_format); }, "self"_a, "memory_format"_a = py::none());

// aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)
m.def("constant_pad_nd", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyAnyTorchScalarValue &value) { return constant_pad_nd(self, pad__, value); }, "self"_a, "pad__"_a, "value"_a = 0);

// aten::contiguous : (Tensor, int) -> (Tensor)
m.def("contiguous", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format) { return contiguous(self, memory_format); }, "self"_a, "memory_format"_a = 0);

// aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)
m.def("conv2d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_IntValue &groups) { return conv2d(input, weight, bias, stride, padding, dilation, groups); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "stride"_a = std::vector<int>{1, 1}, "padding"_a = std::vector<int>{0, 0}, "dilation"_a = std::vector<int>{1, 1}, "groups"_a = 1);

// aten::conv_transpose1d : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose1d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) { return conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "stride"_a = std::vector<int>{1}, "padding"_a = std::vector<int>{0}, "output_padding"_a = std::vector<int>{0}, "groups"_a = 1, "dilation"_a = std::vector<int>{1});

// aten::conv_transpose2d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose2d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) { return conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "stride"_a = std::vector<int>{1, 1}, "padding"_a = std::vector<int>{0, 0}, "output_padding"_a = std::vector<int>{0, 0}, "groups"_a = 1, "dilation"_a = std::vector<int>{1, 1});

// aten::conv_transpose3d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose3d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) { return conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "stride"_a = std::vector<int>{1, 1, 1}, "padding"_a = std::vector<int>{0, 0, 0}, "output_padding"_a = std::vector<int>{0, 0, 0}, "groups"_a = 1, "dilation"_a = std::vector<int>{1, 1, 1});

// prims::convert_element_type : (Tensor, int) -> (Tensor)
m.def("convert_element_type", [](const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype) { return convert_element_type(a, dtype); }, "a"_a, "dtype"_a);

// aten::convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)
m.def("convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups) { return convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a);

// aten::convolution_overrideable : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)
m.def("convolution_overrideable", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups) { return convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a);

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
m.def("copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking) { return copy(self, src, non_blocking); }, "self"_a, "src"_a, "non_blocking"_a = false);

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
m.def("copy_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking) { return copy_(self, src, non_blocking); }, "self"_a, "src"_a, "non_blocking"_a = false);

// aten::cos : (Tensor) -> (Tensor)
m.def("cos", [](const PyAnyTorchTensorValue &self) { return cos(self); }, "self"_a);

// aten::cos_ : (Tensor) -> (Tensor)
m.def("cos_", [](const PyAnyTorchTensorValue &self) { return cos_(self); }, "self"_a);

// aten::cpu : (Tensor) -> (Tensor)
m.def("cpu", [](const PyAnyTorchTensorValue &self) { return cpu(self); }, "self"_a);

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
m.def("cumsum", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) { return cumsum(self, dim, dtype); }, "self"_a, "dim"_a, "dtype"_a = py::none());

// aten::detach : (Tensor) -> (Tensor)
m.def("detach", [](const PyAnyTorchTensorValue &self) { return detach(self); }, "self"_a);

// aten::detach_copy : (Tensor) -> (Tensor)
m.def("detach_copy", [](const PyAnyTorchTensorValue &self) { return detach_copy(self); }, "self"_a);

// prim::device : (Tensor) -> (Device)
m.def("device", [](const PyAnyTorchTensorValue &a) { return device(a); }, "a"_a);

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
m.def("diagonal_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2) { return diagonal_copy(self, offset, dim1, dim2); }, "self"_a, "offset"_a = 0, "dim1"_a = 0, "dim2"_a = 1);

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
m.def("diagonal_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2) { return diagonal_scatter(self, src, offset, dim1, dim2); }, "self"_a, "src"_a, "offset"_a = 0, "dim1"_a = 0, "dim2"_a = 1);

// aten::dim : (Tensor) -> (int)
m.def("dim", [](const PyAnyTorchTensorValue &self) { return dim(self); }, "self"_a);

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("div", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return div(self, other); }, "self"_a, "other"_a);

// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("div", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return div(self, other); }, "self"_a, "other"_a);

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
m.def("div", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode) { return div(self, other, rounding_mode); }, "self"_a, "other"_a, "rounding_mode"_a = py::none());

// aten::div.int : (int, int) -> (float)
m.def("div", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return div(a, b); }, "a"_a, "b"_a);

// aten::div.float : (float, float) -> (float)
m.def("div", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) { return div(a, b); }, "a"_a, "b"_a);

// aten::div : (Scalar, Scalar) -> (float)
m.def("div", [](const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b) { return div(a, b); }, "a"_a, "b"_a);

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("div_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return div_(self, other); }, "self"_a, "other"_a);

// aten::div_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("div_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return div_(self, other); }, "self"_a, "other"_a);

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
m.def("div_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode) { return div_(self, other, rounding_mode); }, "self"_a, "other"_a, "rounding_mode"_a = py::none());

// aten::dropout : (Tensor, float, bool) -> (Tensor)
m.def("dropout", [](const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train) { return dropout(input, p, train); }, "input"_a, "p"_a, "train"_a);

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
m.def("dropout_", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train) { return dropout_(self, p, train); }, "self"_a, "p"_a, "train"_a);

// prim::dtype : (Tensor) -> (int)
m.def("dtype", [](const PyAnyTorchTensorValue &a) { return dtype(a); }, "a"_a);

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
m.def("embedding", [](const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse) { return embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse); }, "weight"_a, "indices"_a, "padding_idx"_a = -1, "scale_grad_by_freq"_a = false, "sparse"_a = false);

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
m.def("embedding_dense_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq) { return embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq); }, "grad_output"_a, "indices"_a, "num_weights"_a, "padding_idx"_a, "scale_grad_by_freq"_a);

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("empty", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) { return empty(size, dtype, layout, device, pin_memory, memory_format); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("empty_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) { return empty_like(self, dtype, layout, device, pin_memory, memory_format); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("eq", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return eq(self, other); }, "self"_a, "other"_a);

// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("eq", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return eq(self, other); }, "self"_a, "other"_a);

// aten::eq.int_list : (int[], int[]) -> (bool)
m.def("eq", [](const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b) { return eq(a, b); }, "a"_a, "b"_a);

// aten::eq.str : (str, str) -> (bool)
m.def("eq", [](const PyTorch_StringValue &a, const PyTorch_StringValue &b) { return eq(a, b); }, "a"_a, "b"_a);

// aten::eq.int : (int, int) -> (bool)
m.def("eq", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return eq(a, b); }, "a"_a, "b"_a);

// aten::eq.float : (float, float) -> (bool)
m.def("eq", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) { return eq(a, b); }, "a"_a, "b"_a);

// aten::eq.device : (Device, Device) -> (bool)
m.def("eq", [](const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b) { return eq(a, b); }, "a"_a, "b"_a);

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("eq_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return eq_(self, other); }, "self"_a, "other"_a);

// aten::eq_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("eq_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return eq_(self, other); }, "self"_a, "other"_a);

// aten::erf : (Tensor) -> (Tensor)
m.def("erf", [](const PyAnyTorchTensorValue &self) { return erf(self); }, "self"_a);

// aten::erf_ : (Tensor) -> (Tensor)
m.def("erf_", [](const PyAnyTorchTensorValue &self) { return erf_(self); }, "self"_a);

// aten::exp : (Tensor) -> (Tensor)
m.def("exp", [](const PyAnyTorchTensorValue &self) { return exp(self); }, "self"_a);

// aten::exp_ : (Tensor) -> (Tensor)
m.def("exp_", [](const PyAnyTorchTensorValue &self) { return exp_(self); }, "self"_a);

// aten::expand : (Tensor, int[], bool) -> (Tensor)
m.def("expand", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit) { return expand(self, size, implicit); }, "self"_a, "size"_a, "implicit"_a = false);

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
m.def("expand_as", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return expand_as(self, other); }, "self"_a, "other"_a);

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
m.def("expand_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit) { return expand_copy(self, size, implicit); }, "self"_a, "size"_a, "implicit"_a = false);

// aten::expm1 : (Tensor) -> (Tensor)
m.def("expm1", [](const PyAnyTorchTensorValue &self) { return expm1(self); }, "self"_a);

// aten::expm1_ : (Tensor) -> (Tensor)
m.def("expm1_", [](const PyAnyTorchTensorValue &self) { return expm1_(self); }, "self"_a);

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
m.def("fft_fft", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm) { return fft_fft(self, n, dim, norm); }, "self"_a, "n"_a = py::none(), "dim"_a = -1, "norm"_a = py::none());

// aten::fill.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fill", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value) { return fill(self, value); }, "self"_a, "value"_a);

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("fill", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value) { return fill(self, value); }, "self"_a, "value"_a);

// aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fill_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value) { return fill_(self, value); }, "self"_a, "value"_a);

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("fill_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value) { return fill_(self, value); }, "self"_a, "value"_a);

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
m.def("flatten", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim) { return flatten(self, start_dim, end_dim); }, "self"_a, "start_dim"_a = 0, "end_dim"_a = -1);

// aten::flip : (Tensor, int[]) -> (Tensor)
m.def("flip", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) { return flip(self, dims); }, "self"_a, "dims"_a);

// aten::floor : (Tensor) -> (Tensor)
m.def("floor", [](const PyAnyTorchTensorValue &self) { return floor(self); }, "self"_a);

// aten::floor_ : (Tensor) -> (Tensor)
m.def("floor_", [](const PyAnyTorchTensorValue &self) { return floor_(self); }, "self"_a);

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
m.def("floor_divide", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return floor_divide(self, other); }, "self"_a, "other"_a);

// aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("floor_divide", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return floor_divide(self, other); }, "self"_a, "other"_a);

// aten::floordiv.int : (int, int) -> (int)
m.def("floordiv", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return floordiv(a, b); }, "a"_a, "b"_a);

// aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fmod", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return fmod(self, other); }, "self"_a, "other"_a);

// aten::fmod_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fmod_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return fmod_(self, other); }, "self"_a, "other"_a);

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
m.def("frobenius_norm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) { return frobenius_norm(self, dim, keepdim); }, "self"_a, "dim"_a, "keepdim"_a = false);

// aten::full : (int[], Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("full", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return full(size, fill_value, dtype, layout, device, pin_memory); }, "size"_a, "fill_value"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::full_like : (Tensor, Scalar, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("full_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) { return full_like(self, fill_value, dtype, layout, device, pin_memory, memory_format); }, "self"_a, "fill_value"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
m.def("gather", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad) { return gather(self, dim, index, sparse_grad); }, "self"_a, "dim"_a, "index"_a, "sparse_grad"_a = false);

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ge", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return ge(self, other); }, "self"_a, "other"_a);

// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ge", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return ge(self, other); }, "self"_a, "other"_a);

// aten::ge.int : (int, int) -> (bool)
m.def("ge", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return ge(a, b); }, "a"_a, "b"_a);

// aten::ge.float : (float, float) -> (bool)
m.def("ge", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) { return ge(a, b); }, "a"_a, "b"_a);

// aten::ge.float_int : (float, int) -> (bool)
m.def("ge", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b) { return ge(a, b); }, "a"_a, "b"_a);

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ge_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return ge_(self, other); }, "self"_a, "other"_a);

// aten::ge_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ge_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return ge_(self, other); }, "self"_a, "other"_a);

// aten::gelu : (Tensor, str) -> (Tensor)
m.def("gelu", [](const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate) { return gelu(self, approximate); }, "self"_a, "approximate"_a = "{s}");

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
m.def("gelu_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate) { return gelu_backward(grad_output, self, approximate); }, "grad_output"_a, "self"_a, "approximate"_a = "{s}");

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("gt", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return gt(self, other); }, "self"_a, "other"_a);

// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("gt", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return gt(self, other); }, "self"_a, "other"_a);

// aten::gt.int : (int, int) -> (bool)
m.def("gt", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return gt(a, b); }, "a"_a, "b"_a);

// aten::gt.float : (float, float) -> (bool)
m.def("gt", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) { return gt(a, b); }, "a"_a, "b"_a);

// aten::gt.float_int : (float, int) -> (bool)
m.def("gt", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b) { return gt(a, b); }, "a"_a, "b"_a);

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("gt_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return gt_(self, other); }, "self"_a, "other"_a);

// aten::gt_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("gt_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return gt_(self, other); }, "self"_a, "other"_a);

// aten::hardsigmoid : (Tensor) -> (Tensor)
m.def("hardsigmoid", [](const PyAnyTorchTensorValue &self) { return hardsigmoid(self); }, "self"_a);

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
m.def("hardsigmoid_", [](const PyAnyTorchTensorValue &self) { return hardsigmoid_(self); }, "self"_a);

// aten::hardswish : (Tensor) -> (Tensor)
m.def("hardswish", [](const PyAnyTorchTensorValue &self) { return hardswish(self); }, "self"_a);

// aten::hardswish_ : (Tensor) -> (Tensor)
m.def("hardswish_", [](const PyAnyTorchTensorValue &self) { return hardswish_(self); }, "self"_a);

// aten::hardtanh : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val) { return hardtanh(self, min_val, max_val); }, "self"_a, "min_val"_a = -1, "max_val"_a = 1);

// aten::hardtanh_ : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val) { return hardtanh_(self, min_val, max_val); }, "self"_a, "min_val"_a = -1, "max_val"_a = 1);

// aten::hardtanh_backward : (Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val) { return hardtanh_backward(grad_output, self, min_val, max_val); }, "grad_output"_a, "self"_a, "min_val"_a, "max_val"_a);

// aten::index.Tensor_hacked_twin : (Tensor, Tensor[]) -> (Tensor)
m.def("index", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices) { return index(self, indices); }, "self"_a, "indices"_a);

// aten::index_put.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
m.def("index_put", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate) { return index_put(self, indices, values, accumulate); }, "self"_a, "indices"_a, "values"_a, "accumulate"_a = false);

// aten::index_put_.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
m.def("index_put_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate) { return index_put_(self, indices, values, accumulate); }, "self"_a, "indices"_a, "values"_a, "accumulate"_a = false);

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
m.def("index_select", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index) { return index_select(self, dim, index); }, "self"_a, "dim"_a, "index"_a);

// aten::is_floating_point : (Tensor) -> (bool)
m.def("is_floating_point", [](const PyAnyTorchTensorValue &self) { return is_floating_point(self); }, "self"_a);

// aten::join : (str, str[]) -> (str)
m.def("join", [](const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values) { return join(self, values); }, "self"_a, "values"_a);

// aten::keys.str : (Dict(str, t)) -> (str[])
m.def("keys", [](const PyTorch_DictValue &self) { return keys(self); }, "self"_a);

// aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)
m.def("layer_norm", [](const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enable) { return layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable); }, "input"_a, "normalized_shape"_a, "weight"_a = py::none(), "bias"_a = py::none(), py::kw_only(), "eps"_a = 1.0000000000000001e-05, "cudnn_enable"_a = true);

// prim::layout : (Tensor) -> (int)
m.def("layout", [](const PyAnyTorchTensorValue &a) { return layout(a); }, "a"_a);

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("le", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return le(self, other); }, "self"_a, "other"_a);

// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("le", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return le(self, other); }, "self"_a, "other"_a);

// aten::le.int : (int, int) -> (bool)
m.def("le", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return le(a, b); }, "a"_a, "b"_a);

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("le_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return le_(self, other); }, "self"_a, "other"_a);

// aten::le_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("le_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return le_(self, other); }, "self"_a, "other"_a);

// aten::leaky_relu : (Tensor, Scalar) -> (Tensor)
m.def("leaky_relu", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope) { return leaky_relu(self, negative_slope); }, "self"_a, "negative_slope"_a = 0.01);

// aten::leaky_relu_ : (Tensor, Scalar) -> (Tensor)
m.def("leaky_relu_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope) { return leaky_relu_(self, negative_slope); }, "self"_a, "negative_slope"_a = 0.01);

// aten::leaky_relu_backward : (Tensor, Tensor, Scalar, bool) -> (Tensor)
m.def("leaky_relu_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, const PyTorch_BoolValue &self_is_result) { return leaky_relu_backward(grad_output, self, negative_slope, self_is_result); }, "grad_output"_a, "self"_a, "negative_slope"_a, "self_is_result"_a);

// aten::len.Tensor : (Tensor) -> (int)
m.def("len", [](const PyAnyTorchTensorValue &t) { return len(t); }, "t"_a);

// aten::len.str : (str) -> (int)
m.def("len", [](const PyTorch_StringValue &s) { return len(s); }, "s"_a);

// aten::len.t : (t[]) -> (int)
m.def("len", [](const PyAnyTorchListValue &a) { return len(a); }, "a"_a);

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("lerp", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight) { return lerp(self, end, weight); }, "self"_a, "end"_a, "weight"_a);

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("lerp_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight) { return lerp_(self, end, weight); }, "self"_a, "end"_a, "weight"_a);

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
m.def("lift_fresh_copy", [](const PyAnyTorchTensorValue &self) { return lift_fresh_copy(self); }, "self"_a);

// aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)
m.def("linear", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias) { return linear(input, weight, bias); }, "input"_a, "weight"_a, "bias"_a = py::none());

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
m.def("linear", [](const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i) { return linear(X, W_prepack, Y_scale_i, Y_zero_point_i); }, "X"_a, "W_prepack"_a, "Y_scale_i"_a, "Y_zero_point_i"_a);

// aten::log : (Tensor) -> (Tensor)
m.def("log", [](const PyAnyTorchTensorValue &self) { return log(self); }, "self"_a);

// aten::log.int : (int) -> (float)
m.def("log", [](const PyTorch_IntValue &a) { return log(a); }, "a"_a);

// aten::log1p : (Tensor) -> (Tensor)
m.def("log1p", [](const PyAnyTorchTensorValue &self) { return log1p(self); }, "self"_a);

// aten::log1p_ : (Tensor) -> (Tensor)
m.def("log1p_", [](const PyAnyTorchTensorValue &self) { return log1p_(self); }, "self"_a);

// aten::log2 : (Tensor) -> (Tensor)
m.def("log2", [](const PyAnyTorchTensorValue &self) { return log2(self); }, "self"_a);

// aten::log2_ : (Tensor) -> (Tensor)
m.def("log2_", [](const PyAnyTorchTensorValue &self) { return log2_(self); }, "self"_a);

// aten::log_ : (Tensor) -> (Tensor)
m.def("log_", [](const PyAnyTorchTensorValue &self) { return log_(self); }, "self"_a);

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
m.def("log_softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) { return log_softmax(self, dim, dtype); }, "self"_a, "dim"_a, "dtype"_a = py::none());

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
m.def("logical_and", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return logical_and(self, other); }, "self"_a, "other"_a);

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_and_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return logical_and_(self, other); }, "self"_a, "other"_a);

// aten::logical_not : (Tensor) -> (Tensor)
m.def("logical_not", [](const PyAnyTorchTensorValue &self) { return logical_not(self); }, "self"_a);

// aten::logical_not_ : (Tensor) -> (Tensor)
m.def("logical_not_", [](const PyAnyTorchTensorValue &self) { return logical_not_(self); }, "self"_a);

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
m.def("logical_or", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return logical_or(self, other); }, "self"_a, "other"_a);

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_or_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return logical_or_(self, other); }, "self"_a, "other"_a);

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
m.def("logical_xor", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return logical_xor(self, other); }, "self"_a, "other"_a);

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_xor_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return logical_xor_(self, other); }, "self"_a, "other"_a);

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
m.def("logsumexp", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) { return logsumexp(self, dim, keepdim); }, "self"_a, "dim"_a, "keepdim"_a = false);

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("lt", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return lt(self, other); }, "self"_a, "other"_a);

// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("lt", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return lt(self, other); }, "self"_a, "other"_a);

// aten::lt.int : (int, int) -> (bool)
m.def("lt", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return lt(a, b); }, "a"_a, "b"_a);

// aten::lt.float : (float, float) -> (bool)
m.def("lt", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) { return lt(a, b); }, "a"_a, "b"_a);

// aten::lt.float_int : (float, int) -> (bool)
m.def("lt", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b) { return lt(a, b); }, "a"_a, "b"_a);

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("lt_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return lt_(self, other); }, "self"_a, "other"_a);

// aten::lt_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("lt_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return lt_(self, other); }, "self"_a, "other"_a);

// aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("masked_fill", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value) { return masked_fill(self, mask, value); }, "self"_a, "mask"_a, "value"_a);

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("masked_fill", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value) { return masked_fill(self, mask, value); }, "self"_a, "mask"_a, "value"_a);

// aten::masked_fill_.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("masked_fill_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value) { return masked_fill_(self, mask, value); }, "self"_a, "mask"_a, "value"_a);

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("masked_fill_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value) { return masked_fill_(self, mask, value); }, "self"_a, "mask"_a, "value"_a);

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
m.def("masked_select", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask) { return masked_select(self, mask); }, "self"_a, "mask"_a);

// aten::matmul : (Tensor, Tensor) -> (Tensor)
m.def("matmul", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return matmul(self, other); }, "self"_a, "other"_a);

// aten::max : (Tensor) -> (Tensor)
m.def("max", [](const PyAnyTorchTensorValue &self) { return max(self); }, "self"_a);

// prim::max.self_int : (int[]) -> (int)
m.def("max", [](const PyAnyTorchListOfTorchIntValue &self) { return max(self); }, "self"_a);

// prim::max.int : (int, int) -> (int)
m.def("max", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return max(a, b); }, "a"_a, "b"_a);

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
m.def("max_pool2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode) { return max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode); }, "self"_a, "kernel_size"_a, "stride"_a = std::vector<int>{}, "padding"_a = std::vector<int>{0, 0}, "dilation"_a = std::vector<int>{1, 1}, "ceil_mode"_a = false);

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
m.def("max_pool2d_with_indices_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices) { return max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices); }, "grad_output"_a, "self"_a, "kernel_size"_a, "stride"_a, "padding"_a, "dilation"_a, "ceil_mode"_a, "indices"_a);

// aten::maximum : (Tensor, Tensor) -> (Tensor)
m.def("maximum", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return maximum(self, other); }, "self"_a, "other"_a);

// aten::mean : (Tensor, int?) -> (Tensor)
m.def("mean", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype) { return mean(self, dtype); }, "self"_a, "dtype"_a = py::none());

// prim::min.self_int : (int[]) -> (int)
m.def("min", [](const PyAnyTorchListOfTorchIntValue &self) { return min(self); }, "self"_a);

// prim::min.int : (int, int) -> (int)
m.def("min", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return min(a, b); }, "a"_a, "b"_a);

// aten::minimum : (Tensor, Tensor) -> (Tensor)
m.def("minimum", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return minimum(self, other); }, "self"_a, "other"_a);

// aten::mish : (Tensor) -> (Tensor)
m.def("mish", [](const PyAnyTorchTensorValue &self) { return mish(self); }, "self"_a);

// aten::mm : (Tensor, Tensor) -> (Tensor)
m.def("mm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2) { return mm(self, mat2); }, "self"_a, "mat2"_a);

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
m.def("mse_loss", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction) { return mse_loss(self, target, reduction); }, "self"_a, "target"_a, "reduction"_a = 1);

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("mul", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return mul(self, other); }, "self"_a, "other"_a);

// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("mul", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return mul(self, other); }, "self"_a, "other"_a);

// aten::mul.int : (int, int) -> (int)
m.def("mul", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return mul(a, b); }, "a"_a, "b"_a);

// aten::mul.float : (float, float) -> (float)
m.def("mul", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) { return mul(a, b); }, "a"_a, "b"_a);

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("mul_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return mul_(self, other); }, "self"_a, "other"_a);

// aten::mul_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("mul_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return mul_(self, other); }, "self"_a, "other"_a);

// aten::mv : (Tensor, Tensor) -> (Tensor)
m.def("mv", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec) { return mv(self, vec); }, "self"_a, "vec"_a);

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
m.def("narrow", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length) { return narrow(self, dim, start, length); }, "self"_a, "dim"_a, "start"_a, "length"_a);

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
m.def("native_dropout_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale) { return native_dropout_backward(grad_output, mask, scale); }, "grad_output"_a, "mask"_a, "scale"_a);

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ne", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return ne(self, other); }, "self"_a, "other"_a);

// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ne", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return ne(self, other); }, "self"_a, "other"_a);

// aten::ne.int_list : (int[], int[]) -> (bool)
m.def("ne", [](const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b) { return ne(a, b); }, "a"_a, "b"_a);

// aten::ne.int : (int, int) -> (bool)
m.def("ne", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return ne(a, b); }, "a"_a, "b"_a);

// aten::ne.float_int : (float, int) -> (bool)
m.def("ne", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b) { return ne(a, b); }, "a"_a, "b"_a);

// aten::ne.bool : (bool, bool) -> (bool)
m.def("ne", [](const PyTorch_BoolValue &a, const PyTorch_BoolValue &b) { return ne(a, b); }, "a"_a, "b"_a);

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ne_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return ne_(self, other); }, "self"_a, "other"_a);

// aten::ne_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ne_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return ne_(self, other); }, "self"_a, "other"_a);

// aten::neg : (Tensor) -> (Tensor)
m.def("neg", [](const PyAnyTorchTensorValue &self) { return neg(self); }, "self"_a);

// aten::neg.int : (int) -> (int)
m.def("neg", [](const PyTorch_IntValue &a) { return neg(a); }, "a"_a);

// aten::neg.float : (float) -> (float)
m.def("neg", [](const PyTorch_FloatValue &a) { return neg(a); }, "a"_a);

// aten::neg_ : (Tensor) -> (Tensor)
m.def("neg_", [](const PyAnyTorchTensorValue &self) { return neg_(self); }, "self"_a);

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_empty", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return new_empty(self, size, dtype, layout, device, pin_memory); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_empty_strided", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return new_empty_strided(self, size, stride, dtype, layout, device, pin_memory); }, "self"_a, "size"_a, "stride"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_ones", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return new_ones(self, size, dtype, layout, device, pin_memory); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_zeros", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return new_zeros(self, size, dtype, layout, device, pin_memory); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::nll_loss_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
m.def("nll_loss_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight) { return nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight); }, "grad_output"_a, "self"_a, "target"_a, "weight"_a = py::none(), py::kw_only(), "reduction"_a, "ignore_index"_a, "total_weight"_a);

// aten::norm.ScalarOpt_dim : (Tensor, Scalar?, int[], bool) -> (Tensor)
m.def("norm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &p, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim) { return norm(self, p, dim, keepdim); }, "self"_a, "p"_a = py::none(), py::kw_only(), "dim"_a, "keepdim"_a = false);

// aten::numel : (Tensor) -> (int)
m.def("numel", [](const PyAnyTorchTensorValue &self) { return numel(self); }, "self"_a);

// aten::numpy_T : (Tensor) -> (Tensor)
m.def("numpy_T", [](const PyAnyTorchTensorValue &self) { return numpy_T(self); }, "self"_a);

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("ones", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return ones(size, dtype, layout, device, pin_memory); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("ones_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) { return ones_like(self, dtype, layout, device, pin_memory, memory_format); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
m.def("pad", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value) { return pad(self, pad__, mode, value); }, "self"_a, "pad__"_a, "mode"_a = "{s}", "value"_a = py::none());

// aten::permute : (Tensor, int[]) -> (Tensor)
m.def("permute", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) { return permute(self, dims); }, "self"_a, "dims"_a);

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
m.def("permute_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims) { return permute_copy(self, dims); }, "self"_a, "dims"_a);

// aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)
m.def("pow", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &exponent) { return pow(self, exponent); }, "self"_a, "exponent"_a);

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
m.def("pow", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent) { return pow(self, exponent); }, "self"_a, "exponent"_a);

// aten::pow.int_float : (int, float) -> (float)
m.def("pow", [](const PyTorch_IntValue &a, const PyTorch_FloatValue &b) { return pow(a, b); }, "a"_a, "b"_a);

// aten::prelu : (Tensor, Tensor) -> (Tensor)
m.def("prelu", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight) { return prelu(self, weight); }, "self"_a, "weight"_a);

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("rand_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) { return rand_like(self, dtype, layout, device, pin_memory, memory_format); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randint", [](const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return randint(low, high, size, dtype, layout, device, pin_memory); }, "low"_a, "high"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randn", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return randn(size, dtype, layout, device, pin_memory); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
m.def("randn", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return randn(size, generator, dtype, layout, device, pin_memory); }, "size"_a, "generator"_a = py::none(), "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("randn_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) { return randn_like(self, dtype, layout, device, pin_memory, memory_format); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::reciprocal : (Tensor) -> (Tensor)
m.def("reciprocal", [](const PyAnyTorchTensorValue &self) { return reciprocal(self); }, "self"_a);

// aten::reciprocal_ : (Tensor) -> (Tensor)
m.def("reciprocal_", [](const PyAnyTorchTensorValue &self) { return reciprocal_(self); }, "self"_a);

// aten::relu : (Tensor) -> (Tensor)
m.def("relu", [](const PyAnyTorchTensorValue &self) { return relu(self); }, "self"_a);

// aten::relu6 : (Tensor) -> (Tensor)
m.def("relu6", [](const PyAnyTorchTensorValue &self) { return relu6(self); }, "self"_a);

// aten::relu6_ : (Tensor) -> (Tensor)
m.def("relu6_", [](const PyAnyTorchTensorValue &self) { return relu6_(self); }, "self"_a);

// aten::relu_ : (Tensor) -> (Tensor)
m.def("relu_", [](const PyAnyTorchTensorValue &self) { return relu_(self); }, "self"_a);

// aten::remainder.int : (int, int) -> (int)
m.def("remainder", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return remainder(a, b); }, "a"_a, "b"_a);

// aten::remainder.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("remainder", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return remainder(self, other); }, "self"_a, "other"_a);

// aten::repeat : (Tensor, int[]) -> (Tensor)
m.def("repeat", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats) { return repeat(self, repeats); }, "self"_a, "repeats"_a);

// aten::reshape : (Tensor, int[]) -> (Tensor)
m.def("reshape", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape) { return reshape(self, shape); }, "self"_a, "shape"_a);

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
m.def("resize_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format) { return resize_(self, size, memory_format); }, "self"_a, "size"_a, "memory_format"_a = py::none());

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
m.def("roll", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims) { return roll(self, shifts, dims); }, "self"_a, "shifts"_a, "dims"_a = std::vector<int>{});

// aten::round : (Tensor) -> (Tensor)
m.def("round", [](const PyAnyTorchTensorValue &self) { return round(self); }, "self"_a);

// aten::round_ : (Tensor) -> (Tensor)
m.def("round_", [](const PyAnyTorchTensorValue &self) { return round_(self); }, "self"_a);

// aten::rsqrt : (Tensor) -> (Tensor)
m.def("rsqrt", [](const PyAnyTorchTensorValue &self) { return rsqrt(self); }, "self"_a);

// aten::rsqrt_ : (Tensor) -> (Tensor)
m.def("rsqrt_", [](const PyAnyTorchTensorValue &self) { return rsqrt_(self); }, "self"_a);

// aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("rsub", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) { return rsub(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_add", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) { return scatter_add(self, dim, index, src); }, "self"_a, "dim"_a, "index"_a, "src"_a);

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_add_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src) { return scatter_add_(self, dim, index, src); }, "self"_a, "dim"_a, "index"_a, "src"_a);

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
m.def("scatter_reduce", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self) { return scatter_reduce(self, dim, index, src, reduce, include_self); }, "self"_a, "dim"_a, "index"_a, "src"_a, "reduce"_a, "include_self"_a = true);

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
m.def("scatter_reduce_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self) { return scatter_reduce_(self, dim, index, src, reduce, include_self); }, "self"_a, "dim"_a, "index"_a, "src"_a, "reduce"_a, "include_self"_a = true);

// aten::select.int : (Tensor, int, int) -> (Tensor)
m.def("select", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) { return select(self, dim, index); }, "self"_a, "dim"_a, "index"_a);

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
m.def("select_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) { return select_copy(self, dim, index); }, "self"_a, "dim"_a, "index"_a);

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
m.def("select_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index) { return select_scatter(self, src, dim, index); }, "self"_a, "src"_a, "dim"_a, "index"_a);

// aten::sigmoid : (Tensor) -> (Tensor)
m.def("sigmoid", [](const PyAnyTorchTensorValue &self) { return sigmoid(self); }, "self"_a);

// aten::sigmoid_ : (Tensor) -> (Tensor)
m.def("sigmoid_", [](const PyAnyTorchTensorValue &self) { return sigmoid_(self); }, "self"_a);

// aten::silu : (Tensor) -> (Tensor)
m.def("silu", [](const PyAnyTorchTensorValue &self) { return silu(self); }, "self"_a);

// aten::silu_ : (Tensor) -> (Tensor)
m.def("silu_", [](const PyAnyTorchTensorValue &self) { return silu_(self); }, "self"_a);

// aten::sin : (Tensor) -> (Tensor)
m.def("sin", [](const PyAnyTorchTensorValue &self) { return sin(self); }, "self"_a);

// aten::sin_ : (Tensor) -> (Tensor)
m.def("sin_", [](const PyAnyTorchTensorValue &self) { return sin_(self); }, "self"_a);

// aten::size : (Tensor) -> (int[])
m.def("size", [](const PyAnyTorchTensorValue &self) { return size(self); }, "self"_a);

// aten::size.int : (Tensor, int) -> (int)
m.def("size", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) { return size(self, dim); }, "self"_a, "dim"_a);

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) { return slice(self, dim, start, end, step); }, "self"_a, "dim"_a = 0, "start"_a = py::none(), "end"_a = py::none(), py::kw_only(), "step"_a = 1);

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) { return slice_copy(self, dim, start, end, step); }, "self"_a, "dim"_a = 0, "start"_a = py::none(), "end"_a = py::none(), py::kw_only(), "step"_a = 1);

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step) { return slice_scatter(self, src, dim, start, end, step); }, "self"_a, "src"_a, "dim"_a = 0, "start"_a = py::none(), "end"_a = py::none(), py::kw_only(), "step"_a = 1);

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
m.def("softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype) { return softmax(self, dim, dtype); }, "self"_a, "dim"_a, "dtype"_a = py::none());

// aten::softplus : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("softplus", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &threshold__) { return softplus(self, beta, threshold__); }, "self"_a, "beta"_a = 1, "threshold__"_a = 20);

// aten::sort.int : (int[], bool) -> ()
m.def("sort", [](const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse) { return sort(self, reverse); }, "self"_a, "reverse"_a = false);

// aten::sqrt : (Tensor) -> (Tensor)
m.def("sqrt", [](const PyAnyTorchTensorValue &self) { return sqrt(self); }, "self"_a);

// aten::sqrt.int : (int) -> (float)
m.def("sqrt", [](const PyTorch_IntValue &a) { return sqrt(a); }, "a"_a);

// aten::sqrt_ : (Tensor) -> (Tensor)
m.def("sqrt_", [](const PyAnyTorchTensorValue &self) { return sqrt_(self); }, "self"_a);

// aten::square : (Tensor) -> (Tensor)
m.def("square", [](const PyAnyTorchTensorValue &self) { return square(self); }, "self"_a);

// aten::square_ : (Tensor) -> (Tensor)
m.def("square_", [](const PyAnyTorchTensorValue &self) { return square_(self); }, "self"_a);

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
m.def("squeeze", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) { return squeeze(self, dim); }, "self"_a, "dim"_a);

// aten::squeeze : (Tensor) -> (Tensor)
m.def("squeeze", [](const PyAnyTorchTensorValue &self) { return squeeze(self); }, "self"_a);

// aten::squeeze_copy : (Tensor) -> (Tensor)
m.def("squeeze_copy", [](const PyAnyTorchTensorValue &self) { return squeeze_copy(self); }, "self"_a);

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
m.def("squeeze_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) { return squeeze_copy(self, dim); }, "self"_a, "dim"_a);

// aten::stack : (Tensor[], int) -> (Tensor)
m.def("stack", [](const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim) { return stack(tensors, dim); }, "tensors"_a, "dim"_a = 0);

// aten::std : (Tensor, bool) -> (Tensor)
m.def("std", [](const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased) { return std(self, unbiased); }, "self"_a, "unbiased"_a = true);

// aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("sub", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha) { return sub(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("sub", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) { return sub(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::sub.int : (int, int) -> (int)
m.def("sub", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b) { return sub(a, b); }, "a"_a, "b"_a);

// aten::sub.float : (float, float) -> (float)
m.def("sub", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b) { return sub(a, b); }, "a"_a, "b"_a);

// aten::sub_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("sub_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha) { return sub_(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::sub_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("sub_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha) { return sub_(self, other, alpha); }, "self"_a, "other"_a, "alpha"_a = 1);

// aten::sum : (Tensor, int?) -> (Tensor)
m.def("sum", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype) { return sum(self, dtype); }, "self"_a, "dtype"_a = py::none());

// aten::t : (Tensor) -> (Tensor)
m.def("t", [](const PyAnyTorchTensorValue &self) { return t(self); }, "self"_a);

// aten::t_copy : (Tensor) -> (Tensor)
m.def("t_copy", [](const PyAnyTorchTensorValue &self) { return t_copy(self); }, "self"_a);

// aten::tanh : (Tensor) -> (Tensor)
m.def("tanh", [](const PyAnyTorchTensorValue &self) { return tanh(self); }, "self"_a);

// aten::tanh_ : (Tensor) -> (Tensor)
m.def("tanh_", [](const PyAnyTorchTensorValue &self) { return tanh_(self); }, "self"_a);

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
m.def("tanh_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output) { return tanh_backward(grad_output, output); }, "grad_output"_a, "output"_a);

// aten::tensor : (t[], int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyAnyTorchListValue &data, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) { return tensor(data, dtype, device, requires_grad); }, "data"_a, "dtype"_a = py::none(), "device"_a = py::none(), py::kw_only(), "requires_grad"_a = false);

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) { return tensor(t, dtype, device, requires_grad); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), py::kw_only(), "requires_grad"_a = false);

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) { return tensor(t, dtype, device, requires_grad); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), py::kw_only(), "requires_grad"_a = false);

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) { return tensor(t, dtype, device, requires_grad); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), py::kw_only(), "requires_grad"_a = false);

// aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("threshold", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value) { return threshold(self, threshold__, value); }, "self"_a, "threshold__"_a, "value"_a);

// aten::threshold_ : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("threshold_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value) { return threshold_(self, threshold__, value); }, "self"_a, "threshold__"_a, "value"_a);

// aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("threshold_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__) { return threshold_backward(grad_output, self, threshold__); }, "grad_output"_a, "self"_a, "threshold__"_a);

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) { return to(self, dtype, non_blocking, copy, memory_format); }, "self"_a, "dtype"_a, "non_blocking"_a = false, "copy"_a = false, "memory_format"_a = py::none());

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) { return to(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "non_blocking"_a = false, "copy"_a = false, "memory_format"_a = py::none());

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) { return to(self, other, non_blocking, copy, memory_format); }, "self"_a, "other"_a, "non_blocking"_a = false, "copy"_a = false, "memory_format"_a = py::none());

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy) { return to(self, device, dtype, non_blocking, copy); }, "self"_a, "device"_a = py::none(), "dtype"_a = py::none(), py::kw_only(), "non_blocking"_a = false, "copy"_a = false);

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format) { return to(self, device, dtype, non_blocking, copy, memory_format); }, "self"_a, "device"_a, "dtype"_a, "non_blocking"_a = false, "copy"_a = false, "memory_format"_a = py::none());

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
m.def("transpose", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1) { return transpose(self, dim0, dim1); }, "self"_a, "dim0"_a, "dim1"_a);

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
m.def("transpose_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1) { return transpose_copy(self, dim0, dim1); }, "self"_a, "dim0"_a, "dim1"_a);

// aten::triu : (Tensor, int) -> (Tensor)
m.def("triu", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal) { return triu(self, diagonal); }, "self"_a, "diagonal"_a = 0);

// aten::triu_ : (Tensor, int) -> (Tensor)
m.def("triu_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal) { return triu_(self, diagonal); }, "self"_a, "diagonal"_a = 0);

// aten::type_as : (Tensor, Tensor) -> (Tensor)
m.def("type_as", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other) { return type_as(self, other); }, "self"_a, "other"_a);

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
m.def("unfold_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step) { return unfold_copy(self, dimension, size, step); }, "self"_a, "dimension"_a, "size"_a, "step"_a);

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
m.def("uniform", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator) { return uniform(self, from, to, generator); }, "self"_a, "from"_a = 0., "to"_a = 1., "generator"_a = py::none());

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
m.def("uniform_", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator) { return uniform_(self, from, to, generator); }, "self"_a, "from"_a = 0., "to"_a = 1., "generator"_a = py::none());

// aten::unsqueeze : (Tensor, int) -> (Tensor)
m.def("unsqueeze", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) { return unsqueeze(self, dim); }, "self"_a, "dim"_a);

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
m.def("unsqueeze_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) { return unsqueeze_(self, dim); }, "self"_a, "dim"_a);

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
m.def("unsqueeze_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim) { return unsqueeze_copy(self, dim); }, "self"_a, "dim"_a);

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
m.def("upsample_nearest2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w) { return upsample_nearest2d(self, output_size, scales_h, scales_w); }, "self"_a, "output_size"_a, "scales_h"_a = py::none(), "scales_w"_a = py::none());

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
m.def("upsample_nearest2d_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w) { return upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h, scales_w); }, "grad_output"_a, "output_size"_a, "input_size"_a, "scales_h"_a = py::none(), "scales_w"_a = py::none());

// aten::var : (Tensor, bool) -> (Tensor)
m.def("var", [](const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased) { return var(self, unbiased); }, "self"_a, "unbiased"_a = true);

// aten::view : (Tensor, int[]) -> (Tensor)
m.def("view", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) { return view(self, size); }, "self"_a, "size"_a);

// aten::view_copy : (Tensor, int[]) -> (Tensor)
m.def("view_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size) { return view_copy(self, size); }, "self"_a, "size"_a);

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
m.def("view_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype) { return view_copy(self, dtype); }, "self"_a, "dtype"_a);

// aten::where.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("where", [](const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchScalarValue &other) { return where(condition, self, other); }, "condition"_a, "self"_a, "other"_a);

// aten::where.ScalarOther : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("where", [](const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other) { return where(condition, self, other); }, "condition"_a, "self"_a, "other"_a);

// aten::where.ScalarSelf : (Tensor, Scalar, Tensor) -> (Tensor)
m.def("where", [](const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &other) { return where(condition, self, other); }, "condition"_a, "self"_a, "other"_a);

// aten::zero : (Tensor) -> (Tensor)
m.def("zero", [](const PyAnyTorchTensorValue &self) { return zero(self); }, "self"_a);

// aten::zero_ : (Tensor) -> (Tensor)
m.def("zero_", [](const PyAnyTorchTensorValue &self) { return zero_(self); }, "self"_a);

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("zeros", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory) { return zeros(size, dtype, layout, device, pin_memory); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("zeros_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format) { return zeros_like(self, dtype, layout, device, pin_memory, memory_format); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());
