
// aten::Bool.Tensor : (Tensor) -> (bool)
m.def("Bool", py::overload_cast<const PyAnyTorchTensorValue &>(&Bool), "a"_a);

// aten::Bool.float : (float) -> (bool)
m.def("Bool", py::overload_cast<const PyTorch_FloatValue &>(&Bool), "a"_a);

// aten::Bool.int : (int) -> (bool)
m.def("Bool", py::overload_cast<const PyTorch_IntValue &>(&Bool), "a"_a);

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
m.def("Delete", py::overload_cast<const PyTorch_DictValue &, const PyTorch_StringValue &>(&Delete), "self"_a, "key"_a);

// aten::Float.Tensor : (Tensor) -> (float)
m.def("Float", py::overload_cast<const PyAnyTorchTensorValue &>(&Float), "a"_a);

// aten::Float.Scalar : (Scalar) -> (float)
m.def("Float", py::overload_cast<const PyAnyTorchScalarValue &>(&Float), "a"_a);

// aten::Float.str : (str) -> (float)
m.def("Float", py::overload_cast<const PyTorch_StringValue &>(&Float), "a"_a);

// aten::FloatImplicit : (Tensor) -> (float)
m.def("FloatImplicit", py::overload_cast<const PyAnyTorchTensorValue &>(&FloatImplicit), "a"_a);

// aten::Int.Tensor : (Tensor) -> (int)
m.def("Int", py::overload_cast<const PyAnyTorchTensorValue &>(&Int), "a"_a);

// aten::Int.float : (float) -> (int)
m.def("Int", py::overload_cast<const PyTorch_FloatValue &>(&Int), "a"_a);

// aten::Int.Scalar : (Scalar) -> (int)
m.def("Int", py::overload_cast<const PyAnyTorchScalarValue &>(&Int), "a"_a);

// aten::Int.bool : (bool) -> (int)
m.def("Int", py::overload_cast<const PyTorch_BoolValue &>(&Int), "a"_a);

// aten::IntImplicit : (Tensor) -> (int)
m.def("IntImplicit", py::overload_cast<const PyAnyTorchTensorValue &>(&IntImplicit), "a"_a);

// prim::NumToTensor.Scalar : (Scalar) -> (Tensor)
m.def("NumToTensor", py::overload_cast<const PyAnyTorchScalarValue &>(&NumToTensor), "a"_a);

// prim::RaiseException : (str, str?) -> ()
m.def("RaiseException", [](const PyTorch_StringValue &msg, const PyDefaultingTorchOptionalStringValue &cls) { return RaiseException(msg, cls.get()); }, "msg"_a, "cls"_a = py::none());

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("__and__", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&__and__), "self"_a, "other"_a);

// aten::__and__.bool : (bool, bool) -> (bool)
m.def("__and__", py::overload_cast<const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&__and__), "a"_a, "b"_a);

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
m.def("__contains__", py::overload_cast<const PyTorch_DictValue &, const PyTorch_StringValue &>(&__contains__), "dict"_a, "key"_a);

// aten::__contains__.int_list : (int[], int) -> (bool)
m.def("__contains__", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyTorch_IntValue &>(&__contains__), "l"_a, "item"_a);

// aten::__derive_index : (int, int, int) -> (int)
m.def("__derive_index", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&__derive_index), "index"_a, "start"_a, "step"_a);

// aten::__not__ : (bool) -> (bool)
m.def("__not__", py::overload_cast<const PyTorch_BoolValue &>(&__not__), "self"_a);

// aten::__range_length : (int, int, int) -> (int)
m.def("__range_length", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&__range_length), "lo"_a, "hi"_a, "step"_a);

// aten::_convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool, bool) -> (Tensor)
m.def("_convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, const PyTorch_BoolValue &allow_tf32) { return _convolution(input, weight, bias.get(), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a, "benchmark"_a, "deterministic"_a, "cudnn_enabled"_a, "allow_tf32"_a);

// aten::_convolution.deprecated : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool) -> (Tensor)
m.def("_convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled) { return _convolution(input, weight, bias.get(), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a, "benchmark"_a, "deterministic"_a, "cudnn_enabled"_a);

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
m.def("_log_softmax", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &>(&_log_softmax), "self"_a, "dim"_a, "half_to_float"_a);

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
m.def("_log_softmax_backward_data", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&_log_softmax_backward_data), "grad_output"_a, "output"_a, "dim"_a, "input_dtype"_a);

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
m.def("_reshape_alias", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&_reshape_alias), "self"_a, "size"_a, "stride"_a);

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
m.def("_reshape_alias_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&_reshape_alias_copy), "self"_a, "size"_a, "stride"_a);

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
m.def("_shape_as_tensor", py::overload_cast<const PyAnyTorchTensorValue &>(&_shape_as_tensor), "self"_a);

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
m.def("_softmax", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &>(&_softmax), "self"_a, "dim"_a, "half_to_float"_a);

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
m.def("_softmax_backward_data", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&_softmax_backward_data), "grad_output"_a, "output"_a, "dim"_a, "input_dtype"_a);

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
m.def("_to_copy", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyDefaultingTorchOptionalIntValue &memory_format) { return _to_copy(self, dtype.get(), layout.get(), device.get(), pin_memory.get(), non_blocking, memory_format.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "non_blocking"_a, "memory_format"_a = py::none());

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
m.def("_unsafe_view", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&_unsafe_view), "self"_a, "size"_a);

// aten::abs : (Tensor) -> (Tensor)
m.def("abs", py::overload_cast<const PyAnyTorchTensorValue &>(&abs), "self"_a);

// aten::abs_ : (Tensor) -> (Tensor)
m.def("abs_", py::overload_cast<const PyAnyTorchTensorValue &>(&abs_), "self"_a);

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
m.def("adaptive_avg_pool2d", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&adaptive_avg_pool2d), "self"_a, "output_size"_a);

// aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("add", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&add), "self"_a, "other"_a, "alpha"_a);

// aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("add", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&add), "self"_a, "other"_a, "alpha"_a);

// aten::add.str : (str, str) -> (str)
m.def("add", py::overload_cast<const PyTorch_StringValue &, const PyTorch_StringValue &>(&add), "a"_a, "b"_a);

// aten::add.int : (int, int) -> (int)
m.def("add", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&add), "a"_a, "b"_a);

// aten::add.float_int : (float, int) -> (float)
m.def("add", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&add), "a"_a, "b"_a);

// aten::add_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("add_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&add_), "self"_a, "other"_a, "alpha"_a);

// aten::add_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("add_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&add_), "self"_a, "other"_a, "alpha"_a);

// aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcdiv", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&addcdiv), "self"_a, "tensor1"_a, "tensor2"_a, "value"_a);

// aten::addcdiv_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcdiv_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&addcdiv_), "self"_a, "tensor1"_a, "tensor2"_a, "value"_a);

// aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcmul", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&addcmul), "self"_a, "tensor1"_a, "tensor2"_a, "value"_a);

// aten::addcmul_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcmul_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&addcmul_), "self"_a, "tensor1"_a, "tensor2"_a, "value"_a);

// aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("addmm", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&addmm), "self"_a, "mat1"_a, "mat2"_a, "beta"_a, "alpha"_a);

// aten::alias_copy : (Tensor) -> (Tensor)
m.def("alias_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&alias_copy), "self"_a);

// aten::all : (Tensor) -> (Tensor)
m.def("all", py::overload_cast<const PyAnyTorchTensorValue &>(&all), "self"_a);

// aten::all.bool : (bool[]) -> (bool)
m.def("all", py::overload_cast<const PyAnyTorchListOfTorchBoolValue &>(&all), "self"_a);

// aten::amax : (Tensor, int[], bool) -> (Tensor)
m.def("amax", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&amax), "self"_a, "dim"_a, "keepdim"_a);

// aten::any : (Tensor) -> (Tensor)
m.def("any", py::overload_cast<const PyAnyTorchTensorValue &>(&any), "self"_a);

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
m.def("any", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &>(&any), "self"_a, "dim"_a, "keepdim"_a);

// aten::any.bool : (bool[]) -> (bool)
m.def("any", py::overload_cast<const PyAnyTorchListOfTorchBoolValue &>(&any), "self"_a);

// aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &end, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return arange(end, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "end"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return arange(start, end, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "start"_a, "end"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return arange(start, end, step, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "start"_a, "end"_a, "step"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::arange.start_out : (Scalar, Scalar, Scalar, Tensor) -> (Tensor)
m.def("arange", py::overload_cast<const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &, const PyAnyTorchTensorValue &>(&arange), "start"_a, "end"_a, "step"_a, "out"_a);

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
m.def("argmax", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim) { return argmax(self, dim.get(), keepdim); }, "self"_a, "dim"_a = py::none(), "keepdim"_a);

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
m.def("as_strided_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyDefaultingTorchOptionalIntValue &storage_offset) { return as_strided_copy(self, size, stride, storage_offset.get()); }, "self"_a, "size"_a, "stride"_a, "storage_offset"_a = py::none());

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
m.def("as_strided_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyDefaultingTorchOptionalIntValue &storage_offset) { return as_strided_scatter(self, src, size, stride, storage_offset.get()); }, "self"_a, "src"_a, "size"_a, "stride"_a, "storage_offset"_a = py::none());

// aten::atan : (Tensor) -> (Tensor)
m.def("atan", py::overload_cast<const PyAnyTorchTensorValue &>(&atan), "self"_a);

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
m.def("atan2", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&atan2), "self"_a, "other"_a);

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
m.def("atan2_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&atan2_), "self"_a, "other"_a);

// aten::atan_ : (Tensor) -> (Tensor)
m.def("atan_", py::overload_cast<const PyAnyTorchTensorValue &>(&atan_), "self"_a);

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
m.def("avg_pool2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyDefaultingTorchOptionalIntValue &divisor_override) { return avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override.get()); }, "self"_a, "kernel_size"_a, "stride"_a, "padding"_a, "ceil_mode"_a, "count_include_pad"_a, "divisor_override"_a = py::none());

// aten::baddbmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("baddbmm", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&baddbmm), "self"_a, "batch1"_a, "batch2"_a, "beta"_a, "alpha"_a);

// aten::baddbmm_ : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("baddbmm_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&baddbmm_), "self"_a, "batch1"_a, "batch2"_a, "beta"_a, "alpha"_a);

// aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)
m.def("batch_norm", [](const PyAnyTorchTensorValue &input, const PyDefaultingTorchOptionalTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyDefaultingTorchOptionalTensorValue &running_mean, const PyDefaultingTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enabled) { return batch_norm(input, weight.get(), bias.get(), running_mean.get(), running_var.get(), training, momentum, eps, cudnn_enabled); }, "input"_a, "weight"_a = py::none(), "bias"_a = py::none(), "running_mean"_a = py::none(), "running_var"_a = py::none(), "training"_a, "momentum"_a, "eps"_a, "cudnn_enabled"_a);

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalGeneratorValue &generator) { return bernoulli(self, generator.get()); }, "self"_a, "generator"_a = py::none());

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyDefaultingTorchOptionalGeneratorValue &generator) { return bernoulli(self, p, generator.get()); }, "self"_a, "p"_a, "generator"_a = py::none());

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyDefaultingTorchOptionalGeneratorValue &generator) { return bernoulli(self, p, generator.get()); }, "self"_a, "p"_a, "generator"_a = py::none());

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
m.def("bernoulli_", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyDefaultingTorchOptionalGeneratorValue &generator) { return bernoulli_(self, p, generator.get()); }, "self"_a, "p"_a, "generator"_a = py::none());

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
m.def("bernoulli_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyDefaultingTorchOptionalGeneratorValue &generator) { return bernoulli_(self, p, generator.get()); }, "self"_a, "p"_a, "generator"_a = py::none());

// aten::bincount : (Tensor, Tensor?, int) -> (Tensor)
m.def("bincount", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalTensorValue &weights, const PyTorch_IntValue &minlength) { return bincount(self, weights.get(), minlength); }, "self"_a, "weights"_a = py::none(), "minlength"_a);

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_and", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_and), "self"_a, "other"_a);

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_and_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_and_), "self"_a, "other"_a);

// aten::bitwise_not : (Tensor) -> (Tensor)
m.def("bitwise_not", py::overload_cast<const PyAnyTorchTensorValue &>(&bitwise_not), "self"_a);

// aten::bitwise_not_ : (Tensor) -> (Tensor)
m.def("bitwise_not_", py::overload_cast<const PyAnyTorchTensorValue &>(&bitwise_not_), "self"_a);

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_or", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_or), "self"_a, "other"_a);

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_or_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_or_), "self"_a, "other"_a);

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_xor", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_xor), "self"_a, "other"_a);

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_xor_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_xor_), "self"_a, "other"_a);

// aten::bmm : (Tensor, Tensor) -> (Tensor)
m.def("bmm", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bmm), "self"_a, "mat2"_a);

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
m.def("broadcast_to", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&broadcast_to), "self"_a, "size"_a);

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
m.def("bucketize", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&bucketize), "self"_a, "boundaries"_a, "out_int32"_a, "right"_a);

// aten::cat : (Tensor[], int) -> (Tensor)
m.def("cat", py::overload_cast<const PyAnyTorchListOfTensorValue &, const PyTorch_IntValue &>(&cat), "tensors"_a, "dim"_a);

// aten::ceil : (Tensor) -> (Tensor)
m.def("ceil", py::overload_cast<const PyAnyTorchTensorValue &>(&ceil), "self"_a);

// aten::ceil.float : (float) -> (int)
m.def("ceil", py::overload_cast<const PyTorch_FloatValue &>(&ceil), "a"_a);

// aten::ceil_ : (Tensor) -> (Tensor)
m.def("ceil_", py::overload_cast<const PyAnyTorchTensorValue &>(&ceil_), "self"_a);

// aten::clamp.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
m.def("clamp", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalTensorValue &min, const PyDefaultingTorchOptionalTensorValue &max) { return clamp(self, min.get(), max.get()); }, "self"_a, "min"_a = py::none(), "max"_a = py::none());

// aten::clamp_.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
m.def("clamp_", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalTensorValue &min, const PyDefaultingTorchOptionalTensorValue &max) { return clamp_(self, min.get(), max.get()); }, "self"_a, "min"_a = py::none(), "max"_a = py::none());

// aten::clamp_max : (Tensor, Scalar) -> (Tensor)
m.def("clamp_max", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&clamp_max), "self"_a, "max"_a);

// aten::clamp_max_ : (Tensor, Scalar) -> (Tensor)
m.def("clamp_max_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&clamp_max_), "self"_a, "max"_a);

// aten::clamp_min : (Tensor, Scalar) -> (Tensor)
m.def("clamp_min", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&clamp_min), "self"_a, "min"_a);

// aten::clamp_min_ : (Tensor, Scalar) -> (Tensor)
m.def("clamp_min_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&clamp_min_), "self"_a, "min"_a);

// aten::clone : (Tensor, int?) -> (Tensor)
m.def("clone", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &memory_format) { return clone(self, memory_format.get()); }, "self"_a, "memory_format"_a = py::none());

// aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)
m.def("constant_pad_nd", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchScalarValue &>(&constant_pad_nd), "self"_a, "pad__"_a, "value"_a);

// aten::contiguous : (Tensor, int) -> (Tensor)
m.def("contiguous", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&contiguous), "self"_a, "memory_format"_a);

// aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)
m.def("conv2d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_IntValue &groups) { return conv2d(input, weight, bias.get(), stride, padding, dilation, groups); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "dilation"_a, "groups"_a);

// aten::conv_transpose1d : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose1d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) { return conv_transpose1d(input, weight, bias.get(), stride, padding, output_padding, groups, dilation); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "output_padding"_a, "groups"_a, "dilation"_a);

// aten::conv_transpose2d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose2d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) { return conv_transpose2d(input, weight, bias.get(), stride, padding, output_padding, groups, dilation); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "output_padding"_a, "groups"_a, "dilation"_a);

// aten::conv_transpose3d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose3d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation) { return conv_transpose3d(input, weight, bias.get(), stride, padding, output_padding, groups, dilation); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "output_padding"_a, "groups"_a, "dilation"_a);

// prims::convert_element_type : (Tensor, int) -> (Tensor)
m.def("convert_element_type", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&convert_element_type), "a"_a, "dtype"_a);

// aten::convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)
m.def("convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups) { return convolution(input, weight, bias.get(), stride, padding, dilation, transposed, output_padding, groups); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a);

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
m.def("copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&copy), "self"_a, "src"_a, "non_blocking"_a);

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
m.def("copy_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&copy_), "self"_a, "src"_a, "non_blocking"_a);

// aten::cos : (Tensor) -> (Tensor)
m.def("cos", py::overload_cast<const PyAnyTorchTensorValue &>(&cos), "self"_a);

// aten::cos_ : (Tensor) -> (Tensor)
m.def("cos_", py::overload_cast<const PyAnyTorchTensorValue &>(&cos_), "self"_a);

// aten::cpu : (Tensor) -> (Tensor)
m.def("cpu", py::overload_cast<const PyAnyTorchTensorValue &>(&cpu), "self"_a);

// aten::cross_entropy_loss : (Tensor, Tensor, Tensor?, int, int, float) -> (Tensor)
m.def("cross_entropy_loss", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyDefaultingTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyTorch_FloatValue &label_smoothing) { return cross_entropy_loss(self, target, weight.get(), reduction, ignore_index, label_smoothing); }, "self"_a, "target"_a, "weight"_a = py::none(), "reduction"_a, "ignore_index"_a, "label_smoothing"_a);

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
m.def("cumsum", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyDefaultingTorchOptionalIntValue &dtype) { return cumsum(self, dim, dtype.get()); }, "self"_a, "dim"_a, "dtype"_a = py::none());

// aten::detach : (Tensor) -> (Tensor)
m.def("detach", py::overload_cast<const PyAnyTorchTensorValue &>(&detach), "self"_a);

// aten::detach_copy : (Tensor) -> (Tensor)
m.def("detach_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&detach_copy), "self"_a);

// prim::device : (Tensor) -> (Device)
m.def("device", py::overload_cast<const PyAnyTorchTensorValue &>(&device), "a"_a);

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
m.def("diagonal_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&diagonal_copy), "self"_a, "offset"_a, "dim1"_a, "dim2"_a);

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
m.def("diagonal_scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&diagonal_scatter), "self"_a, "src"_a, "offset"_a, "dim1"_a, "dim2"_a);

// aten::dim : (Tensor) -> (int)
m.def("dim", py::overload_cast<const PyAnyTorchTensorValue &>(&dim), "self"_a);

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("div", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&div), "self"_a, "other"_a);

// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("div", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&div), "self"_a, "other"_a);

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
m.def("div", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyDefaultingTorchOptionalStringValue &rounding_mode) { return div(self, other, rounding_mode.get()); }, "self"_a, "other"_a, "rounding_mode"_a = py::none());

// aten::div.int : (int, int) -> (float)
m.def("div", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&div), "a"_a, "b"_a);

// aten::div.float : (float, float) -> (float)
m.def("div", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&div), "a"_a, "b"_a);

// aten::div : (Scalar, Scalar) -> (float)
m.def("div", py::overload_cast<const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&div), "a"_a, "b"_a);

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("div_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&div_), "self"_a, "other"_a);

// aten::div_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("div_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&div_), "self"_a, "other"_a);

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
m.def("div_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyDefaultingTorchOptionalStringValue &rounding_mode) { return div_(self, other, rounding_mode.get()); }, "self"_a, "other"_a, "rounding_mode"_a = py::none());

// aten::dropout : (Tensor, float, bool) -> (Tensor)
m.def("dropout", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_FloatValue &, const PyTorch_BoolValue &>(&dropout), "input"_a, "p"_a, "train"_a);

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
m.def("dropout_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_FloatValue &, const PyTorch_BoolValue &>(&dropout_), "self"_a, "p"_a, "train"_a);

// prim::dtype : (Tensor) -> (int)
m.def("dtype", py::overload_cast<const PyAnyTorchTensorValue &>(&dtype), "a"_a);

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
m.def("embedding", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&embedding), "weight"_a, "indices"_a, "padding_idx"_a, "scale_grad_by_freq"_a, "sparse"_a);

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
m.def("embedding_dense_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &>(&embedding_dense_backward), "grad_output"_a, "indices"_a, "num_weights"_a, "padding_idx"_a, "scale_grad_by_freq"_a);

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("empty", [](const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyDefaultingTorchOptionalIntValue &memory_format) { return empty(size, dtype.get(), layout.get(), device.get(), pin_memory.get(), memory_format.get()); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("empty_like", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyDefaultingTorchOptionalIntValue &memory_format) { return empty_like(self, dtype.get(), layout.get(), device.get(), pin_memory.get(), memory_format.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("eq", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&eq), "self"_a, "other"_a);

// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("eq", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&eq), "self"_a, "other"_a);

// aten::eq.int_list : (int[], int[]) -> (bool)
m.def("eq", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&eq), "a"_a, "b"_a);

// aten::eq.str : (str, str) -> (bool)
m.def("eq", py::overload_cast<const PyTorch_StringValue &, const PyTorch_StringValue &>(&eq), "a"_a, "b"_a);

// aten::eq.int : (int, int) -> (bool)
m.def("eq", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&eq), "a"_a, "b"_a);

// aten::eq.float : (float, float) -> (bool)
m.def("eq", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&eq), "a"_a, "b"_a);

// aten::eq.device : (Device, Device) -> (bool)
m.def("eq", py::overload_cast<const PyTorch_DeviceValue &, const PyTorch_DeviceValue &>(&eq), "a"_a, "b"_a);

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("eq_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&eq_), "self"_a, "other"_a);

// aten::eq_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("eq_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&eq_), "self"_a, "other"_a);

// aten::erf : (Tensor) -> (Tensor)
m.def("erf", py::overload_cast<const PyAnyTorchTensorValue &>(&erf), "self"_a);

// aten::erf_ : (Tensor) -> (Tensor)
m.def("erf_", py::overload_cast<const PyAnyTorchTensorValue &>(&erf_), "self"_a);

// aten::exp : (Tensor) -> (Tensor)
m.def("exp", py::overload_cast<const PyAnyTorchTensorValue &>(&exp), "self"_a);

// aten::exp_ : (Tensor) -> (Tensor)
m.def("exp_", py::overload_cast<const PyAnyTorchTensorValue &>(&exp_), "self"_a);

// aten::expand : (Tensor, int[], bool) -> (Tensor)
m.def("expand", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&expand), "self"_a, "size"_a, "implicit"_a);

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
m.def("expand_as", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&expand_as), "self"_a, "other"_a);

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
m.def("expand_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&expand_copy), "self"_a, "size"_a, "implicit"_a);

// aten::expm1 : (Tensor) -> (Tensor)
m.def("expm1", py::overload_cast<const PyAnyTorchTensorValue &>(&expm1), "self"_a);

// aten::expm1_ : (Tensor) -> (Tensor)
m.def("expm1_", py::overload_cast<const PyAnyTorchTensorValue &>(&expm1_), "self"_a);

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
m.def("fft_fft", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyDefaultingTorchOptionalStringValue &norm) { return fft_fft(self, n.get(), dim, norm.get()); }, "self"_a, "n"_a = py::none(), "dim"_a, "norm"_a = py::none());

// aten::fill.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fill", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&fill), "self"_a, "value"_a);

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("fill", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&fill), "self"_a, "value"_a);

// aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fill_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&fill_), "self"_a, "value"_a);

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("fill_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&fill_), "self"_a, "value"_a);

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
m.def("flatten", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&flatten), "self"_a, "start_dim"_a, "end_dim"_a);

// aten::flip : (Tensor, int[]) -> (Tensor)
m.def("flip", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&flip), "self"_a, "dims"_a);

// aten::floor : (Tensor) -> (Tensor)
m.def("floor", py::overload_cast<const PyAnyTorchTensorValue &>(&floor), "self"_a);

// aten::floor_ : (Tensor) -> (Tensor)
m.def("floor_", py::overload_cast<const PyAnyTorchTensorValue &>(&floor_), "self"_a);

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
m.def("floor_divide", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&floor_divide), "self"_a, "other"_a);

// aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("floor_divide", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&floor_divide), "self"_a, "other"_a);

// aten::floordiv.int : (int, int) -> (int)
m.def("floordiv", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&floordiv), "a"_a, "b"_a);

// aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fmod", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&fmod), "self"_a, "other"_a);

// aten::fmod_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fmod_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&fmod_), "self"_a, "other"_a);

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
m.def("frobenius_norm", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&frobenius_norm), "self"_a, "dim"_a, "keepdim"_a);

// aten::full : (int[], Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("full", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchScalarValue &fill_value, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return full(size, fill_value, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "size"_a, "fill_value"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::full_like : (Tensor, Scalar, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("full_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &fill_value, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyDefaultingTorchOptionalIntValue &memory_format) { return full_like(self, fill_value, dtype.get(), layout.get(), device.get(), pin_memory.get(), memory_format.get()); }, "self"_a, "fill_value"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
m.def("gather", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&gather), "self"_a, "dim"_a, "index"_a, "sparse_grad"_a);

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ge", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&ge), "self"_a, "other"_a);

// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ge", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&ge), "self"_a, "other"_a);

// aten::ge.int : (int, int) -> (bool)
m.def("ge", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&ge), "a"_a, "b"_a);

// aten::ge.float : (float, float) -> (bool)
m.def("ge", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&ge), "a"_a, "b"_a);

// aten::ge.float_int : (float, int) -> (bool)
m.def("ge", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&ge), "a"_a, "b"_a);

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ge_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&ge_), "self"_a, "other"_a);

// aten::ge_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ge_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&ge_), "self"_a, "other"_a);

// aten::gelu : (Tensor, str) -> (Tensor)
m.def("gelu", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_StringValue &>(&gelu), "self"_a, "approximate"_a);

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
m.def("gelu_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_StringValue &>(&gelu_backward), "grad_output"_a, "self"_a, "approximate"_a);

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("gt", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&gt), "self"_a, "other"_a);

// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("gt", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&gt), "self"_a, "other"_a);

// aten::gt.int : (int, int) -> (bool)
m.def("gt", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&gt), "a"_a, "b"_a);

// aten::gt.float : (float, float) -> (bool)
m.def("gt", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&gt), "a"_a, "b"_a);

// aten::gt.float_int : (float, int) -> (bool)
m.def("gt", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&gt), "a"_a, "b"_a);

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("gt_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&gt_), "self"_a, "other"_a);

// aten::gt_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("gt_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&gt_), "self"_a, "other"_a);

// aten::hardsigmoid : (Tensor) -> (Tensor)
m.def("hardsigmoid", py::overload_cast<const PyAnyTorchTensorValue &>(&hardsigmoid), "self"_a);

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
m.def("hardsigmoid_", py::overload_cast<const PyAnyTorchTensorValue &>(&hardsigmoid_), "self"_a);

// aten::hardswish : (Tensor) -> (Tensor)
m.def("hardswish", py::overload_cast<const PyAnyTorchTensorValue &>(&hardswish), "self"_a);

// aten::hardswish_ : (Tensor) -> (Tensor)
m.def("hardswish_", py::overload_cast<const PyAnyTorchTensorValue &>(&hardswish_), "self"_a);

// aten::hardtanh : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&hardtanh), "self"_a, "min_val"_a, "max_val"_a);

// aten::hardtanh_ : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&hardtanh_), "self"_a, "min_val"_a, "max_val"_a);

// aten::hardtanh_backward : (Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&hardtanh_backward), "grad_output"_a, "self"_a, "min_val"_a, "max_val"_a);

// aten::imag : (Tensor) -> (Tensor)
m.def("imag", py::overload_cast<const PyAnyTorchTensorValue &>(&imag), "self"_a);

// aten::index.Tensor_hacked_twin : (Tensor, Tensor[]) -> (Tensor)
m.def("index", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTensorValue &>(&index), "self"_a, "indices"_a);

// aten::index_put.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
m.def("index_put", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&index_put), "self"_a, "indices"_a, "values"_a, "accumulate"_a);

// aten::index_put_.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
m.def("index_put_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&index_put_), "self"_a, "indices"_a, "values"_a, "accumulate"_a);

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
m.def("index_select", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &>(&index_select), "self"_a, "dim"_a, "index"_a);

// aten::is_floating_point : (Tensor) -> (bool)
m.def("is_floating_point", py::overload_cast<const PyAnyTorchTensorValue &>(&is_floating_point), "self"_a);

// aten::join : (str, str[]) -> (str)
m.def("join", py::overload_cast<const PyTorch_StringValue &, const PyAnyTorchListOfTorchStringValue &>(&join), "self"_a, "values"_a);

// aten::keys.str : (Dict(str, t)) -> (str[])
m.def("keys", py::overload_cast<const PyTorch_DictValue &>(&keys), "self"_a);

// aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)
m.def("layer_norm", [](const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyDefaultingTorchOptionalTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enable) { return layer_norm(input, normalized_shape, weight.get(), bias.get(), eps, cudnn_enable); }, "input"_a, "normalized_shape"_a, "weight"_a = py::none(), "bias"_a = py::none(), "eps"_a, "cudnn_enable"_a);

// prim::layout : (Tensor) -> (int)
m.def("layout", py::overload_cast<const PyAnyTorchTensorValue &>(&layout), "a"_a);

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("le", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&le), "self"_a, "other"_a);

// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("le", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&le), "self"_a, "other"_a);

// aten::le.int : (int, int) -> (bool)
m.def("le", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&le), "a"_a, "b"_a);

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("le_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&le_), "self"_a, "other"_a);

// aten::le_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("le_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&le_), "self"_a, "other"_a);

// aten::leaky_relu : (Tensor, Scalar) -> (Tensor)
m.def("leaky_relu", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&leaky_relu), "self"_a, "negative_slope"_a);

// aten::leaky_relu_ : (Tensor, Scalar) -> (Tensor)
m.def("leaky_relu_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&leaky_relu_), "self"_a, "negative_slope"_a);

// aten::leaky_relu_backward : (Tensor, Tensor, Scalar, bool) -> (Tensor)
m.def("leaky_relu_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyTorch_BoolValue &>(&leaky_relu_backward), "grad_output"_a, "self"_a, "negative_slope"_a, "self_is_result"_a);

// aten::len.Tensor : (Tensor) -> (int)
m.def("len", py::overload_cast<const PyAnyTorchTensorValue &>(&len), "t"_a);

// aten::len.str : (str) -> (int)
m.def("len", py::overload_cast<const PyTorch_StringValue &>(&len), "s"_a);

// aten::len.t : (t[]) -> (int)
m.def("len", py::overload_cast<const PyAnyTorchListValue &>(&len), "a"_a);

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("lerp", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&lerp), "self"_a, "end"_a, "weight"_a);

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("lerp_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&lerp_), "self"_a, "end"_a, "weight"_a);

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
m.def("lift_fresh_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&lift_fresh_copy), "self"_a);

// aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)
m.def("linear", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyDefaultingTorchOptionalTensorValue &bias) { return linear(input, weight, bias.get()); }, "input"_a, "weight"_a, "bias"_a = py::none());

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
m.def("linear", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_LinearParamsValue &, const PyTorch_FloatValue &, const PyTorch_IntValue &>(&linear), "X"_a, "W_prepack"_a, "Y_scale_i"_a, "Y_zero_point_i"_a);

// aten::log : (Tensor) -> (Tensor)
m.def("log", py::overload_cast<const PyAnyTorchTensorValue &>(&log), "self"_a);

// aten::log.int : (int) -> (float)
m.def("log", py::overload_cast<const PyTorch_IntValue &>(&log), "a"_a);

// aten::log1p : (Tensor) -> (Tensor)
m.def("log1p", py::overload_cast<const PyAnyTorchTensorValue &>(&log1p), "self"_a);

// aten::log1p_ : (Tensor) -> (Tensor)
m.def("log1p_", py::overload_cast<const PyAnyTorchTensorValue &>(&log1p_), "self"_a);

// aten::log2 : (Tensor) -> (Tensor)
m.def("log2", py::overload_cast<const PyAnyTorchTensorValue &>(&log2), "self"_a);

// aten::log2_ : (Tensor) -> (Tensor)
m.def("log2_", py::overload_cast<const PyAnyTorchTensorValue &>(&log2_), "self"_a);

// aten::log_ : (Tensor) -> (Tensor)
m.def("log_", py::overload_cast<const PyAnyTorchTensorValue &>(&log_), "self"_a);

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
m.def("log_softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyDefaultingTorchOptionalIntValue &dtype) { return log_softmax(self, dim, dtype.get()); }, "self"_a, "dim"_a, "dtype"_a = py::none());

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
m.def("logical_and", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_and), "self"_a, "other"_a);

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_and_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_and_), "self"_a, "other"_a);

// aten::logical_not : (Tensor) -> (Tensor)
m.def("logical_not", py::overload_cast<const PyAnyTorchTensorValue &>(&logical_not), "self"_a);

// aten::logical_not_ : (Tensor) -> (Tensor)
m.def("logical_not_", py::overload_cast<const PyAnyTorchTensorValue &>(&logical_not_), "self"_a);

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
m.def("logical_or", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_or), "self"_a, "other"_a);

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_or_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_or_), "self"_a, "other"_a);

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
m.def("logical_xor", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_xor), "self"_a, "other"_a);

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_xor_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_xor_), "self"_a, "other"_a);

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
m.def("logsumexp", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&logsumexp), "self"_a, "dim"_a, "keepdim"_a);

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("lt", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&lt), "self"_a, "other"_a);

// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("lt", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&lt), "self"_a, "other"_a);

// aten::lt.int : (int, int) -> (bool)
m.def("lt", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&lt), "a"_a, "b"_a);

// aten::lt.float : (float, float) -> (bool)
m.def("lt", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&lt), "a"_a, "b"_a);

// aten::lt.float_int : (float, int) -> (bool)
m.def("lt", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&lt), "a"_a, "b"_a);

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("lt_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&lt_), "self"_a, "other"_a);

// aten::lt_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("lt_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&lt_), "self"_a, "other"_a);

// aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("masked_fill", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&masked_fill), "self"_a, "mask"_a, "value"_a);

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("masked_fill", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&masked_fill), "self"_a, "mask"_a, "value"_a);

// aten::masked_fill_.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("masked_fill_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&masked_fill_), "self"_a, "mask"_a, "value"_a);

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("masked_fill_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&masked_fill_), "self"_a, "mask"_a, "value"_a);

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
m.def("masked_select", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&masked_select), "self"_a, "mask"_a);

// aten::matmul : (Tensor, Tensor) -> (Tensor)
m.def("matmul", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&matmul), "self"_a, "other"_a);

// aten::max : (Tensor) -> (Tensor)
m.def("max", py::overload_cast<const PyAnyTorchTensorValue &>(&max), "self"_a);

// prim::max.self_int : (int[]) -> (int)
m.def("max", py::overload_cast<const PyAnyTorchListOfTorchIntValue &>(&max), "self"_a);

// prim::max.int : (int, int) -> (int)
m.def("max", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&max), "a"_a, "b"_a);

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
m.def("max_pool2d", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&max_pool2d), "self"_a, "kernel_size"_a, "stride"_a, "padding"_a, "dilation"_a, "ceil_mode"_a);

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
m.def("max_pool2d_with_indices_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &, const PyAnyTorchTensorValue &>(&max_pool2d_with_indices_backward), "grad_output"_a, "self"_a, "kernel_size"_a, "stride"_a, "padding"_a, "dilation"_a, "ceil_mode"_a, "indices"_a);

// aten::maximum : (Tensor, Tensor) -> (Tensor)
m.def("maximum", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&maximum), "self"_a, "other"_a);

// aten::mean : (Tensor, int?) -> (Tensor)
m.def("mean", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype) { return mean(self, dtype.get()); }, "self"_a, "dtype"_a = py::none());

// prim::min.self_int : (int[]) -> (int)
m.def("min", py::overload_cast<const PyAnyTorchListOfTorchIntValue &>(&min), "self"_a);

// prim::min.int : (int, int) -> (int)
m.def("min", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&min), "a"_a, "b"_a);

// aten::minimum : (Tensor, Tensor) -> (Tensor)
m.def("minimum", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&minimum), "self"_a, "other"_a);

// aten::mish : (Tensor) -> (Tensor)
m.def("mish", py::overload_cast<const PyAnyTorchTensorValue &>(&mish), "self"_a);

// aten::mm : (Tensor, Tensor) -> (Tensor)
m.def("mm", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&mm), "self"_a, "mat2"_a);

// aten::movedim.int : (Tensor, int, int) -> (Tensor)
m.def("movedim", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&movedim), "self"_a, "source"_a, "destination"_a);

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
m.def("mse_loss", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&mse_loss), "self"_a, "target"_a, "reduction"_a);

// aten::mse_loss_backward : (Tensor, Tensor, Tensor, int) -> (Tensor)
m.def("mse_loss_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&mse_loss_backward), "grad_output"_a, "self"_a, "target"_a, "reduction"_a);

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("mul", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&mul), "self"_a, "other"_a);

// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("mul", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&mul), "self"_a, "other"_a);

// aten::mul.int : (int, int) -> (int)
m.def("mul", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&mul), "a"_a, "b"_a);

// aten::mul.float : (float, float) -> (float)
m.def("mul", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&mul), "a"_a, "b"_a);

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("mul_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&mul_), "self"_a, "other"_a);

// aten::mul_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("mul_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&mul_), "self"_a, "other"_a);

// aten::mv : (Tensor, Tensor) -> (Tensor)
m.def("mv", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&mv), "self"_a, "vec"_a);

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
m.def("narrow", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&narrow), "self"_a, "dim"_a, "start"_a, "length"_a);

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
m.def("native_dropout_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_FloatValue &>(&native_dropout_backward), "grad_output"_a, "mask"_a, "scale"_a);

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ne", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&ne), "self"_a, "other"_a);

// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ne", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&ne), "self"_a, "other"_a);

// aten::ne.int_list : (int[], int[]) -> (bool)
m.def("ne", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&ne), "a"_a, "b"_a);

// aten::ne.int : (int, int) -> (bool)
m.def("ne", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&ne), "a"_a, "b"_a);

// aten::ne.float_int : (float, int) -> (bool)
m.def("ne", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&ne), "a"_a, "b"_a);

// aten::ne.bool : (bool, bool) -> (bool)
m.def("ne", py::overload_cast<const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&ne), "a"_a, "b"_a);

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ne_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&ne_), "self"_a, "other"_a);

// aten::ne_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ne_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&ne_), "self"_a, "other"_a);

// aten::neg : (Tensor) -> (Tensor)
m.def("neg", py::overload_cast<const PyAnyTorchTensorValue &>(&neg), "self"_a);

// aten::neg.int : (int) -> (int)
m.def("neg", py::overload_cast<const PyTorch_IntValue &>(&neg), "a"_a);

// aten::neg.float : (float) -> (float)
m.def("neg", py::overload_cast<const PyTorch_FloatValue &>(&neg), "a"_a);

// aten::neg_ : (Tensor) -> (Tensor)
m.def("neg_", py::overload_cast<const PyAnyTorchTensorValue &>(&neg_), "self"_a);

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_empty", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return new_empty(self, size, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_empty_strided", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return new_empty_strided(self, size, stride, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "self"_a, "size"_a, "stride"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_ones", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return new_ones(self, size, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_zeros", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return new_zeros(self, size, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::nll_loss2d_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
m.def("nll_loss2d_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyDefaultingTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight) { return nll_loss2d_backward(grad_output, self, target, weight.get(), reduction, ignore_index, total_weight); }, "grad_output"_a, "self"_a, "target"_a, "weight"_a = py::none(), "reduction"_a, "ignore_index"_a, "total_weight"_a);

// aten::nll_loss_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
m.def("nll_loss_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyDefaultingTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight) { return nll_loss_backward(grad_output, self, target, weight.get(), reduction, ignore_index, total_weight); }, "grad_output"_a, "self"_a, "target"_a, "weight"_a = py::none(), "reduction"_a, "ignore_index"_a, "total_weight"_a);

// aten::numel : (Tensor) -> (int)
m.def("numel", py::overload_cast<const PyAnyTorchTensorValue &>(&numel), "self"_a);

// aten::numpy_T : (Tensor) -> (Tensor)
m.def("numpy_T", py::overload_cast<const PyAnyTorchTensorValue &>(&numpy_T), "self"_a);

// aten::one_hot : (Tensor, int) -> (Tensor)
m.def("one_hot", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&one_hot), "self"_a, "num_classes"_a);

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("ones", [](const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return ones(size, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("ones_like", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyDefaultingTorchOptionalIntValue &memory_format) { return ones_like(self, dtype.get(), layout.get(), device.get(), pin_memory.get(), memory_format.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
m.def("pad", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyTorch_StringValue &mode, const PyDefaultingTorchOptionalFloatValue &value) { return pad(self, pad__, mode, value.get()); }, "self"_a, "pad__"_a, "mode"_a, "value"_a = py::none());

// aten::permute : (Tensor, int[]) -> (Tensor)
m.def("permute", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&permute), "self"_a, "dims"_a);

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
m.def("permute_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&permute_copy), "self"_a, "dims"_a);

// aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)
m.def("pow", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&pow), "self"_a, "exponent"_a);

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
m.def("pow", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&pow), "self"_a, "exponent"_a);

// aten::pow.Scalar : (Scalar, Tensor) -> (Tensor)
m.def("pow", py::overload_cast<const PyAnyTorchScalarValue &, const PyAnyTorchTensorValue &>(&pow), "self"_a, "exponent"_a);

// aten::pow.int_float : (int, float) -> (float)
m.def("pow", py::overload_cast<const PyTorch_IntValue &, const PyTorch_FloatValue &>(&pow), "a"_a, "b"_a);

// aten::prelu : (Tensor, Tensor) -> (Tensor)
m.def("prelu", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&prelu), "self"_a, "weight"_a);

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("rand_like", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyDefaultingTorchOptionalIntValue &memory_format) { return rand_like(self, dtype.get(), layout.get(), device.get(), pin_memory.get(), memory_format.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randint", [](const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return randint(low, high, size, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "low"_a, "high"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::randint : (int, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randint", [](const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return randint(high, size, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "high"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randn", [](const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return randn(size, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
m.def("randn", [](const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalGeneratorValue &generator, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return randn(size, generator.get(), dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "size"_a, "generator"_a = py::none(), "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("randn_like", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyDefaultingTorchOptionalIntValue &memory_format) { return randn_like(self, dtype.get(), layout.get(), device.get(), pin_memory.get(), memory_format.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());

// aten::real : (Tensor) -> (Tensor)
m.def("real", py::overload_cast<const PyAnyTorchTensorValue &>(&real), "self"_a);

// aten::reciprocal : (Tensor) -> (Tensor)
m.def("reciprocal", py::overload_cast<const PyAnyTorchTensorValue &>(&reciprocal), "self"_a);

// aten::reciprocal_ : (Tensor) -> (Tensor)
m.def("reciprocal_", py::overload_cast<const PyAnyTorchTensorValue &>(&reciprocal_), "self"_a);

// aten::relu : (Tensor) -> (Tensor)
m.def("relu", py::overload_cast<const PyAnyTorchTensorValue &>(&relu), "self"_a);

// aten::relu6 : (Tensor) -> (Tensor)
m.def("relu6", py::overload_cast<const PyAnyTorchTensorValue &>(&relu6), "self"_a);

// aten::relu6_ : (Tensor) -> (Tensor)
m.def("relu6_", py::overload_cast<const PyAnyTorchTensorValue &>(&relu6_), "self"_a);

// aten::relu_ : (Tensor) -> (Tensor)
m.def("relu_", py::overload_cast<const PyAnyTorchTensorValue &>(&relu_), "self"_a);

// aten::remainder.int : (int, int) -> (int)
m.def("remainder", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&remainder), "a"_a, "b"_a);

// aten::remainder.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("remainder", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&remainder), "self"_a, "other"_a);

// aten::repeat : (Tensor, int[]) -> (Tensor)
m.def("repeat", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&repeat), "self"_a, "repeats"_a);

// aten::reshape : (Tensor, int[]) -> (Tensor)
m.def("reshape", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&reshape), "self"_a, "shape"_a);

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
m.def("resize_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &memory_format) { return resize_(self, size, memory_format.get()); }, "self"_a, "size"_a, "memory_format"_a = py::none());

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
m.def("roll", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&roll), "self"_a, "shifts"_a, "dims"_a);

// aten::round : (Tensor) -> (Tensor)
m.def("round", py::overload_cast<const PyAnyTorchTensorValue &>(&round), "self"_a);

// aten::round_ : (Tensor) -> (Tensor)
m.def("round_", py::overload_cast<const PyAnyTorchTensorValue &>(&round_), "self"_a);

// aten::rsqrt : (Tensor) -> (Tensor)
m.def("rsqrt", py::overload_cast<const PyAnyTorchTensorValue &>(&rsqrt), "self"_a);

// aten::rsqrt_ : (Tensor) -> (Tensor)
m.def("rsqrt_", py::overload_cast<const PyAnyTorchTensorValue &>(&rsqrt_), "self"_a);

// aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("rsub", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&rsub), "self"_a, "other"_a, "alpha"_a);

// aten::scatter.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&scatter), "self"_a, "dim"_a, "index"_a, "src"_a);

// aten::scatter.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
m.def("scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&scatter), "self"_a, "dim"_a, "index"_a, "value"_a);

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_add", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&scatter_add), "self"_a, "dim"_a, "index"_a, "src"_a);

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_add_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&scatter_add_), "self"_a, "dim"_a, "index"_a, "src"_a);

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
m.def("scatter_reduce", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_StringValue &, const PyTorch_BoolValue &>(&scatter_reduce), "self"_a, "dim"_a, "index"_a, "src"_a, "reduce"_a, "include_self"_a);

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
m.def("scatter_reduce_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_StringValue &, const PyTorch_BoolValue &>(&scatter_reduce_), "self"_a, "dim"_a, "index"_a, "src"_a, "reduce"_a, "include_self"_a);

// aten::select.int : (Tensor, int, int) -> (Tensor)
m.def("select", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&select), "self"_a, "dim"_a, "index"_a);

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
m.def("select_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&select_copy), "self"_a, "dim"_a, "index"_a);

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
m.def("select_scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&select_scatter), "self"_a, "src"_a, "dim"_a, "index"_a);

// aten::sigmoid : (Tensor) -> (Tensor)
m.def("sigmoid", py::overload_cast<const PyAnyTorchTensorValue &>(&sigmoid), "self"_a);

// aten::sigmoid_ : (Tensor) -> (Tensor)
m.def("sigmoid_", py::overload_cast<const PyAnyTorchTensorValue &>(&sigmoid_), "self"_a);

// aten::silu : (Tensor) -> (Tensor)
m.def("silu", py::overload_cast<const PyAnyTorchTensorValue &>(&silu), "self"_a);

// aten::silu_ : (Tensor) -> (Tensor)
m.def("silu_", py::overload_cast<const PyAnyTorchTensorValue &>(&silu_), "self"_a);

// aten::sin : (Tensor) -> (Tensor)
m.def("sin", py::overload_cast<const PyAnyTorchTensorValue &>(&sin), "self"_a);

// aten::sin_ : (Tensor) -> (Tensor)
m.def("sin_", py::overload_cast<const PyAnyTorchTensorValue &>(&sin_), "self"_a);

// aten::size : (Tensor) -> (int[])
m.def("size", py::overload_cast<const PyAnyTorchTensorValue &>(&size), "self"_a);

// aten::size.int : (Tensor, int) -> (int)
m.def("size", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&size), "self"_a, "dim"_a);

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyDefaultingTorchOptionalIntValue &start, const PyDefaultingTorchOptionalIntValue &end, const PyTorch_IntValue &step) { return slice(self, dim, start.get(), end.get(), step); }, "self"_a, "dim"_a, "start"_a = py::none(), "end"_a = py::none(), "step"_a);

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyDefaultingTorchOptionalIntValue &start, const PyDefaultingTorchOptionalIntValue &end, const PyTorch_IntValue &step) { return slice_copy(self, dim, start.get(), end.get(), step); }, "self"_a, "dim"_a, "start"_a = py::none(), "end"_a = py::none(), "step"_a);

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyDefaultingTorchOptionalIntValue &start, const PyDefaultingTorchOptionalIntValue &end, const PyTorch_IntValue &step) { return slice_scatter(self, src, dim, start.get(), end.get(), step); }, "self"_a, "src"_a, "dim"_a, "start"_a = py::none(), "end"_a = py::none(), "step"_a);

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
m.def("softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyDefaultingTorchOptionalIntValue &dtype) { return softmax(self, dim, dtype.get()); }, "self"_a, "dim"_a, "dtype"_a = py::none());

// aten::softplus : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("softplus", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&softplus), "self"_a, "beta"_a, "threshold"_a);

// aten::sort.int : (int[], bool) -> ()
m.def("sort", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&sort), "self"_a, "reverse"_a);

// aten::sqrt : (Tensor) -> (Tensor)
m.def("sqrt", py::overload_cast<const PyAnyTorchTensorValue &>(&sqrt), "self"_a);

// aten::sqrt.int : (int) -> (float)
m.def("sqrt", py::overload_cast<const PyTorch_IntValue &>(&sqrt), "a"_a);

// aten::sqrt_ : (Tensor) -> (Tensor)
m.def("sqrt_", py::overload_cast<const PyAnyTorchTensorValue &>(&sqrt_), "self"_a);

// aten::square : (Tensor) -> (Tensor)
m.def("square", py::overload_cast<const PyAnyTorchTensorValue &>(&square), "self"_a);

// aten::square_ : (Tensor) -> (Tensor)
m.def("square_", py::overload_cast<const PyAnyTorchTensorValue &>(&square_), "self"_a);

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
m.def("squeeze", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&squeeze), "self"_a, "dim"_a);

// aten::squeeze : (Tensor) -> (Tensor)
m.def("squeeze", py::overload_cast<const PyAnyTorchTensorValue &>(&squeeze), "self"_a);

// prims::squeeze : (Tensor, int[]) -> (Tensor)
m.def("squeeze", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&squeeze), "a"_a, "dimensions"_a);

// aten::squeeze_copy : (Tensor) -> (Tensor)
m.def("squeeze_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&squeeze_copy), "self"_a);

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
m.def("squeeze_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&squeeze_copy), "self"_a, "dim"_a);

// aten::stack : (Tensor[], int) -> (Tensor)
m.def("stack", py::overload_cast<const PyAnyTorchListOfTensorValue &, const PyTorch_IntValue &>(&stack), "tensors"_a, "dim"_a);

// aten::std : (Tensor, bool) -> (Tensor)
m.def("std", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&std), "self"_a, "unbiased"_a);

// aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("sub", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&sub), "self"_a, "other"_a, "alpha"_a);

// aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("sub", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&sub), "self"_a, "other"_a, "alpha"_a);

// aten::sub.int : (int, int) -> (int)
m.def("sub", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&sub), "a"_a, "b"_a);

// aten::sub.float : (float, float) -> (float)
m.def("sub", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&sub), "a"_a, "b"_a);

// aten::sub_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("sub_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&sub_), "self"_a, "other"_a, "alpha"_a);

// aten::sub_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("sub_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&sub_), "self"_a, "other"_a, "alpha"_a);

// aten::sum : (Tensor, int?) -> (Tensor)
m.def("sum", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype) { return sum(self, dtype.get()); }, "self"_a, "dtype"_a = py::none());

// aten::t : (Tensor) -> (Tensor)
m.def("t", py::overload_cast<const PyAnyTorchTensorValue &>(&t), "self"_a);

// aten::t_copy : (Tensor) -> (Tensor)
m.def("t_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&t_copy), "self"_a);

// aten::tanh : (Tensor) -> (Tensor)
m.def("tanh", py::overload_cast<const PyAnyTorchTensorValue &>(&tanh), "self"_a);

// aten::tanh_ : (Tensor) -> (Tensor)
m.def("tanh_", py::overload_cast<const PyAnyTorchTensorValue &>(&tanh_), "self"_a);

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
m.def("tanh_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&tanh_backward), "grad_output"_a, "output"_a);

// aten::tensor : (t[], int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyAnyTorchListValue &data, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) { return tensor(data, dtype.get(), device.get(), requires_grad); }, "data"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a);

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_BoolValue &t, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) { return tensor(t, dtype.get(), device.get(), requires_grad); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a);

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_IntValue &t, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) { return tensor(t, dtype.get(), device.get(), requires_grad); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a);

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_FloatValue &t, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad) { return tensor(t, dtype.get(), device.get(), requires_grad); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a);

// aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("threshold", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&threshold), "self"_a, "threshold"_a, "value"_a);

// aten::threshold_ : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("threshold_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&threshold_), "self"_a, "threshold"_a, "value"_a);

// aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("threshold_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&threshold_backward), "grad_output"_a, "self"_a, "threshold"_a);

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyDefaultingTorchOptionalIntValue &memory_format) { return to(self, dtype, non_blocking, copy, memory_format.get()); }, "self"_a, "dtype"_a, "non_blocking"_a, "copy"_a, "memory_format"_a = py::none());

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyDefaultingTorchOptionalIntValue &memory_format) { return to(self, dtype.get(), layout.get(), device.get(), pin_memory.get(), non_blocking, copy, memory_format.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "non_blocking"_a, "copy"_a, "memory_format"_a = py::none());

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyDefaultingTorchOptionalIntValue &memory_format) { return to(self, other, non_blocking, copy, memory_format.get()); }, "self"_a, "other"_a, "non_blocking"_a, "copy"_a, "memory_format"_a = py::none());

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy) { return to(self, device.get(), dtype.get(), non_blocking, copy); }, "self"_a, "device"_a = py::none(), "dtype"_a = py::none(), "non_blocking"_a, "copy"_a);

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyDefaultingTorchOptionalIntValue &memory_format) { return to(self, device, dtype, non_blocking, copy, memory_format.get()); }, "self"_a, "device"_a, "dtype"_a, "non_blocking"_a, "copy"_a, "memory_format"_a = py::none());

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
m.def("transpose", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&transpose), "self"_a, "dim0"_a, "dim1"_a);

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
m.def("transpose_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&transpose_copy), "self"_a, "dim0"_a, "dim1"_a);

// aten::triu : (Tensor, int) -> (Tensor)
m.def("triu", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&triu), "self"_a, "diagonal"_a);

// aten::triu_ : (Tensor, int) -> (Tensor)
m.def("triu_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&triu_), "self"_a, "diagonal"_a);

// aten::type_as : (Tensor, Tensor) -> (Tensor)
m.def("type_as", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&type_as), "self"_a, "other"_a);

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
m.def("unfold_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&unfold_copy), "self"_a, "dimension"_a, "size"_a, "step"_a);

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
m.def("uniform", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyDefaultingTorchOptionalGeneratorValue &generator) { return uniform(self, from, to, generator.get()); }, "self"_a, "from"_a, "to"_a, "generator"_a = py::none());

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
m.def("uniform_", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyDefaultingTorchOptionalGeneratorValue &generator) { return uniform_(self, from, to, generator.get()); }, "self"_a, "from"_a, "to"_a, "generator"_a = py::none());

// aten::unsqueeze : (Tensor, int) -> (Tensor)
m.def("unsqueeze", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&unsqueeze), "self"_a, "dim"_a);

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
m.def("unsqueeze_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&unsqueeze_), "self"_a, "dim"_a);

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
m.def("unsqueeze_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&unsqueeze_copy), "self"_a, "dim"_a);

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
m.def("upsample_nearest2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyDefaultingTorchOptionalFloatValue &scales_h, const PyDefaultingTorchOptionalFloatValue &scales_w) { return upsample_nearest2d(self, output_size, scales_h.get(), scales_w.get()); }, "self"_a, "output_size"_a, "scales_h"_a = py::none(), "scales_w"_a = py::none());

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
m.def("upsample_nearest2d_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyDefaultingTorchOptionalFloatValue &scales_h, const PyDefaultingTorchOptionalFloatValue &scales_w) { return upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h.get(), scales_w.get()); }, "grad_output"_a, "output_size"_a, "input_size"_a, "scales_h"_a = py::none(), "scales_w"_a = py::none());

// aten::var : (Tensor, bool) -> (Tensor)
m.def("var", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&var), "self"_a, "unbiased"_a);

// aten::view : (Tensor, int[]) -> (Tensor)
m.def("view", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&view), "self"_a, "size"_a);

// aten::view_as_complex : (Tensor) -> (Tensor)
m.def("view_as_complex", py::overload_cast<const PyAnyTorchTensorValue &>(&view_as_complex), "self"_a);

// aten::view_copy : (Tensor, int[]) -> (Tensor)
m.def("view_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&view_copy), "self"_a, "size"_a);

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
m.def("view_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&view_copy), "self"_a, "dtype"_a);

// prims::view_of : (Tensor) -> (Tensor)
m.def("view_of", py::overload_cast<const PyAnyTorchTensorValue &>(&view_of), "a"_a);

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("where", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&where), "condition"_a, "self"_a, "other"_a);

// aten::where.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("where", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchScalarValue &>(&where), "condition"_a, "self"_a, "other"_a);

// aten::where.ScalarOther : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("where", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &>(&where), "condition"_a, "self"_a, "other"_a);

// aten::where.ScalarSelf : (Tensor, Scalar, Tensor) -> (Tensor)
m.def("where", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchScalarValue &, const PyAnyTorchTensorValue &>(&where), "condition"_a, "self"_a, "other"_a);

// aten::zero : (Tensor) -> (Tensor)
m.def("zero", py::overload_cast<const PyAnyTorchTensorValue &>(&zero), "self"_a);

// aten::zero_ : (Tensor) -> (Tensor)
m.def("zero_", py::overload_cast<const PyAnyTorchTensorValue &>(&zero_), "self"_a);

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("zeros", [](const PyAnyTorchListOfTorchIntValue &size, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory) { return zeros(size, dtype.get(), layout.get(), device.get(), pin_memory.get()); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none());

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("zeros_like", [](const PyAnyTorchTensorValue &self, const PyDefaultingTorchOptionalIntValue &dtype, const PyDefaultingTorchOptionalIntValue &layout, const PyDefaultingTorchOptionalDeviceValue &device, const PyDefaultingTorchOptionalBoolValue &pin_memory, const PyDefaultingTorchOptionalIntValue &memory_format) { return zeros_like(self, dtype.get(), layout.get(), device.get(), pin_memory.get(), memory_format.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none());
