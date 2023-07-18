
// aten::abs : (Tensor) -> (Tensor)
m.def("abs", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return abs(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::abs_ : (Tensor) -> (Tensor)
m.def("abs_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return abs_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
m.def("adaptive_avg_pool2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return adaptive_avg_pool2d(self, output_size, loc.get(), ip.get()); }, "self"_a, "output_size"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::add.float_int : (float, int) -> (float)
m.def("add", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return add(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::add.int : (int, int) -> (int)
m.def("add", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return add(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("add", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return add(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::add.str : (str, str) -> (str)
m.def("add", [](const PyTorch_StringValue &a, const PyTorch_StringValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_StringValue { return add(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::add.t : (t[], t[]) -> (t[])
m.def("add", [](const PyAnyTorchListValue &a, const PyAnyTorchListValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListValue { return add(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("add", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return add(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::add_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("add_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return add_(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::add_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("add_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return add_(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcdiv", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return addcdiv(self, tensor1, tensor2, value, loc.get(), ip.get()); }, "self"_a, "tensor1"_a, "tensor2"_a, "value"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::addcdiv_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcdiv_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return addcdiv_(self, tensor1, tensor2, value, loc.get(), ip.get()); }, "self"_a, "tensor1"_a, "tensor2"_a, "value"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcmul", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return addcmul(self, tensor1, tensor2, value, loc.get(), ip.get()); }, "self"_a, "tensor1"_a, "tensor2"_a, "value"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::addcmul_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
m.def("addcmul_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return addcmul_(self, tensor1, tensor2, value, loc.get(), ip.get()); }, "self"_a, "tensor1"_a, "tensor2"_a, "value"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("addmm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat1, const PyAnyTorchTensorValue &mat2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return addmm(self, mat1, mat2, beta, alpha, loc.get(), ip.get()); }, "self"_a, "mat1"_a, "mat2"_a, "beta"_a = 1, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::alias_copy : (Tensor) -> (Tensor)
m.def("alias_copy", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return alias_copy(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::all.bool : (bool[]) -> (bool)
m.def("all", [](const PyAnyTorchListOfTorchBoolValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return all(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::all : (Tensor) -> (Tensor)
m.def("all", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return all(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::amax : (Tensor, int[], bool) -> (Tensor)
m.def("amax", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return amax(self, dim, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a = std::vector<int>{}, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::any.bool : (bool[]) -> (bool)
m.def("any", [](const PyAnyTorchListOfTorchBoolValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return any(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
m.def("any", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return any(self, dim, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::any : (Tensor) -> (Tensor)
m.def("any", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return any(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return arange(end, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "end"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return arange(start, end, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "start"_a, "end"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::arange.start_out : (Scalar, Scalar, Scalar, Tensor) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchTensorValue &out, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return arange(start, end, step, out, loc.get(), ip.get()); }, "start"_a, "end"_a, "step"_a = 1, "out"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("arange", [](const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return arange(start, end, step, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "start"_a, "end"_a, "step"_a = 1, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
m.def("argmax", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return argmax(self, dim, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
m.def("as_strided_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return as_strided_copy(self, size, stride, storage_offset, loc.get(), ip.get()); }, "self"_a, "size"_a, "stride"_a, "storage_offset"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
m.def("as_strided_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return as_strided_scatter(self, src, size, stride, storage_offset, loc.get(), ip.get()); }, "self"_a, "src"_a, "size"_a, "stride"_a, "storage_offset"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
m.def("atan2", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return atan2(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
m.def("atan2_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return atan2_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::atan : (Tensor) -> (Tensor)
m.def("atan", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return atan(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::atan_ : (Tensor) -> (Tensor)
m.def("atan_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return atan_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
m.def("avg_pool2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, loc.get(), ip.get()); }, "self"_a, "kernel_size"_a, "stride"_a = std::vector<int>{}, "padding"_a = std::vector<int>{0, 0}, "ceil_mode"_a = false, "count_include_pad"_a = true, "divisor_override"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::baddbmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("baddbmm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return baddbmm(self, batch1, batch2, beta, alpha, loc.get(), ip.get()); }, "self"_a, "batch1"_a, "batch2"_a, "beta"_a = 1, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::baddbmm_ : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("baddbmm_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return baddbmm_(self, batch1, batch2, beta, alpha, loc.get(), ip.get()); }, "self"_a, "batch1"_a, "batch2"_a, "beta"_a = 1, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)
m.def("batch_norm", [](const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enabled, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled, loc.get(), ip.get()); }, "input"_a, "weight"_a = py::none(), "bias"_a = py::none(), "running_mean"_a = py::none(), "running_var"_a = py::none(), "training"_a, "momentum"_a, "eps"_a, "cudnn_enabled"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bernoulli(self, generator, loc.get(), ip.get()); }, "self"_a, "generator"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bernoulli(self, p, generator, loc.get(), ip.get()); }, "self"_a, "p"_a, "generator"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
m.def("bernoulli", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bernoulli(self, p, generator, loc.get(), ip.get()); }, "self"_a, "p"_a, "generator"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
m.def("bernoulli_", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bernoulli_(self, p, generator, loc.get(), ip.get()); }, "self"_a, "p"_a = 0.5, "generator"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
m.def("bernoulli_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bernoulli_(self, p, generator, loc.get(), ip.get()); }, "self"_a, "p"_a, "generator"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bincount : (Tensor, Tensor?, int) -> (Tensor)
m.def("bincount", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &weights, const PyTorch_IntValue &minlength, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bincount(self, weights, minlength, loc.get(), ip.get()); }, "self"_a, "weights"_a = py::none(), "minlength"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_and", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bitwise_and(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_and_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bitwise_and_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bitwise_not : (Tensor) -> (Tensor)
m.def("bitwise_not", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bitwise_not(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bitwise_not_ : (Tensor) -> (Tensor)
m.def("bitwise_not_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bitwise_not_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_or", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bitwise_or(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_or_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bitwise_or_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_xor", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bitwise_xor(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_xor_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bitwise_xor_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bmm : (Tensor, Tensor) -> (Tensor)
m.def("bmm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bmm(self, mat2, loc.get(), ip.get()); }, "self"_a, "mat2"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Bool.float : (float) -> (bool)
m.def("Bool", [](const PyTorch_FloatValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return Bool(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Bool.int : (int) -> (bool)
m.def("Bool", [](const PyTorch_IntValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return Bool(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Bool.Tensor : (Tensor) -> (bool)
m.def("Bool", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return Bool(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
m.def("broadcast_to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return broadcast_to(self, size, loc.get(), ip.get()); }, "self"_a, "size"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
m.def("bucketize", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return bucketize(self, boundaries, out_int32, right, loc.get(), ip.get()); }, "self"_a, "boundaries"_a, "out_int32"_a = false, "right"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::cat : (Tensor[], int) -> (Tensor)
m.def("cat", [](const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return cat(tensors, dim, loc.get(), ip.get()); }, "tensors"_a, "dim"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ceil.float : (float) -> (int)
m.def("ceil", [](const PyTorch_FloatValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return ceil(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ceil : (Tensor) -> (Tensor)
m.def("ceil", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ceil(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ceil_ : (Tensor) -> (Tensor)
m.def("ceil_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ceil_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clamp_max : (Tensor, Scalar) -> (Tensor)
m.def("clamp_max", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clamp_max(self, max, loc.get(), ip.get()); }, "self"_a, "max"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clamp_max_ : (Tensor, Scalar) -> (Tensor)
m.def("clamp_max_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clamp_max_(self, max, loc.get(), ip.get()); }, "self"_a, "max"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clamp_min : (Tensor, Scalar) -> (Tensor)
m.def("clamp_min", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clamp_min(self, min, loc.get(), ip.get()); }, "self"_a, "min"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clamp_min_ : (Tensor, Scalar) -> (Tensor)
m.def("clamp_min_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clamp_min_(self, min, loc.get(), ip.get()); }, "self"_a, "min"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clamp : (Tensor, Scalar?, Scalar?) -> (Tensor)
m.def("clamp", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clamp(self, min, max, loc.get(), ip.get()); }, "self"_a, "min"_a = py::none(), "max"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clamp.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
m.def("clamp", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clamp(self, min, max, loc.get(), ip.get()); }, "self"_a, "min"_a = py::none(), "max"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clamp_ : (Tensor, Scalar?, Scalar?) -> (Tensor)
m.def("clamp_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clamp_(self, min, max, loc.get(), ip.get()); }, "self"_a, "min"_a = py::none(), "max"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clamp_.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
m.def("clamp_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clamp_(self, min, max, loc.get(), ip.get()); }, "self"_a, "min"_a = py::none(), "max"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::clone : (Tensor, int?) -> (Tensor)
m.def("clone", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return clone(self, memory_format, loc.get(), ip.get()); }, "self"_a, "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)
m.def("constant_pad_nd", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return constant_pad_nd(self, pad__, value, loc.get(), ip.get()); }, "self"_a, "pad__"_a, "value"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::contiguous : (Tensor, int) -> (Tensor)
m.def("contiguous", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return contiguous(self, memory_format, loc.get(), ip.get()); }, "self"_a, "memory_format"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)
m.def("conv2d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_IntValue &groups, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return conv2d(input, weight, bias, stride, padding, dilation, groups, loc.get(), ip.get()); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a = std::vector<int>{1, 1}, "padding"_a = std::vector<int>{0, 0}, "dilation"_a = std::vector<int>{1, 1}, "groups"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::conv_transpose1d : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose1d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation, loc.get(), ip.get()); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a = std::vector<int>{1}, "padding"_a = std::vector<int>{0}, "output_padding"_a = std::vector<int>{0}, "groups"_a = 1, "dilation"_a = std::vector<int>{1}, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::conv_transpose2d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose2d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation, loc.get(), ip.get()); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a = std::vector<int>{1, 1}, "padding"_a = std::vector<int>{0, 0}, "output_padding"_a = std::vector<int>{0, 0}, "groups"_a = 1, "dilation"_a = std::vector<int>{1, 1}, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::conv_transpose3d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
m.def("conv_transpose3d", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation, loc.get(), ip.get()); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a = std::vector<int>{1, 1, 1}, "padding"_a = std::vector<int>{0, 0, 0}, "output_padding"_a = std::vector<int>{0, 0, 0}, "groups"_a = 1, "dilation"_a = std::vector<int>{1, 1, 1}, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::convolution_backward : (Tensor, Tensor, Tensor, int[]?, int[], int[], int[], bool, int[], int, bool[]) -> (Tensor, Tensor, Tensor)
m.def("convolution_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalListOfTorchIntValue &bias_sizes, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchBoolValue &output_mask, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask, loc.get(), ip.get()); }, "grad_output"_a, "input"_a, "weight"_a, "bias_sizes"_a = py::none(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a, "output_mask"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)
m.def("convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, loc.get(), ip.get()); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
m.def("copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return copy(self, src, non_blocking, loc.get(), ip.get()); }, "self"_a, "src"_a, "non_blocking"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
m.def("copy_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return copy_(self, src, non_blocking, loc.get(), ip.get()); }, "self"_a, "src"_a, "non_blocking"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::cos : (Tensor) -> (Tensor)
m.def("cos", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return cos(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::cos_ : (Tensor) -> (Tensor)
m.def("cos_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return cos_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::cpu : (Tensor) -> (Tensor)
m.def("cpu", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return cpu(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::cross_entropy_loss : (Tensor, Tensor, Tensor?, int, int, float) -> (Tensor)
m.def("cross_entropy_loss", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyTorch_FloatValue &label_smoothing, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return cross_entropy_loss(self, target, weight, reduction, ignore_index, label_smoothing, loc.get(), ip.get()); }, "self"_a, "target"_a, "weight"_a = py::none(), "reduction"_a = 1, "ignore_index"_a = -100, "label_smoothing"_a = 0., py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::cuda : (Tensor) -> (Tensor)
m.def("cuda", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return cuda(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
m.def("cumsum", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return cumsum(self, dim, dtype, loc.get(), ip.get()); }, "self"_a, "dim"_a, "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
m.def("Delete", [](const PyTorch_DictValue &self, const PyTorch_StringValue &key, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> void { return Delete(self, key, loc.get(), ip.get()); }, "self"_a, "key"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::detach_copy : (Tensor) -> (Tensor)
m.def("detach_copy", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return detach_copy(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::detach : (Tensor) -> (Tensor)
m.def("detach", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return detach(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
m.def("diagonal_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return diagonal_copy(self, offset, dim1, dim2, loc.get(), ip.get()); }, "self"_a, "offset"_a = 0, "dim1"_a = 0, "dim2"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
m.def("diagonal_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return diagonal_scatter(self, src, offset, dim1, dim2, loc.get(), ip.get()); }, "self"_a, "src"_a, "offset"_a = 0, "dim1"_a = 0, "dim2"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::dim : (Tensor) -> (int)
m.def("dim", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return dim(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div.float : (float, float) -> (float)
m.def("div", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return div(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div.int : (int, int) -> (float)
m.def("div", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return div(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div : (Scalar, Scalar) -> (float)
m.def("div", [](const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return div(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("div", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return div(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
m.def("div", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return div(self, other, rounding_mode, loc.get(), ip.get()); }, "self"_a, "other"_a, "rounding_mode"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("div", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return div(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("div_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return div_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
m.def("div_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return div_(self, other, rounding_mode, loc.get(), ip.get()); }, "self"_a, "other"_a, "rounding_mode"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("div_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return div_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::dropout : (Tensor, float, bool) -> (Tensor)
m.def("dropout", [](const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return dropout(input, p, train, loc.get(), ip.get()); }, "input"_a, "p"_a, "train"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
m.def("dropout_", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return dropout_(self, p, train, loc.get(), ip.get()); }, "self"_a, "p"_a, "train"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::embedding_bag.padding_idx : (Tensor, Tensor, Tensor, bool, int, bool, Tensor?, bool, int?) -> (Tensor, Tensor, Tensor, Tensor)
m.def("embedding_bag", [](const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyAnyTorchTensorValue &offsets, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_IntValue &mode, const PyTorch_BoolValue &sparse, const PyAnyTorchOptionalTensorValue &per_sample_weights, const PyTorch_BoolValue &include_last_offset, const PyAnyTorchOptionalIntValue &padding_idx, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, loc.get(), ip.get()); }, "weight"_a, "indices"_a, "offsets"_a, "scale_grad_by_freq"_a, "mode"_a, "sparse"_a, "per_sample_weights"_a = py::none(), "include_last_offset"_a, "padding_idx"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
m.def("embedding_dense_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, loc.get(), ip.get()); }, "grad_output"_a, "indices"_a, "num_weights"_a, "padding_idx"_a, "scale_grad_by_freq"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
m.def("embedding", [](const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse, loc.get(), ip.get()); }, "weight"_a, "indices"_a, "padding_idx"_a = -1, "scale_grad_by_freq"_a = false, "sparse"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("empty_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return empty_like(self, dtype, layout, device, pin_memory, memory_format, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("empty", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return empty(size, dtype, layout, device, pin_memory, memory_format, loc.get(), ip.get()); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq.device : (Device, Device) -> (bool)
m.def("eq", [](const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return eq(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq.float : (float, float) -> (bool)
m.def("eq", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return eq(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq.int_list : (int[], int[]) -> (bool)
m.def("eq", [](const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return eq(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq.int : (int, int) -> (bool)
m.def("eq", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return eq(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("eq", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return eq(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq.str : (str, str) -> (bool)
m.def("eq", [](const PyTorch_StringValue &a, const PyTorch_StringValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return eq(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("eq", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return eq(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("eq_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return eq_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("eq_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return eq_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::erf : (Tensor) -> (Tensor)
m.def("erf", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return erf(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::erf_ : (Tensor) -> (Tensor)
m.def("erf_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return erf_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::exp : (Tensor) -> (Tensor)
m.def("exp", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return exp(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::exp_ : (Tensor) -> (Tensor)
m.def("exp_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return exp_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
m.def("expand_as", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return expand_as(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
m.def("expand_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return expand_copy(self, size, implicit, loc.get(), ip.get()); }, "self"_a, "size"_a, "implicit"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::expand : (Tensor, int[], bool) -> (Tensor)
m.def("expand", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return expand(self, size, implicit, loc.get(), ip.get()); }, "self"_a, "size"_a, "implicit"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::expm1 : (Tensor) -> (Tensor)
m.def("expm1", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return expm1(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::expm1_ : (Tensor) -> (Tensor)
m.def("expm1_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return expm1_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
m.def("fft_fft", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return fft_fft(self, n, dim, norm, loc.get(), ip.get()); }, "self"_a, "n"_a = py::none(), "dim"_a = -1, "norm"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::fill.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fill", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return fill(self, value, loc.get(), ip.get()); }, "self"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("fill", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return fill(self, value, loc.get(), ip.get()); }, "self"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fill_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return fill_(self, value, loc.get(), ip.get()); }, "self"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("fill_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return fill_(self, value, loc.get(), ip.get()); }, "self"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
m.def("flatten", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return flatten(self, start_dim, end_dim, loc.get(), ip.get()); }, "self"_a, "start_dim"_a = 0, "end_dim"_a = -1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::flip : (Tensor, int[]) -> (Tensor)
m.def("flip", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return flip(self, dims, loc.get(), ip.get()); }, "self"_a, "dims"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::FloatImplicit : (Tensor) -> (float)
m.def("FloatImplicit", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return FloatImplicit(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Float.Scalar : (Scalar) -> (float)
m.def("Float", [](const PyAnyTorchScalarValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return Float(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Float.str : (str) -> (float)
m.def("Float", [](const PyTorch_StringValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return Float(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Float.Tensor : (Tensor) -> (float)
m.def("Float", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return Float(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
m.def("floor_divide", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return floor_divide(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("floor_divide", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return floor_divide(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::floor : (Tensor) -> (Tensor)
m.def("floor", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return floor(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::floor_ : (Tensor) -> (Tensor)
m.def("floor_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return floor_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::floordiv.int : (int, int) -> (int)
m.def("floordiv", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return floordiv(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fmod", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return fmod(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::fmod_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("fmod_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return fmod_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
m.def("frobenius_norm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return frobenius_norm(self, dim, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::full_like : (Tensor, Scalar, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("full_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return full_like(self, fill_value, dtype, layout, device, pin_memory, memory_format, loc.get(), ip.get()); }, "self"_a, "fill_value"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::full : (int[], Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("full", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return full(size, fill_value, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "size"_a, "fill_value"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
m.def("gather", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return gather(self, dim, index, sparse_grad, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "sparse_grad"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ge.float_int : (float, int) -> (bool)
m.def("ge", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return ge(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ge.float : (float, float) -> (bool)
m.def("ge", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return ge(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ge.int : (int, int) -> (bool)
m.def("ge", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return ge(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ge", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ge(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ge", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ge(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ge_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ge_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ge_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ge_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ge_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
m.def("gelu_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return gelu_backward(grad_output, self, approximate, loc.get(), ip.get()); }, "grad_output"_a, "self"_a, "approximate"_a = "{s}", py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gelu : (Tensor, str) -> (Tensor)
m.def("gelu", [](const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return gelu(self, approximate, loc.get(), ip.get()); }, "self"_a, "approximate"_a = "{s}", py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gt.float_int : (float, int) -> (bool)
m.def("gt", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return gt(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gt.float : (float, float) -> (bool)
m.def("gt", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return gt(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gt.int : (int, int) -> (bool)
m.def("gt", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return gt(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("gt", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return gt(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("gt", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return gt(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gt_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("gt_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return gt_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("gt_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return gt_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::hardsigmoid : (Tensor) -> (Tensor)
m.def("hardsigmoid", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return hardsigmoid(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
m.def("hardsigmoid_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return hardsigmoid_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::hardswish : (Tensor) -> (Tensor)
m.def("hardswish", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return hardswish(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::hardswish_ : (Tensor) -> (Tensor)
m.def("hardswish_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return hardswish_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::hardtanh_backward : (Tensor, Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return hardtanh_backward(grad_output, self, min_val, max_val, loc.get(), ip.get()); }, "grad_output"_a, "self"_a, "min_val"_a, "max_val"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::hardtanh : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return hardtanh(self, min_val, max_val, loc.get(), ip.get()); }, "self"_a, "min_val"_a = -1, "max_val"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::hardtanh_ : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("hardtanh_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return hardtanh_(self, min_val, max_val, loc.get(), ip.get()); }, "self"_a, "min_val"_a = -1, "max_val"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::imag : (Tensor) -> (Tensor)
m.def("imag", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return imag(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::index_put.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
m.def("index_put", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return index_put(self, indices, values, accumulate, loc.get(), ip.get()); }, "self"_a, "indices"_a, "values"_a, "accumulate"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::index_put : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)
m.def("index_put", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return index_put(self, indices, values, accumulate, loc.get(), ip.get()); }, "self"_a, "indices"_a, "values"_a, "accumulate"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::index_put_.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
m.def("index_put_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return index_put_(self, indices, values, accumulate, loc.get(), ip.get()); }, "self"_a, "indices"_a, "values"_a, "accumulate"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::index_put_ : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)
m.def("index_put_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return index_put_(self, indices, values, accumulate, loc.get(), ip.get()); }, "self"_a, "indices"_a, "values"_a, "accumulate"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
m.def("index_select", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return index_select(self, dim, index, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::index.Tensor_hacked_twin : (Tensor, Tensor[]) -> (Tensor)
m.def("index", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return index(self, indices, loc.get(), ip.get()); }, "self"_a, "indices"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::index.Tensor : (Tensor, Tensor?[]) -> (Tensor)
m.def("index", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return index(self, indices, loc.get(), ip.get()); }, "self"_a, "indices"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Int.bool : (bool) -> (int)
m.def("Int", [](const PyTorch_BoolValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return Int(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Int.float : (float) -> (int)
m.def("Int", [](const PyTorch_FloatValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return Int(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::IntImplicit : (Tensor) -> (int)
m.def("IntImplicit", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return IntImplicit(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Int.Scalar : (Scalar) -> (int)
m.def("Int", [](const PyAnyTorchScalarValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return Int(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::Int.Tensor : (Tensor) -> (int)
m.def("Int", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return Int(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::is_floating_point : (Tensor) -> (bool)
m.def("is_floating_point", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return is_floating_point(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::isnan : (Tensor) -> (Tensor)
m.def("isnan", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return isnan(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::join : (str, str[]) -> (str)
m.def("join", [](const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_StringValue { return join(self, values, loc.get(), ip.get()); }, "self"_a, "values"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::keys.str : (Dict(str, t)) -> (str[])
m.def("keys", [](const PyTorch_DictValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListOfTorchStringValue { return keys(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)
m.def("layer_norm", [](const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enable, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable, loc.get(), ip.get()); }, "input"_a, "normalized_shape"_a, "weight"_a = py::none(), "bias"_a = py::none(), "eps"_a = 1.0000000000000001e-05, "cudnn_enable"_a = true, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::le.int : (int, int) -> (bool)
m.def("le", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return le(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("le", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return le(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("le", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return le(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::le_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("le_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return le_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("le_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return le_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::leaky_relu_backward : (Tensor, Tensor, Scalar, bool) -> (Tensor)
m.def("leaky_relu_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, const PyTorch_BoolValue &self_is_result, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return leaky_relu_backward(grad_output, self, negative_slope, self_is_result, loc.get(), ip.get()); }, "grad_output"_a, "self"_a, "negative_slope"_a, "self_is_result"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::leaky_relu : (Tensor, Scalar) -> (Tensor)
m.def("leaky_relu", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return leaky_relu(self, negative_slope, loc.get(), ip.get()); }, "self"_a, "negative_slope"_a = 0.01, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::leaky_relu_ : (Tensor, Scalar) -> (Tensor)
m.def("leaky_relu_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return leaky_relu_(self, negative_slope, loc.get(), ip.get()); }, "self"_a, "negative_slope"_a = 0.01, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::len.str : (str) -> (int)
m.def("len", [](const PyTorch_StringValue &s, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return len(s, loc.get(), ip.get()); }, "s"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::len.t : (t[]) -> (int)
m.def("len", [](const PyAnyTorchListValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return len(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::len.Tensor : (Tensor) -> (int)
m.def("len", [](const PyAnyTorchTensorValue &t, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return len(t, loc.get(), ip.get()); }, "t"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("lerp", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return lerp(self, end, weight, loc.get(), ip.get()); }, "self"_a, "end"_a, "weight"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("lerp_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return lerp_(self, end, weight, loc.get(), ip.get()); }, "self"_a, "end"_a, "weight"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
m.def("lift_fresh_copy", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return lift_fresh_copy(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::linalg_vector_norm : (Tensor, Scalar, int[]?, bool, int?) -> (Tensor)
m.def("linalg_vector_norm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &ord, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return linalg_vector_norm(self, ord, dim, keepdim, dtype, loc.get(), ip.get()); }, "self"_a, "ord"_a = 2, "dim"_a = py::none(), "keepdim"_a = false, "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)
m.def("linear", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return linear(input, weight, bias, loc.get(), ip.get()); }, "input"_a, "weight"_a, "bias"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::list.t : (t[]) -> (t[])
m.def("list", [](const PyAnyTorchListValue &l, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListValue { return list(l, loc.get(), ip.get()); }, "l"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::log1p : (Tensor) -> (Tensor)
m.def("log1p", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return log1p(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::log1p_ : (Tensor) -> (Tensor)
m.def("log1p_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return log1p_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::log2 : (Tensor) -> (Tensor)
m.def("log2", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return log2(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::log2_ : (Tensor) -> (Tensor)
m.def("log2_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return log2_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::log.int : (int) -> (float)
m.def("log", [](const PyTorch_IntValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return log(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::log : (Tensor) -> (Tensor)
m.def("log", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return log(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
m.def("log_softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return log_softmax(self, dim, dtype, loc.get(), ip.get()); }, "self"_a, "dim"_a, "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::log_ : (Tensor) -> (Tensor)
m.def("log_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return log_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
m.def("logical_and", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logical_and(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_and_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logical_and_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logical_not : (Tensor) -> (Tensor)
m.def("logical_not", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logical_not(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logical_not_ : (Tensor) -> (Tensor)
m.def("logical_not_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logical_not_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
m.def("logical_or", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logical_or(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_or_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logical_or_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
m.def("logical_xor", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logical_xor(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_xor_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logical_xor_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
m.def("logsumexp", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return logsumexp(self, dim, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lt.float_int : (float, int) -> (bool)
m.def("lt", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return lt(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lt.float : (float, float) -> (bool)
m.def("lt", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return lt(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lt.int : (int, int) -> (bool)
m.def("lt", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return lt(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("lt", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return lt(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("lt", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return lt(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lt_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("lt_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return lt_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("lt_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return lt_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("masked_fill", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return masked_fill(self, mask, value, loc.get(), ip.get()); }, "self"_a, "mask"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("masked_fill", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return masked_fill(self, mask, value, loc.get(), ip.get()); }, "self"_a, "mask"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::masked_fill_.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("masked_fill_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return masked_fill_(self, mask, value, loc.get(), ip.get()); }, "self"_a, "mask"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("masked_fill_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return masked_fill_(self, mask, value, loc.get(), ip.get()); }, "self"_a, "mask"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
m.def("masked_select", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return masked_select(self, mask, loc.get(), ip.get()); }, "self"_a, "mask"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::matmul : (Tensor, Tensor) -> (Tensor)
m.def("matmul", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return matmul(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::max.dim : (Tensor, int, bool) -> (Tensor, Tensor)
m.def("max", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return max(self, dim, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::max : (Tensor) -> (Tensor)
m.def("max", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return max(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
m.def("max_pool2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode, loc.get(), ip.get()); }, "self"_a, "kernel_size"_a, "stride"_a = std::vector<int>{}, "padding"_a = std::vector<int>{0, 0}, "dilation"_a = std::vector<int>{1, 1}, "ceil_mode"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
m.def("max_pool2d_with_indices_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, loc.get(), ip.get()); }, "grad_output"_a, "self"_a, "kernel_size"_a, "stride"_a, "padding"_a, "dilation"_a, "ceil_mode"_a, "indices"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::maximum : (Tensor, Tensor) -> (Tensor)
m.def("maximum", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return maximum(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mean.dim : (Tensor, int[]?, bool, int?) -> (Tensor)
m.def("mean", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mean(self, dim, keepdim, dtype, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "keepdim"_a = false, "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mean : (Tensor, int?) -> (Tensor)
m.def("mean", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mean(self, dtype, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::minimum : (Tensor, Tensor) -> (Tensor)
m.def("minimum", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return minimum(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mish : (Tensor) -> (Tensor)
m.def("mish", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mish(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mm : (Tensor, Tensor) -> (Tensor)
m.def("mm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mm(self, mat2, loc.get(), ip.get()); }, "self"_a, "mat2"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::movedim.int : (Tensor, int, int) -> (Tensor)
m.def("movedim", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &source, const PyTorch_IntValue &destination, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return movedim(self, source, destination, loc.get(), ip.get()); }, "self"_a, "source"_a, "destination"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mse_loss_backward : (Tensor, Tensor, Tensor, int) -> (Tensor)
m.def("mse_loss_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mse_loss_backward(grad_output, self, target, reduction, loc.get(), ip.get()); }, "grad_output"_a, "self"_a, "target"_a, "reduction"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
m.def("mse_loss", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mse_loss(self, target, reduction, loc.get(), ip.get()); }, "self"_a, "target"_a, "reduction"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mul.float : (float, float) -> (float)
m.def("mul", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return mul(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mul.int : (int, int) -> (int)
m.def("mul", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return mul(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("mul", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mul(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("mul", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mul(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mul_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("mul_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mul_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("mul_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mul_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::mv : (Tensor, Tensor) -> (Tensor)
m.def("mv", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return mv(self, vec, loc.get(), ip.get()); }, "self"_a, "vec"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
m.def("narrow", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return narrow(self, dim, start, length, loc.get(), ip.get()); }, "self"_a, "dim"_a, "start"_a, "length"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::native_batch_norm_backward : (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, bool[]) -> (Tensor, Tensor, Tensor)
m.def("native_batch_norm_backward", [](const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyAnyTorchOptionalTensorValue &save_mean, const PyAnyTorchOptionalTensorValue &save_invstd, const PyTorch_BoolValue &train, const PyTorch_FloatValue &eps, const PyAnyTorchListOfTorchBoolValue &output_mask, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, loc.get(), ip.get()); }, "grad_out"_a, "input"_a, "weight"_a = py::none(), "running_mean"_a = py::none(), "running_var"_a = py::none(), "save_mean"_a = py::none(), "save_invstd"_a = py::none(), "train"_a, "eps"_a, "output_mask"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::native_batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float) -> (Tensor, Tensor, Tensor)
m.def("native_batch_norm", [](const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, loc.get(), ip.get()); }, "input"_a, "weight"_a = py::none(), "bias"_a = py::none(), "running_mean"_a = py::none(), "running_var"_a = py::none(), "training"_a, "momentum"_a, "eps"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
m.def("native_dropout_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return native_dropout_backward(grad_output, mask, scale, loc.get(), ip.get()); }, "grad_output"_a, "mask"_a, "scale"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::native_dropout : (Tensor, float, bool?) -> (Tensor, Tensor)
m.def("native_dropout", [](const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyAnyTorchOptionalBoolValue &train, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return native_dropout(input, p, train, loc.get(), ip.get()); }, "input"_a, "p"_a, "train"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::native_group_norm_backward : (Tensor, Tensor, Tensor, Tensor, Tensor?, int, int, int, int, bool[]) -> (Tensor, Tensor, Tensor)
m.def("native_group_norm_backward", [](const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &mean, const PyAnyTorchTensorValue &rstd, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &N, const PyTorch_IntValue &C, const PyTorch_IntValue &HxW, const PyTorch_IntValue &group, const PyAnyTorchListOfTorchBoolValue &output_mask, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask, loc.get(), ip.get()); }, "grad_out"_a, "input"_a, "mean"_a, "rstd"_a, "weight"_a = py::none(), "N"_a, "C"_a, "HxW"_a, "group"_a, "output_mask"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::native_group_norm : (Tensor, Tensor?, Tensor?, int, int, int, int, float) -> (Tensor, Tensor, Tensor)
m.def("native_group_norm", [](const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_IntValue &N, const PyTorch_IntValue &C, const PyTorch_IntValue &HxW, const PyTorch_IntValue &group, const PyTorch_FloatValue &eps, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return native_group_norm(input, weight, bias, N, C, HxW, group, eps, loc.get(), ip.get()); }, "input"_a, "weight"_a = py::none(), "bias"_a = py::none(), "N"_a, "C"_a, "HxW"_a, "group"_a, "eps"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::native_layer_norm_backward : (Tensor, Tensor, int[], Tensor, Tensor, Tensor?, Tensor?, bool[]) -> (Tensor, Tensor, Tensor)
m.def("native_layer_norm_backward", [](const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchTensorValue &mean, const PyAnyTorchTensorValue &rstd, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchBoolValue &output_mask, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask, loc.get(), ip.get()); }, "grad_out"_a, "input"_a, "normalized_shape"_a, "mean"_a, "rstd"_a, "weight"_a = py::none(), "bias"_a = py::none(), "output_mask"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::native_layer_norm : (Tensor, int[], Tensor?, Tensor?, float) -> (Tensor, Tensor, Tensor)
m.def("native_layer_norm", [](const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return native_layer_norm(input, normalized_shape, weight, bias, eps, loc.get(), ip.get()); }, "input"_a, "normalized_shape"_a, "weight"_a = py::none(), "bias"_a = py::none(), "eps"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ne.bool : (bool, bool) -> (bool)
m.def("ne", [](const PyTorch_BoolValue &a, const PyTorch_BoolValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return ne(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ne.float_int : (float, int) -> (bool)
m.def("ne", [](const PyTorch_FloatValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return ne(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ne.int_list : (int[], int[]) -> (bool)
m.def("ne", [](const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return ne(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ne.int : (int, int) -> (bool)
m.def("ne", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return ne(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ne", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ne(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ne", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ne(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ne_.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("ne_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ne_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ne_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ne_(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::neg.float : (float) -> (float)
m.def("neg", [](const PyTorch_FloatValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return neg(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::neg.int : (int) -> (int)
m.def("neg", [](const PyTorch_IntValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return neg(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::neg : (Tensor) -> (Tensor)
m.def("neg", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return neg(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::neg_ : (Tensor) -> (Tensor)
m.def("neg_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return neg_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_empty", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return new_empty(self, size, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_empty_strided", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return new_empty_strided(self, size, stride, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "self"_a, "size"_a, "stride"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_ones", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return new_ones(self, size, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_zeros", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return new_zeros(self, size, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "self"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::nll_loss2d_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
m.def("nll_loss2d_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight, loc.get(), ip.get()); }, "grad_output"_a, "self"_a, "target"_a, "weight"_a = py::none(), "reduction"_a, "ignore_index"_a, "total_weight"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::nll_loss2d_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)
m.def("nll_loss2d_forward", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return nll_loss2d_forward(self, target, weight, reduction, ignore_index, loc.get(), ip.get()); }, "self"_a, "target"_a, "weight"_a = py::none(), "reduction"_a, "ignore_index"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::nll_loss_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
m.def("nll_loss_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight, loc.get(), ip.get()); }, "grad_output"_a, "self"_a, "target"_a, "weight"_a = py::none(), "reduction"_a, "ignore_index"_a, "total_weight"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::nll_loss_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)
m.def("nll_loss_forward", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return nll_loss_forward(self, target, weight, reduction, ignore_index, loc.get(), ip.get()); }, "self"_a, "target"_a, "weight"_a = py::none(), "reduction"_a, "ignore_index"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::norm.ScalarOpt_dim : (Tensor, Scalar?, int[], bool) -> (Tensor)
m.def("norm", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &p, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return norm(self, p, dim, keepdim, loc.get(), ip.get()); }, "self"_a, "p"_a = py::none(), "dim"_a, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::numel : (Tensor) -> (int)
m.def("numel", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return numel(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::numpy_T : (Tensor) -> (Tensor)
m.def("numpy_T", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return numpy_T(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::one_hot : (Tensor, int) -> (Tensor)
m.def("one_hot", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &num_classes, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return one_hot(self, num_classes, loc.get(), ip.get()); }, "self"_a, "num_classes"_a = -1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("ones_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ones_like(self, dtype, layout, device, pin_memory, memory_format, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("ones", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return ones(size, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
m.def("pad", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return pad(self, pad__, mode, value, loc.get(), ip.get()); }, "self"_a, "pad__"_a, "mode"_a = "{s}", "value"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
m.def("permute_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return permute_copy(self, dims, loc.get(), ip.get()); }, "self"_a, "dims"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::permute : (Tensor, int[]) -> (Tensor)
m.def("permute", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return permute(self, dims, loc.get(), ip.get()); }, "self"_a, "dims"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::pow.int_float : (int, float) -> (float)
m.def("pow", [](const PyTorch_IntValue &a, const PyTorch_FloatValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return pow(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::pow.Scalar : (Scalar, Tensor) -> (Tensor)
m.def("pow", [](const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &exponent, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return pow(self, exponent, loc.get(), ip.get()); }, "self"_a, "exponent"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)
m.def("pow", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &exponent, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return pow(self, exponent, loc.get(), ip.get()); }, "self"_a, "exponent"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
m.def("pow", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return pow(self, exponent, loc.get(), ip.get()); }, "self"_a, "exponent"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::prelu : (Tensor, Tensor) -> (Tensor)
m.def("prelu", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return prelu(self, weight, loc.get(), ip.get()); }, "self"_a, "weight"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("rand_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return rand_like(self, dtype, layout, device, pin_memory, memory_format, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randint", [](const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return randint(low, high, size, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "low"_a, "high"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::randint : (int, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randint", [](const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return randint(high, size, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "high"_a, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
m.def("randn", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return randn(size, generator, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "size"_a, "generator"_a = py::none(), "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("randn_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return randn_like(self, dtype, layout, device, pin_memory, memory_format, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randn", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return randn(size, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::real : (Tensor) -> (Tensor)
m.def("real", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return real(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::reciprocal : (Tensor) -> (Tensor)
m.def("reciprocal", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return reciprocal(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::reciprocal_ : (Tensor) -> (Tensor)
m.def("reciprocal_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return reciprocal_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::relu6 : (Tensor) -> (Tensor)
m.def("relu6", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return relu6(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::relu6_ : (Tensor) -> (Tensor)
m.def("relu6_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return relu6_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::relu : (Tensor) -> (Tensor)
m.def("relu", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return relu(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::relu_ : (Tensor) -> (Tensor)
m.def("relu_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return relu_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::remainder.int : (int, int) -> (int)
m.def("remainder", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return remainder(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::remainder.Scalar : (Tensor, Scalar) -> (Tensor)
m.def("remainder", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return remainder(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::repeat : (Tensor, int[]) -> (Tensor)
m.def("repeat", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return repeat(self, repeats, loc.get(), ip.get()); }, "self"_a, "repeats"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::reshape : (Tensor, int[]) -> (Tensor)
m.def("reshape", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return reshape(self, shape, loc.get(), ip.get()); }, "self"_a, "shape"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
m.def("resize_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return resize_(self, size, memory_format, loc.get(), ip.get()); }, "self"_a, "size"_a, "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
m.def("roll", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return roll(self, shifts, dims, loc.get(), ip.get()); }, "self"_a, "shifts"_a, "dims"_a = std::vector<int>{}, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::round : (Tensor) -> (Tensor)
m.def("round", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return round(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::round_ : (Tensor) -> (Tensor)
m.def("round_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return round_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::rsqrt : (Tensor) -> (Tensor)
m.def("rsqrt", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return rsqrt(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::rsqrt_ : (Tensor) -> (Tensor)
m.def("rsqrt_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return rsqrt_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("rsub", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return rsub(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scalar_tensor : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
m.def("scalar_tensor", [](const PyAnyTorchScalarValue &s, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scalar_tensor(s, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "s"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scaled_dot_product_attention : (Tensor, Tensor, Tensor, Tensor?, float, bool, float?) -> (Tensor)
m.def("scaled_dot_product_attention", [](const PyAnyTorchTensorValue &query, const PyAnyTorchTensorValue &key, const PyAnyTorchTensorValue &value, const PyAnyTorchOptionalTensorValue &attn_mask, const PyTorch_FloatValue &dropout_p, const PyTorch_BoolValue &is_causal, const PyAnyTorchOptionalFloatValue &scale, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, loc.get(), ip.get()); }, "query"_a, "key"_a, "value"_a, "attn_mask"_a = py::none(), "dropout_p"_a = 0., "is_causal"_a = false, "scale"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_add", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scatter_add(self, dim, index, src, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "src"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_add_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scatter_add_(self, dim, index, src, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "src"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
m.def("scatter_reduce", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scatter_reduce(self, dim, index, src, reduce, include_self, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "src"_a, "reduce"_a, "include_self"_a = true, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
m.def("scatter_reduce_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scatter_reduce_(self, dim, index, src, reduce, include_self, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "src"_a, "reduce"_a, "include_self"_a = true, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scatter.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scatter(self, dim, index, src, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "src"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scatter.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
m.def("scatter", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scatter(self, dim, index, value, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scatter_.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scatter_(self, dim, index, src, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "src"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::scatter_.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
m.def("scatter_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return scatter_(self, dim, index, value, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
m.def("select_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return select_copy(self, dim, index, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::select.int : (Tensor, int, int) -> (Tensor)
m.def("select", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return select(self, dim, index, loc.get(), ip.get()); }, "self"_a, "dim"_a, "index"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
m.def("select_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return select_scatter(self, src, dim, index, loc.get(), ip.get()); }, "self"_a, "src"_a, "dim"_a, "index"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sigmoid : (Tensor) -> (Tensor)
m.def("sigmoid", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sigmoid(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sigmoid_ : (Tensor) -> (Tensor)
m.def("sigmoid_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sigmoid_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sign : (Tensor) -> (Tensor)
m.def("sign", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sign(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sign_ : (Tensor) -> (Tensor)
m.def("sign_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sign_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::silu : (Tensor) -> (Tensor)
m.def("silu", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return silu(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::silu_ : (Tensor) -> (Tensor)
m.def("silu_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return silu_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sin : (Tensor) -> (Tensor)
m.def("sin", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sin(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sin_ : (Tensor) -> (Tensor)
m.def("sin_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sin_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::size.int : (Tensor, int) -> (int)
m.def("size", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return size(self, dim, loc.get(), ip.get()); }, "self"_a, "dim"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::size : (Tensor) -> (int[])
m.def("size", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListOfTorchIntValue { return size(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return slice_copy(self, dim, start, end, step, loc.get(), ip.get()); }, "self"_a, "dim"_a = 0, "start"_a = py::none(), "end"_a = py::none(), "step"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice_scatter", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return slice_scatter(self, src, dim, start, end, step, loc.get(), ip.get()); }, "self"_a, "src"_a, "dim"_a = 0, "start"_a = py::none(), "end"_a = py::none(), "step"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::slice.t : (t[], int?, int?, int) -> (t[])
m.def("slice", [](const PyAnyTorchListValue &l, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListValue { return slice(l, start, end, step, loc.get(), ip.get()); }, "l"_a, "start"_a = py::none(), "end"_a = py::none(), "step"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return slice(self, dim, start, end, step, loc.get(), ip.get()); }, "self"_a, "dim"_a = 0, "start"_a = py::none(), "end"_a = py::none(), "step"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
m.def("softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return softmax(self, dim, dtype, loc.get(), ip.get()); }, "self"_a, "dim"_a, "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sort.int : (int[], bool) -> ()
m.def("sort", [](const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> void { return sort(self, reverse, loc.get(), ip.get()); }, "self"_a, "reverse"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sort : (Tensor, int, bool) -> (Tensor, Tensor)
m.def("sort", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &descending, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return sort(self, dim, descending, loc.get(), ip.get()); }, "self"_a, "dim"_a = -1, "descending"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::split.Tensor : (Tensor, int, int) -> (Tensor[])
m.def("split", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &split_size, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListOfTensorValue { return split(self, split_size, dim, loc.get(), ip.get()); }, "self"_a, "split_size"_a, "dim"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sqrt.int : (int) -> (float)
m.def("sqrt", [](const PyTorch_IntValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return sqrt(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sqrt : (Tensor) -> (Tensor)
m.def("sqrt", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sqrt(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sqrt_ : (Tensor) -> (Tensor)
m.def("sqrt_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sqrt_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::square : (Tensor) -> (Tensor)
m.def("square", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return square(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::square_ : (Tensor) -> (Tensor)
m.def("square_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return square_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
m.def("squeeze_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return squeeze_copy(self, dim, loc.get(), ip.get()); }, "self"_a, "dim"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::squeeze_copy : (Tensor) -> (Tensor)
m.def("squeeze_copy", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return squeeze_copy(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
m.def("squeeze", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return squeeze(self, dim, loc.get(), ip.get()); }, "self"_a, "dim"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::squeeze : (Tensor) -> (Tensor)
m.def("squeeze", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return squeeze(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::stack : (Tensor[], int) -> (Tensor)
m.def("stack", [](const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return stack(tensors, dim, loc.get(), ip.get()); }, "tensors"_a, "dim"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::std.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
m.def("std", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return std(self, dim, correction, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "correction"_a = py::none(), "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::std.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
m.def("std", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return std(self, dim, unbiased, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "unbiased"_a = true, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::std : (Tensor, bool) -> (Tensor)
m.def("std", [](const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return std(self, unbiased, loc.get(), ip.get()); }, "self"_a, "unbiased"_a = true, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sub.float : (float, float) -> (float)
m.def("sub", [](const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_FloatValue { return sub(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sub.int : (int, int) -> (int)
m.def("sub", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return sub(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("sub", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sub(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("sub", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sub(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sub_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("sub_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sub_(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sub_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("sub_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sub_(self, other, alpha, loc.get(), ip.get()); }, "self"_a, "other"_a, "alpha"_a = 1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sum.dim_IntList : (Tensor, int[]?, bool, int?) -> (Tensor)
m.def("sum", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sum(self, dim, keepdim, dtype, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "keepdim"_a = false, "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::sum : (Tensor, int?) -> (Tensor)
m.def("sum", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return sum(self, dtype, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::t_copy : (Tensor) -> (Tensor)
m.def("t_copy", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return t_copy(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::t : (Tensor) -> (Tensor)
m.def("t", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return t(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
m.def("tanh_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tanh_backward(grad_output, output, loc.get(), ip.get()); }, "grad_output"_a, "output"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tanh : (Tensor) -> (Tensor)
m.def("tanh", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tanh(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tanh_ : (Tensor) -> (Tensor)
m.def("tanh_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tanh_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tensor(t, dtype, device, requires_grad, loc.get(), ip.get()); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tensor(t, dtype, device, requires_grad, loc.get(), ip.get()); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tensor(t, dtype, device, requires_grad, loc.get(), ip.get()); }, "t"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tensor : (t[], int?, Device?, bool) -> (Tensor)
m.def("tensor", [](const PyAnyTorchListValue &data, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tensor(data, dtype, device, requires_grad, loc.get(), ip.get()); }, "data"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("threshold_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return threshold_backward(grad_output, self, threshold__, loc.get(), ip.get()); }, "grad_output"_a, "self"_a, "threshold__"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("threshold", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return threshold(self, threshold__, value, loc.get(), ip.get()); }, "self"_a, "threshold__"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::threshold_ : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("threshold_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return threshold_(self, threshold__, value, loc.get(), ip.get()); }, "self"_a, "threshold__"_a, "value"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return to(self, device, dtype, non_blocking, copy, memory_format, loc.get(), ip.get()); }, "self"_a, "device"_a, "dtype"_a, "non_blocking"_a = false, "copy"_a = false, "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return to(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "non_blocking"_a = false, "copy"_a = false, "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return to(self, dtype, non_blocking, copy, memory_format, loc.get(), ip.get()); }, "self"_a, "dtype"_a, "non_blocking"_a = false, "copy"_a = false, "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return to(self, other, non_blocking, copy, memory_format, loc.get(), ip.get()); }, "self"_a, "other"_a, "non_blocking"_a = false, "copy"_a = false, "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
m.def("to", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return to(self, device, dtype, non_blocking, copy, loc.get(), ip.get()); }, "self"_a, "device"_a = py::none(), "dtype"_a = py::none(), "non_blocking"_a = false, "copy"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::topk : (Tensor, int, int, bool, bool) -> (Tensor, Tensor)
m.def("topk", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &k, const PyTorch_IntValue &dim, const PyTorch_BoolValue &largest, const PyTorch_BoolValue &sorted, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return topk(self, k, dim, largest, sorted, loc.get(), ip.get()); }, "self"_a, "k"_a, "dim"_a = -1, "largest"_a = true, "sorted"_a = true, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
m.def("transpose_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return transpose_copy(self, dim0, dim1, loc.get(), ip.get()); }, "self"_a, "dim0"_a, "dim1"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
m.def("transpose", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return transpose(self, dim0, dim1, loc.get(), ip.get()); }, "self"_a, "dim0"_a, "dim1"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tril : (Tensor, int) -> (Tensor)
m.def("tril", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tril(self, diagonal, loc.get(), ip.get()); }, "self"_a, "diagonal"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::tril_ : (Tensor, int) -> (Tensor)
m.def("tril_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return tril_(self, diagonal, loc.get(), ip.get()); }, "self"_a, "diagonal"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::triu : (Tensor, int) -> (Tensor)
m.def("triu", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return triu(self, diagonal, loc.get(), ip.get()); }, "self"_a, "diagonal"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::triu_ : (Tensor, int) -> (Tensor)
m.def("triu_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return triu_(self, diagonal, loc.get(), ip.get()); }, "self"_a, "diagonal"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::type_as : (Tensor, Tensor) -> (Tensor)
m.def("type_as", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return type_as(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::unbind.int : (Tensor, int) -> (Tensor[])
m.def("unbind", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListOfTensorValue { return unbind(self, dim, loc.get(), ip.get()); }, "self"_a, "dim"_a = 0, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
m.def("unfold_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return unfold_copy(self, dimension, size, step, loc.get(), ip.get()); }, "self"_a, "dimension"_a, "size"_a, "step"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
m.def("uniform", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return uniform(self, from, to, generator, loc.get(), ip.get()); }, "self"_a, "from"_a = 0., "to"_a = 1., "generator"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
m.def("uniform_", [](const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return uniform_(self, from, to, generator, loc.get(), ip.get()); }, "self"_a, "from"_a = 0., "to"_a = 1., "generator"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
m.def("unsqueeze_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return unsqueeze_copy(self, dim, loc.get(), ip.get()); }, "self"_a, "dim"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::unsqueeze : (Tensor, int) -> (Tensor)
m.def("unsqueeze", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return unsqueeze(self, dim, loc.get(), ip.get()); }, "self"_a, "dim"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
m.def("unsqueeze_", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return unsqueeze_(self, dim, loc.get(), ip.get()); }, "self"_a, "dim"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
m.def("upsample_nearest2d_backward", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h, scales_w, loc.get(), ip.get()); }, "grad_output"_a, "output_size"_a, "input_size"_a, "scales_h"_a = py::none(), "scales_w"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
m.def("upsample_nearest2d", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return upsample_nearest2d(self, output_size, scales_h, scales_w, loc.get(), ip.get()); }, "self"_a, "output_size"_a, "scales_h"_a = py::none(), "scales_w"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::var.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
m.def("var", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return var(self, dim, correction, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "correction"_a = py::none(), "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::var.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
m.def("var", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return var(self, dim, unbiased, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "unbiased"_a = true, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::var_mean.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor, Tensor)
m.def("var_mean", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return var_mean(self, dim, correction, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "correction"_a = py::none(), "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::var_mean.dim : (Tensor, int[]?, bool, bool) -> (Tensor, Tensor)
m.def("var_mean", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return var_mean(self, dim, unbiased, keepdim, loc.get(), ip.get()); }, "self"_a, "dim"_a = py::none(), "unbiased"_a = true, "keepdim"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::var_mean : (Tensor, bool) -> (Tensor, Tensor)
m.def("var_mean", [](const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return var_mean(self, unbiased, loc.get(), ip.get()); }, "self"_a, "unbiased"_a = true, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::var : (Tensor, bool) -> (Tensor)
m.def("var", [](const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return var(self, unbiased, loc.get(), ip.get()); }, "self"_a, "unbiased"_a = true, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::view_as_complex : (Tensor) -> (Tensor)
m.def("view_as_complex", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return view_as_complex(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
m.def("view_copy", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return view_copy(self, dtype, loc.get(), ip.get()); }, "self"_a, "dtype"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::view_copy : (Tensor, int[]) -> (Tensor)
m.def("view_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return view_copy(self, size, loc.get(), ip.get()); }, "self"_a, "size"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::view : (Tensor, int[]) -> (Tensor)
m.def("view", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return view(self, size, loc.get(), ip.get()); }, "self"_a, "size"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::where.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
m.def("where", [](const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return where(condition, self, other, loc.get(), ip.get()); }, "condition"_a, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::where.ScalarOther : (Tensor, Tensor, Scalar) -> (Tensor)
m.def("where", [](const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return where(condition, self, other, loc.get(), ip.get()); }, "condition"_a, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::where.ScalarSelf : (Tensor, Scalar, Tensor) -> (Tensor)
m.def("where", [](const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return where(condition, self, other, loc.get(), ip.get()); }, "condition"_a, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("where", [](const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return where(condition, self, other, loc.get(), ip.get()); }, "condition"_a, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::zero : (Tensor) -> (Tensor)
m.def("zero", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return zero(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::zero_ : (Tensor) -> (Tensor)
m.def("zero_", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return zero_(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("zeros_like", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return zeros_like(self, dtype, layout, device, pin_memory, memory_format, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("zeros", [](const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return zeros(size, dtype, layout, device, pin_memory, loc.get(), ip.get()); }, "size"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_convolution.deprecated : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool) -> (Tensor)
m.def("_convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, loc.get(), ip.get()); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a, "benchmark"_a, "deterministic"_a, "cudnn_enabled"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool, bool) -> (Tensor)
m.def("_convolution", [](const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, const PyTorch_BoolValue &allow_tf32, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, loc.get(), ip.get()); }, "input"_a, "weight"_a, "bias"_a = py::none(), "stride"_a, "padding"_a, "dilation"_a, "transposed"_a, "output_padding"_a, "groups"_a, "benchmark"_a, "deterministic"_a, "cudnn_enabled"_a, "allow_tf32"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_embedding_bag : (Tensor, Tensor, Tensor, bool, int, bool, Tensor?, bool, int) -> (Tensor, Tensor, Tensor, Tensor)
m.def("_embedding_bag", [](const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyAnyTorchTensorValue &offsets, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_IntValue &mode, const PyTorch_BoolValue &sparse, const PyAnyTorchOptionalTensorValue &per_sample_weights, const PyTorch_BoolValue &include_last_offset, const PyTorch_IntValue &padding_idx, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> { return _embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, loc.get(), ip.get()); }, "weight"_a, "indices"_a, "offsets"_a, "scale_grad_by_freq"_a = false, "mode"_a = 0, "sparse"_a = false, "per_sample_weights"_a = py::none(), "include_last_offset"_a = false, "padding_idx"_a = -1, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_index_put_impl : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)
m.def("_index_put_impl", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, const PyTorch_BoolValue &unsafe, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _index_put_impl(self, indices, values, accumulate, unsafe, loc.get(), ip.get()); }, "self"_a, "indices"_a, "values"_a, "accumulate"_a = false, "unsafe"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_index_put_impl_ : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)
m.def("_index_put_impl_", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, const PyTorch_BoolValue &unsafe, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _index_put_impl_(self, indices, values, accumulate, unsafe, loc.get(), ip.get()); }, "self"_a, "indices"_a, "values"_a, "accumulate"_a = false, "unsafe"_a = false, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
m.def("_log_softmax_backward_data", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _log_softmax_backward_data(grad_output, output, dim, input_dtype, loc.get(), ip.get()); }, "grad_output"_a, "output"_a, "dim"_a, "input_dtype"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
m.def("_log_softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _log_softmax(self, dim, half_to_float, loc.get(), ip.get()); }, "self"_a, "dim"_a, "half_to_float"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
m.def("_reshape_alias_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _reshape_alias_copy(self, size, stride, loc.get(), ip.get()); }, "self"_a, "size"_a, "stride"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
m.def("_reshape_alias", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _reshape_alias(self, size, stride, loc.get(), ip.get()); }, "self"_a, "size"_a, "stride"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
m.def("_shape_as_tensor", [](const PyAnyTorchTensorValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _shape_as_tensor(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
m.def("_softmax_backward_data", [](const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _softmax_backward_data(grad_output, output, dim, input_dtype, loc.get(), ip.get()); }, "grad_output"_a, "output"_a, "dim"_a, "input_dtype"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
m.def("_softmax", [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _softmax(self, dim, half_to_float, loc.get(), ip.get()); }, "self"_a, "dim"_a, "half_to_float"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
m.def("_to_copy", [](const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _to_copy(self, dtype, layout, device, pin_memory, non_blocking, memory_format, loc.get(), ip.get()); }, "self"_a, "dtype"_a = py::none(), "layout"_a = py::none(), "device"_a = py::none(), "pin_memory"_a = py::none(), "non_blocking"_a = false, "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
m.def("_unsafe_view", [](const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return _unsafe_view(self, size, loc.get(), ip.get()); }, "self"_a, "size"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::__and__.bool : (bool, bool) -> (bool)
m.def("__and__", [](const PyTorch_BoolValue &a, const PyTorch_BoolValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return __and__(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("__and__", [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return __and__(self, other, loc.get(), ip.get()); }, "self"_a, "other"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::__contains__.int_list : (int[], int) -> (bool)
m.def("__contains__", [](const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return __contains__(l, item, loc.get(), ip.get()); }, "l"_a, "item"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
m.def("__contains__", [](const PyTorch_DictValue &dict, const PyTorch_StringValue &key, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return __contains__(dict, key, loc.get(), ip.get()); }, "dict"_a, "key"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::__derive_index : (int, int, int) -> (int)
m.def("__derive_index", [](const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return __derive_index(index, start, step, loc.get(), ip.get()); }, "index"_a, "start"_a, "step"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::__not__ : (bool) -> (bool)
m.def("__not__", [](const PyTorch_BoolValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_BoolValue { return __not__(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// aten::__range_length : (int, int, int) -> (int)
m.def("__range_length", [](const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return __range_length(lo, hi, step, loc.get(), ip.get()); }, "lo"_a, "hi"_a, "step"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::device : (Tensor) -> (Device)
m.def("device", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_DeviceValue { return device(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::dtype : (Tensor) -> (int)
m.def("dtype", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return dtype(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::layout : (Tensor) -> (int)
m.def("layout", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return layout(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::max.int : (int, int) -> (int)
m.def("max", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return max(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::max.self_int : (int[]) -> (int)
m.def("max", [](const PyAnyTorchListOfTorchIntValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return max(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::min.int : (int, int) -> (int)
m.def("min", [](const PyTorch_IntValue &a, const PyTorch_IntValue &b, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return min(a, b, loc.get(), ip.get()); }, "a"_a, "b"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::min.self_int : (int[]) -> (int)
m.def("min", [](const PyAnyTorchListOfTorchIntValue &self, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyTorch_IntValue { return min(self, loc.get(), ip.get()); }, "self"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::NumToTensor.Scalar : (Scalar) -> (Tensor)
m.def("NumToTensor", [](const PyAnyTorchScalarValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return NumToTensor(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prim::RaiseException : (str, str?) -> ()
m.def("RaiseException", [](const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> void { return RaiseException(msg, cls, loc.get(), ip.get()); }, "msg"_a, "cls"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prims::convert_element_type : (Tensor, int) -> (Tensor)
m.def("convert_element_type", [](const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return convert_element_type(a, dtype, loc.get(), ip.get()); }, "a"_a, "dtype"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prims::squeeze : (Tensor, int[]) -> (Tensor)
m.def("squeeze", [](const PyAnyTorchTensorValue &a, const PyAnyTorchListOfTorchIntValue &dimensions, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return squeeze(a, dimensions, loc.get(), ip.get()); }, "a"_a, "dimensions"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prims::var : (Tensor, int[]?, float, int?) -> (Tensor)
m.def("var", [](const PyAnyTorchTensorValue &inp, const PyAnyTorchOptionalListOfTorchIntValue &dims, const PyTorch_FloatValue &correction, const PyAnyTorchOptionalIntValue &output_dtype, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return var(inp, dims, correction, output_dtype, loc.get(), ip.get()); }, "inp"_a, "dims"_a = py::none(), "correction"_a, "output_dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// prims::view_of : (Tensor) -> (Tensor)
m.def("view_of", [](const PyAnyTorchTensorValue &a, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return view_of(a, loc.get(), ip.get()); }, "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
m.def("linear", [](const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i, DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue { return linear(X, W_prepack, Y_scale_i, Y_zero_point_i, loc.get(), ip.get()); }, "X"_a, "W_prepack"_a, "Y_scale_i"_a, "Y_zero_point_i"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());
