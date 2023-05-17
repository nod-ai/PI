
// aten::Bool.Tensor : (Tensor) -> (bool)
py::object Bool(const PyAnyTorchTensorValue &a);

// aten::Bool.float : (float) -> (bool)
py::object Bool(const PyTorch_FloatValue &a);

// aten::Bool.int : (int) -> (bool)
py::object Bool(const PyTorch_IntValue &a);

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
py::object Delete(const PyTorch_DictValue &self, const PyTorch_StringValue &key);

// aten::Float.Tensor : (Tensor) -> (float)
py::object Float(const PyAnyTorchTensorValue &a);

// aten::Float.str : (str) -> (float)
py::object Float(const PyTorch_StringValue &a);

// aten::FloatImplicit : (Tensor) -> (float)
py::object FloatImplicit(const PyAnyTorchTensorValue &a);

// aten::Int.Tensor : (Tensor) -> (int)
py::object Int(const PyAnyTorchTensorValue &a);

// aten::Int.float : (float) -> (int)
py::object Int(const PyTorch_FloatValue &a);

// aten::Int.bool : (bool) -> (int)
py::object Int(const PyTorch_BoolValue &a);

// aten::IntImplicit : (Tensor) -> (int)
py::object IntImplicit(const PyAnyTorchTensorValue &a);

// prim::RaiseException : (str, str?) -> ()
py::object RaiseException(const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls);

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
py::object __and__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::__and__.bool : (bool, bool) -> (bool)
py::object __and__(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b);

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
py::object __contains__(const PyTorch_DictValue &dict, const PyTorch_StringValue &key);

// aten::__contains__.int_list : (int[], int) -> (bool)
py::object __contains__(const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item);

// aten::__derive_index : (int, int, int) -> (int)
py::object __derive_index(const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step);

// aten::__not__ : (bool) -> (bool)
py::object __not__(const PyTorch_BoolValue &self);

// aten::__range_length : (int, int, int) -> (int)
py::object __range_length(const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step);

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
py::object _log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float);

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
py::object _log_softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype);

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
py::object _reshape_alias(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride);

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
py::object _reshape_alias_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride);

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
py::object _shape_as_tensor(const PyAnyTorchTensorValue &self);

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
py::object _softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float);

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
py::object _softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype);

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
py::object _to_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format);

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
py::object _unsafe_view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size);

// aten::abs : (Tensor) -> (Tensor)
py::object abs(const PyAnyTorchTensorValue &self);

// aten::abs_ : (Tensor) -> (Tensor)
py::object abs_(const PyAnyTorchTensorValue &self);

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
py::object adaptive_avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size);

// aten::add.str : (str, str) -> (str)
py::object add(const PyTorch_StringValue &a, const PyTorch_StringValue &b);

// aten::add.int : (int, int) -> (int)
py::object add(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::add.float_int : (float, int) -> (float)
py::object add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::alias_copy : (Tensor) -> (Tensor)
py::object alias_copy(const PyAnyTorchTensorValue &self);

// aten::all : (Tensor) -> (Tensor)
py::object all(const PyAnyTorchTensorValue &self);

// aten::all.bool : (bool[]) -> (bool)
py::object all(const PyAnyTorchListOfTorchBoolValue &self);

// aten::amax : (Tensor, int[], bool) -> (Tensor)
py::object amax(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::any : (Tensor) -> (Tensor)
py::object any(const PyAnyTorchTensorValue &self);

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
py::object any(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::any.bool : (bool[]) -> (bool)
py::object any(const PyAnyTorchListOfTorchBoolValue &self);

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
py::object argmax(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
py::object as_strided_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset);

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
py::object as_strided_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset);

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
py::object atan2(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
py::object atan2_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
py::object avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override);

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
py::object bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
py::object bernoulli_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
py::object bernoulli_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_not : (Tensor) -> (Tensor)
py::object bitwise_not(const PyAnyTorchTensorValue &self);

// aten::bitwise_not_ : (Tensor) -> (Tensor)
py::object bitwise_not_(const PyAnyTorchTensorValue &self);

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object bitwise_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bmm : (Tensor, Tensor) -> (Tensor)
py::object bmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2);

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
py::object broadcast_to(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size);

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
py::object bucketize(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right);

// aten::ceil : (Tensor) -> (Tensor)
py::object ceil(const PyAnyTorchTensorValue &self);

// aten::ceil.float : (float) -> (int)
py::object ceil(const PyTorch_FloatValue &a);

// aten::ceil_ : (Tensor) -> (Tensor)
py::object ceil_(const PyAnyTorchTensorValue &self);

// aten::clone : (Tensor, int?) -> (Tensor)
py::object clone(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format);

// aten::contiguous : (Tensor, int) -> (Tensor)
py::object contiguous(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format);

// prims::convert_element_type : (Tensor, int) -> (Tensor)
py::object convert_element_type(const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype);

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
py::object copy(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking);

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
py::object copy_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking);

// aten::cos : (Tensor) -> (Tensor)
py::object cos(const PyAnyTorchTensorValue &self);

// aten::cos_ : (Tensor) -> (Tensor)
py::object cos_(const PyAnyTorchTensorValue &self);

// aten::cpu : (Tensor) -> (Tensor)
py::object cpu(const PyAnyTorchTensorValue &self);

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
py::object cumsum(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype);

// aten::detach : (Tensor) -> (Tensor)
py::object detach(const PyAnyTorchTensorValue &self);

// aten::detach_copy : (Tensor) -> (Tensor)
py::object detach_copy(const PyAnyTorchTensorValue &self);

// prim::device : (Tensor) -> (Device)
py::object device(const PyAnyTorchTensorValue &a);

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
py::object diagonal_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2);

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
py::object diagonal_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2);

// aten::dim : (Tensor) -> (int)
py::object dim(const PyAnyTorchTensorValue &self);

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
py::object div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
py::object div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode);

// aten::div.int : (int, int) -> (float)
py::object div(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::div.float : (float, float) -> (float)
py::object div(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
py::object div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode);

// aten::dropout : (Tensor, float, bool) -> (Tensor)
py::object dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train);

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
py::object dropout_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train);

// prim::dtype : (Tensor) -> (int)
py::object dtype(const PyAnyTorchTensorValue &a);

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
py::object embedding(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse);

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
py::object embedding_dense_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq);

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
py::object empty(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object empty_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
py::object eq(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::eq.int_list : (int[], int[]) -> (bool)
py::object eq(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b);

// aten::eq.str : (str, str) -> (bool)
py::object eq(const PyTorch_StringValue &a, const PyTorch_StringValue &b);

// aten::eq.int : (int, int) -> (bool)
py::object eq(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::eq.float : (float, float) -> (bool)
py::object eq(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::eq.device : (Device, Device) -> (bool)
py::object eq(const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b);

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::erf : (Tensor) -> (Tensor)
py::object erf(const PyAnyTorchTensorValue &self);

// aten::erf_ : (Tensor) -> (Tensor)
py::object erf_(const PyAnyTorchTensorValue &self);

// aten::exp : (Tensor) -> (Tensor)
py::object exp(const PyAnyTorchTensorValue &self);

// aten::exp_ : (Tensor) -> (Tensor)
py::object exp_(const PyAnyTorchTensorValue &self);

// aten::expand : (Tensor, int[], bool) -> (Tensor)
py::object expand(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit);

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
py::object expand_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
py::object expand_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit);

// aten::expm1 : (Tensor) -> (Tensor)
py::object expm1(const PyAnyTorchTensorValue &self);

// aten::expm1_ : (Tensor) -> (Tensor)
py::object expm1_(const PyAnyTorchTensorValue &self);

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
py::object fft_fft(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm);

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
py::object fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value);

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value);

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
py::object flatten(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim);

// aten::flip : (Tensor, int[]) -> (Tensor)
py::object flip(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims);

// aten::floor : (Tensor) -> (Tensor)
py::object floor(const PyAnyTorchTensorValue &self);

// aten::floor_ : (Tensor) -> (Tensor)
py::object floor_(const PyAnyTorchTensorValue &self);

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
py::object floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::floordiv.int : (int, int) -> (int)
py::object floordiv(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
py::object frobenius_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
py::object gather(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad);

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ge(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::ge.int : (int, int) -> (bool)
py::object ge(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::ge.float : (float, float) -> (bool)
py::object ge(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::ge.float_int : (float, int) -> (bool)
py::object ge(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::gelu : (Tensor, str) -> (Tensor)
py::object gelu(const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate);

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
py::object gelu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate);

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
py::object gt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::gt.int : (int, int) -> (bool)
py::object gt(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::gt.float : (float, float) -> (bool)
py::object gt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::gt.float_int : (float, int) -> (bool)
py::object gt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::hardsigmoid : (Tensor) -> (Tensor)
py::object hardsigmoid(const PyAnyTorchTensorValue &self);

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
py::object hardsigmoid_(const PyAnyTorchTensorValue &self);

// aten::hardswish : (Tensor) -> (Tensor)
py::object hardswish(const PyAnyTorchTensorValue &self);

// aten::hardswish_ : (Tensor) -> (Tensor)
py::object hardswish_(const PyAnyTorchTensorValue &self);

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
py::object index_select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index);

// aten::is_floating_point : (Tensor) -> (bool)
py::object is_floating_point(const PyAnyTorchTensorValue &self);

// aten::join : (str, str[]) -> (str)
py::object join(const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values);

// aten::keys.str : (Dict(str, t)) -> (str[])
py::object keys(const PyTorch_DictValue &self);

// prim::layout : (Tensor) -> (int)
py::object layout(const PyAnyTorchTensorValue &a);

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
py::object le(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::le.int : (int, int) -> (bool)
py::object le(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object le_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::len.Tensor : (Tensor) -> (int)
py::object len(const PyAnyTorchTensorValue &t);

// aten::len.str : (str) -> (int)
py::object len(const PyTorch_StringValue &s);

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object lerp(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight);

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object lerp_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight);

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
py::object lift_fresh_copy(const PyAnyTorchTensorValue &self);

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
py::object linear(const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i);

// aten::log : (Tensor) -> (Tensor)
py::object log(const PyAnyTorchTensorValue &self);

// aten::log.int : (int) -> (float)
py::object log(const PyTorch_IntValue &a);

// aten::log1p : (Tensor) -> (Tensor)
py::object log1p(const PyAnyTorchTensorValue &self);

// aten::log1p_ : (Tensor) -> (Tensor)
py::object log1p_(const PyAnyTorchTensorValue &self);

// aten::log2 : (Tensor) -> (Tensor)
py::object log2(const PyAnyTorchTensorValue &self);

// aten::log2_ : (Tensor) -> (Tensor)
py::object log2_(const PyAnyTorchTensorValue &self);

// aten::log_ : (Tensor) -> (Tensor)
py::object log_(const PyAnyTorchTensorValue &self);

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
py::object log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype);

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
py::object logical_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
py::object logical_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_not : (Tensor) -> (Tensor)
py::object logical_not(const PyAnyTorchTensorValue &self);

// aten::logical_not_ : (Tensor) -> (Tensor)
py::object logical_not_(const PyAnyTorchTensorValue &self);

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
py::object logical_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
py::object logical_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
py::object logical_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
py::object logical_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
py::object logsumexp(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
py::object lt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::lt.int : (int, int) -> (bool)
py::object lt(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::lt.float : (float, float) -> (bool)
py::object lt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::lt.float_int : (float, int) -> (bool)
py::object lt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value);

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
py::object masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value);

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
py::object masked_select(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask);

// aten::matmul : (Tensor, Tensor) -> (Tensor)
py::object matmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::max : (Tensor) -> (Tensor)
py::object max(const PyAnyTorchTensorValue &self);

// prim::max.self_int : (int[]) -> (int)
py::object max(const PyAnyTorchListOfTorchIntValue &self);

// prim::max.int : (int, int) -> (int)
py::object max(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
py::object max_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode);

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
py::object max_pool2d_with_indices_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices);

// aten::maximum : (Tensor, Tensor) -> (Tensor)
py::object maximum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::mean : (Tensor, int?) -> (Tensor)
py::object mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype);

// prim::min.self_int : (int[]) -> (int)
py::object min(const PyAnyTorchListOfTorchIntValue &self);

// prim::min.int : (int, int) -> (int)
py::object min(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::minimum : (Tensor, Tensor) -> (Tensor)
py::object minimum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::mish : (Tensor) -> (Tensor)
py::object mish(const PyAnyTorchTensorValue &self);

// aten::mm : (Tensor, Tensor) -> (Tensor)
py::object mm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2);

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
py::object mse_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction);

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
py::object mul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::mul.int : (int, int) -> (int)
py::object mul(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::mul.float : (float, float) -> (float)
py::object mul(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::mv : (Tensor, Tensor) -> (Tensor)
py::object mv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec);

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
py::object narrow(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length);

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
py::object native_dropout_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale);

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ne(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::ne.int_list : (int[], int[]) -> (bool)
py::object ne(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b);

// aten::ne.int : (int, int) -> (bool)
py::object ne(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::ne.float_int : (float, int) -> (bool)
py::object ne(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::ne.bool : (bool, bool) -> (bool)
py::object ne(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b);

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
py::object ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::neg : (Tensor) -> (Tensor)
py::object neg(const PyAnyTorchTensorValue &self);

// aten::neg.int : (int) -> (int)
py::object neg(const PyTorch_IntValue &a);

// aten::neg.float : (float) -> (float)
py::object neg(const PyTorch_FloatValue &a);

// aten::neg_ : (Tensor) -> (Tensor)
py::object neg_(const PyAnyTorchTensorValue &self);

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_empty(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_empty_strided(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_ones(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object new_zeros(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::numel : (Tensor) -> (int)
py::object numel(const PyAnyTorchTensorValue &self);

// aten::numpy_T : (Tensor) -> (Tensor)
py::object numpy_T(const PyAnyTorchTensorValue &self);

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object ones(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object ones_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
py::object pad(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value);

// aten::permute : (Tensor, int[]) -> (Tensor)
py::object permute(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims);

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
py::object permute_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims);

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
py::object pow(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent);

// aten::pow.int_float : (int, float) -> (float)
py::object pow(const PyTorch_IntValue &a, const PyTorch_FloatValue &b);

// aten::prelu : (Tensor, Tensor) -> (Tensor)
py::object prelu(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight);

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object rand_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
py::object randint(const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
py::object randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object randn_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::reciprocal : (Tensor) -> (Tensor)
py::object reciprocal(const PyAnyTorchTensorValue &self);

// aten::reciprocal_ : (Tensor) -> (Tensor)
py::object reciprocal_(const PyAnyTorchTensorValue &self);

// aten::relu : (Tensor) -> (Tensor)
py::object relu(const PyAnyTorchTensorValue &self);

// aten::relu6 : (Tensor) -> (Tensor)
py::object relu6(const PyAnyTorchTensorValue &self);

// aten::relu6_ : (Tensor) -> (Tensor)
py::object relu6_(const PyAnyTorchTensorValue &self);

// aten::relu_ : (Tensor) -> (Tensor)
py::object relu_(const PyAnyTorchTensorValue &self);

// aten::remainder.int : (int, int) -> (int)
py::object remainder(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::repeat : (Tensor, int[]) -> (Tensor)
py::object repeat(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats);

// aten::reshape : (Tensor, int[]) -> (Tensor)
py::object reshape(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape);

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
py::object resize_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format);

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
py::object roll(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims);

// aten::round : (Tensor) -> (Tensor)
py::object round(const PyAnyTorchTensorValue &self);

// aten::round_ : (Tensor) -> (Tensor)
py::object round_(const PyAnyTorchTensorValue &self);

// aten::rsqrt : (Tensor) -> (Tensor)
py::object rsqrt(const PyAnyTorchTensorValue &self);

// aten::rsqrt_ : (Tensor) -> (Tensor)
py::object rsqrt_(const PyAnyTorchTensorValue &self);

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
py::object scatter_add(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src);

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
py::object scatter_add_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src);

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
py::object scatter_reduce(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self);

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
py::object scatter_reduce_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self);

// aten::select.int : (Tensor, int, int) -> (Tensor)
py::object select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index);

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
py::object select_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index);

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
py::object select_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index);

// aten::sigmoid : (Tensor) -> (Tensor)
py::object sigmoid(const PyAnyTorchTensorValue &self);

// aten::sigmoid_ : (Tensor) -> (Tensor)
py::object sigmoid_(const PyAnyTorchTensorValue &self);

// aten::silu : (Tensor) -> (Tensor)
py::object silu(const PyAnyTorchTensorValue &self);

// aten::silu_ : (Tensor) -> (Tensor)
py::object silu_(const PyAnyTorchTensorValue &self);

// aten::sin : (Tensor) -> (Tensor)
py::object sin(const PyAnyTorchTensorValue &self);

// aten::sin_ : (Tensor) -> (Tensor)
py::object sin_(const PyAnyTorchTensorValue &self);

// aten::size : (Tensor) -> (int[])
py::object size(const PyAnyTorchTensorValue &self);

// aten::size.int : (Tensor, int) -> (int)
py::object size(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
py::object slice(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step);

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
py::object slice_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step);

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
py::object slice_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step);

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
py::object softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype);

// aten::sort.int : (int[], bool) -> ()
py::object sort(const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse);

// aten::sqrt : (Tensor) -> (Tensor)
py::object sqrt(const PyAnyTorchTensorValue &self);

// aten::sqrt.int : (int) -> (float)
py::object sqrt(const PyTorch_IntValue &a);

// aten::sqrt_ : (Tensor) -> (Tensor)
py::object sqrt_(const PyAnyTorchTensorValue &self);

// aten::square : (Tensor) -> (Tensor)
py::object square(const PyAnyTorchTensorValue &self);

// aten::square_ : (Tensor) -> (Tensor)
py::object square_(const PyAnyTorchTensorValue &self);

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
py::object squeeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::squeeze : (Tensor) -> (Tensor)
py::object squeeze(const PyAnyTorchTensorValue &self);

// aten::squeeze_copy : (Tensor) -> (Tensor)
py::object squeeze_copy(const PyAnyTorchTensorValue &self);

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
py::object squeeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::std : (Tensor, bool) -> (Tensor)
py::object std(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased);

// aten::sub.int : (int, int) -> (int)
py::object sub(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::sub.float : (float, float) -> (float)
py::object sub(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::sum : (Tensor, int?) -> (Tensor)
py::object sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype);

// aten::t : (Tensor) -> (Tensor)
py::object t(const PyAnyTorchTensorValue &self);

// aten::t_copy : (Tensor) -> (Tensor)
py::object t_copy(const PyAnyTorchTensorValue &self);

// aten::tanh : (Tensor) -> (Tensor)
py::object tanh(const PyAnyTorchTensorValue &self);

// aten::tanh_ : (Tensor) -> (Tensor)
py::object tanh_(const PyAnyTorchTensorValue &self);

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
py::object tanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output);

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad);

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad);

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
py::object tensor(const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad);

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format);

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format);

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format);

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy);

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
py::object to(const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format);

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
py::object transpose(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1);

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
py::object transpose_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1);

// aten::triu : (Tensor, int) -> (Tensor)
py::object triu(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal);

// aten::triu_ : (Tensor, int) -> (Tensor)
py::object triu_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal);

// aten::type_as : (Tensor, Tensor) -> (Tensor)
py::object type_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
py::object unfold_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step);

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
py::object uniform(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
py::object uniform_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::unsqueeze : (Tensor, int) -> (Tensor)
py::object unsqueeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
py::object unsqueeze_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
py::object unsqueeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
py::object upsample_nearest2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w);

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
py::object upsample_nearest2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w);

// aten::var : (Tensor, bool) -> (Tensor)
py::object var(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased);

// aten::view : (Tensor, int[]) -> (Tensor)
py::object view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size);

// aten::view_copy : (Tensor, int[]) -> (Tensor)
py::object view_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size);

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
py::object view_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype);

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
py::object where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::zero : (Tensor) -> (Tensor)
py::object zero(const PyAnyTorchTensorValue &self);

// aten::zero_ : (Tensor) -> (Tensor)
py::object zero_(const PyAnyTorchTensorValue &self);

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
py::object zeros(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
py::object zeros_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);