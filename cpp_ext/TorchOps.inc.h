
// aten::abs : (Tensor) -> (Tensor)
PyAnyTorchTensorValue abs(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::abs_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue abs_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue adaptive_avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, PyLocation *loc, PyInsertionPoint *ip);

// aten::add.float_int : (float, int) -> (float)
PyTorch_FloatValue add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::add.int : (int, int) -> (int)
PyTorch_IntValue add(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue add(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::add.str : (str, str) -> (str)
PyTorch_StringValue add(const PyTorch_StringValue &a, const PyTorch_StringValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::add.t : (t[], t[]) -> (t[])
PyAnyTorchListValue add(const PyAnyTorchListValue &a, const PyAnyTorchListValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue add(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::add_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue add_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::add_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue add_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcdiv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::addcdiv_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcdiv_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::addcmul_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcmul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue addmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat1, const PyAnyTorchTensorValue &mat2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::alias_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue alias_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::all.bool : (bool[]) -> (bool)
PyTorch_BoolValue all(const PyAnyTorchListOfTorchBoolValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::all : (Tensor) -> (Tensor)
PyAnyTorchTensorValue all(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::amax : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue amax(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::any.bool : (bool[]) -> (bool)
PyTorch_BoolValue any(const PyAnyTorchListOfTorchBoolValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue any(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::any : (Tensor) -> (Tensor)
PyAnyTorchTensorValue any(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::arange.start_out : (Scalar, Scalar, Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchTensorValue &out, PyLocation *loc, PyInsertionPoint *ip);

// aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
PyAnyTorchTensorValue argmax(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
PyAnyTorchTensorValue as_strided_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset, PyLocation *loc, PyInsertionPoint *ip);

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
PyAnyTorchTensorValue as_strided_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset, PyLocation *loc, PyInsertionPoint *ip);

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue atan2(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue atan2_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::atan : (Tensor) -> (Tensor)
PyAnyTorchTensorValue atan(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::atan_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue atan_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override, PyLocation *loc, PyInsertionPoint *ip);

// aten::baddbmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue baddbmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::baddbmm_ : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue baddbmm_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)
PyAnyTorchTensorValue batch_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enabled, PyLocation *loc, PyInsertionPoint *ip);

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip);

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip);

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip);

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip);

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip);

// aten::bincount : (Tensor, Tensor?, int) -> (Tensor)
PyAnyTorchTensorValue bincount(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &weights, const PyTorch_IntValue &minlength, PyLocation *loc, PyInsertionPoint *ip);

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::bitwise_not : (Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_not(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::bitwise_not_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_not_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::bmm : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2, PyLocation *loc, PyInsertionPoint *ip);

// aten::Bool.float : (float) -> (bool)
PyTorch_BoolValue Bool(const PyTorch_FloatValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::Bool.int : (int) -> (bool)
PyTorch_BoolValue Bool(const PyTorch_IntValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::Bool.Tensor : (Tensor) -> (bool)
PyTorch_BoolValue Bool(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue broadcast_to(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, PyLocation *loc, PyInsertionPoint *ip);

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue bucketize(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right, PyLocation *loc, PyInsertionPoint *ip);

// aten::cat : (Tensor[], int) -> (Tensor)
PyAnyTorchTensorValue cat(const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::ceil.float : (float) -> (int)
PyTorch_IntValue ceil(const PyTorch_FloatValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::ceil : (Tensor) -> (Tensor)
PyAnyTorchTensorValue ceil(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::ceil_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue ceil_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::clamp_max : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_max(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max, PyLocation *loc, PyInsertionPoint *ip);

// aten::clamp_max_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_max_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max, PyLocation *loc, PyInsertionPoint *ip);

// aten::clamp_min : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_min(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min, PyLocation *loc, PyInsertionPoint *ip);

// aten::clamp_min_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_min_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min, PyLocation *loc, PyInsertionPoint *ip);

// aten::clamp : (Tensor, Scalar?, Scalar?) -> (Tensor)
PyAnyTorchTensorValue clamp(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max, PyLocation *loc, PyInsertionPoint *ip);

// aten::clamp.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
PyAnyTorchTensorValue clamp(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max, PyLocation *loc, PyInsertionPoint *ip);

// aten::clamp_ : (Tensor, Scalar?, Scalar?) -> (Tensor)
PyAnyTorchTensorValue clamp_(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max, PyLocation *loc, PyInsertionPoint *ip);

// aten::clamp_.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
PyAnyTorchTensorValue clamp_(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max, PyLocation *loc, PyInsertionPoint *ip);

// aten::clone : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue clone(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)
PyAnyTorchTensorValue constant_pad_nd(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::contiguous : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue contiguous(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)
PyAnyTorchTensorValue conv2d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_IntValue &groups, PyLocation *loc, PyInsertionPoint *ip);

// aten::conv_transpose1d : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose1d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, PyLocation *loc, PyInsertionPoint *ip);

// aten::conv_transpose2d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose2d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, PyLocation *loc, PyInsertionPoint *ip);

// aten::conv_transpose3d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose3d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, PyLocation *loc, PyInsertionPoint *ip);

// aten::convolution_backward : (Tensor, Tensor, Tensor, int[]?, int[], int[], int[], bool, int[], int, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> convolution_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalListOfTorchIntValue &bias_sizes, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchBoolValue &output_mask, PyLocation *loc, PyInsertionPoint *ip);

// aten::convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)
PyAnyTorchTensorValue convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, PyLocation *loc, PyInsertionPoint *ip);

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue copy(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking, PyLocation *loc, PyInsertionPoint *ip);

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue copy_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking, PyLocation *loc, PyInsertionPoint *ip);

// aten::cos : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cos(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::cos_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cos_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::cpu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cpu(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::cross_entropy_loss : (Tensor, Tensor, Tensor?, int, int, float) -> (Tensor)
PyAnyTorchTensorValue cross_entropy_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyTorch_FloatValue &label_smoothing, PyLocation *loc, PyInsertionPoint *ip);

// aten::cuda : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cuda(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue cumsum(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
void Delete(const PyTorch_DictValue &self, const PyTorch_StringValue &key, PyLocation *loc, PyInsertionPoint *ip);

// aten::detach_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue detach_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::detach : (Tensor) -> (Tensor)
PyAnyTorchTensorValue detach(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue diagonal_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2, PyLocation *loc, PyInsertionPoint *ip);

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue diagonal_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2, PyLocation *loc, PyInsertionPoint *ip);

// aten::dim : (Tensor) -> (int)
PyTorch_IntValue dim(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::div.float : (float, float) -> (float)
PyTorch_FloatValue div(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::div.int : (int, int) -> (float)
PyTorch_FloatValue div(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::div : (Scalar, Scalar) -> (float)
PyTorch_FloatValue div(const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode, PyLocation *loc, PyInsertionPoint *ip);

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::div_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode, PyLocation *loc, PyInsertionPoint *ip);

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::dropout : (Tensor, float, bool) -> (Tensor)
PyAnyTorchTensorValue dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train, PyLocation *loc, PyInsertionPoint *ip);

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
PyAnyTorchTensorValue dropout_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train, PyLocation *loc, PyInsertionPoint *ip);

// aten::embedding_bag.padding_idx : (Tensor, Tensor, Tensor, bool, int, bool, Tensor?, bool, int?) -> (Tensor, Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> embedding_bag(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyAnyTorchTensorValue &offsets, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_IntValue &mode, const PyTorch_BoolValue &sparse, const PyAnyTorchOptionalTensorValue &per_sample_weights, const PyTorch_BoolValue &include_last_offset, const PyAnyTorchOptionalIntValue &padding_idx, PyLocation *loc, PyInsertionPoint *ip);

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
PyAnyTorchTensorValue embedding_dense_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, PyLocation *loc, PyInsertionPoint *ip);

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
PyAnyTorchTensorValue embedding(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse, PyLocation *loc, PyInsertionPoint *ip);

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue empty_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue empty(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq.device : (Device, Device) -> (bool)
PyTorch_BoolValue eq(const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq.float : (float, float) -> (bool)
PyTorch_BoolValue eq(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq.int_list : (int[], int[]) -> (bool)
PyTorch_BoolValue eq(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq.int : (int, int) -> (bool)
PyTorch_BoolValue eq(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue eq(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq.str : (str, str) -> (bool)
PyTorch_BoolValue eq(const PyTorch_StringValue &a, const PyTorch_StringValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue eq(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::erf : (Tensor) -> (Tensor)
PyAnyTorchTensorValue erf(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::erf_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue erf_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::exp : (Tensor) -> (Tensor)
PyAnyTorchTensorValue exp(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::exp_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue exp_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue expand_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue expand_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit, PyLocation *loc, PyInsertionPoint *ip);

// aten::expand : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue expand(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit, PyLocation *loc, PyInsertionPoint *ip);

// aten::expm1 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue expm1(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::expm1_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue expm1_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
PyAnyTorchTensorValue fft_fft(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm, PyLocation *loc, PyInsertionPoint *ip);

// aten::fill.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fill(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue flatten(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::flip : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue flip(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, PyLocation *loc, PyInsertionPoint *ip);

// aten::FloatImplicit : (Tensor) -> (float)
PyTorch_FloatValue FloatImplicit(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::Float.Scalar : (Scalar) -> (float)
PyTorch_FloatValue Float(const PyAnyTorchScalarValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::Float.str : (str) -> (float)
PyTorch_FloatValue Float(const PyTorch_StringValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::Float.Tensor : (Tensor) -> (float)
PyTorch_FloatValue Float(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::floor : (Tensor) -> (Tensor)
PyAnyTorchTensorValue floor(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::floor_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue floor_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::floordiv.int : (int, int) -> (int)
PyTorch_IntValue floordiv(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fmod(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::fmod_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fmod_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue frobenius_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::full_like : (Tensor, Scalar, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue full_like(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::full : (int[], Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue full(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue gather(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad, PyLocation *loc, PyInsertionPoint *ip);

// aten::ge.float_int : (float, int) -> (bool)
PyTorch_BoolValue ge(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::ge.float : (float, float) -> (bool)
PyTorch_BoolValue ge(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::ge.int : (int, int) -> (bool)
PyTorch_BoolValue ge(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ge(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ge(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::ge_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
PyAnyTorchTensorValue gelu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate, PyLocation *loc, PyInsertionPoint *ip);

// aten::gelu : (Tensor, str) -> (Tensor)
PyAnyTorchTensorValue gelu(const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate, PyLocation *loc, PyInsertionPoint *ip);

// aten::gt.float_int : (float, int) -> (bool)
PyTorch_BoolValue gt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::gt.float : (float, float) -> (bool)
PyTorch_BoolValue gt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::gt.int : (int, int) -> (bool)
PyTorch_BoolValue gt(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue gt(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue gt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::gt_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::hardsigmoid : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardsigmoid(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardsigmoid_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::hardswish : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardswish(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::hardswish_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardswish_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::hardtanh_backward : (Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, PyLocation *loc, PyInsertionPoint *ip);

// aten::hardtanh : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, PyLocation *loc, PyInsertionPoint *ip);

// aten::hardtanh_ : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, PyLocation *loc, PyInsertionPoint *ip);

// aten::imag : (Tensor) -> (Tensor)
PyAnyTorchTensorValue imag(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::index_put.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, PyLocation *loc, PyInsertionPoint *ip);

// aten::index_put : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, PyLocation *loc, PyInsertionPoint *ip);

// aten::index_put_.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, PyLocation *loc, PyInsertionPoint *ip);

// aten::index_put_ : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, PyLocation *loc, PyInsertionPoint *ip);

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue index_select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, PyLocation *loc, PyInsertionPoint *ip);

// aten::index.Tensor_hacked_twin : (Tensor, Tensor[]) -> (Tensor)
PyAnyTorchTensorValue index(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, PyLocation *loc, PyInsertionPoint *ip);

// aten::index.Tensor : (Tensor, Tensor?[]) -> (Tensor)
PyAnyTorchTensorValue index(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, PyLocation *loc, PyInsertionPoint *ip);

// aten::Int.bool : (bool) -> (int)
PyTorch_IntValue Int(const PyTorch_BoolValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::Int.float : (float) -> (int)
PyTorch_IntValue Int(const PyTorch_FloatValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::IntImplicit : (Tensor) -> (int)
PyTorch_IntValue IntImplicit(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::Int.Scalar : (Scalar) -> (int)
PyTorch_IntValue Int(const PyAnyTorchScalarValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::Int.Tensor : (Tensor) -> (int)
PyTorch_IntValue Int(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::is_floating_point : (Tensor) -> (bool)
PyTorch_BoolValue is_floating_point(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::isnan : (Tensor) -> (Tensor)
PyAnyTorchTensorValue isnan(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::join : (str, str[]) -> (str)
PyTorch_StringValue join(const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values, PyLocation *loc, PyInsertionPoint *ip);

// aten::keys.str : (Dict(str, t)) -> (str[])
PyAnyTorchListOfTorchStringValue keys(const PyTorch_DictValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)
PyAnyTorchTensorValue layer_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enable, PyLocation *loc, PyInsertionPoint *ip);

// aten::le.int : (int, int) -> (bool)
PyTorch_BoolValue le(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue le(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue le(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::le_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue le_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue le_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::leaky_relu_backward : (Tensor, Tensor, Scalar, bool) -> (Tensor)
PyAnyTorchTensorValue leaky_relu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, const PyTorch_BoolValue &self_is_result, PyLocation *loc, PyInsertionPoint *ip);

// aten::leaky_relu : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue leaky_relu(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, PyLocation *loc, PyInsertionPoint *ip);

// aten::leaky_relu_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue leaky_relu_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, PyLocation *loc, PyInsertionPoint *ip);

// aten::len.str : (str) -> (int)
PyTorch_IntValue len(const PyTorch_StringValue &s, PyLocation *loc, PyInsertionPoint *ip);

// aten::len.t : (t[]) -> (int)
PyTorch_IntValue len(const PyAnyTorchListValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::len.Tensor : (Tensor) -> (int)
PyTorch_IntValue len(const PyAnyTorchTensorValue &t, PyLocation *loc, PyInsertionPoint *ip);

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lerp(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight, PyLocation *loc, PyInsertionPoint *ip);

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lerp_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight, PyLocation *loc, PyInsertionPoint *ip);

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue lift_fresh_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::linalg_vector_norm : (Tensor, Scalar, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue linalg_vector_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &ord, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, PyLocation *loc, PyInsertionPoint *ip);

// aten::list.t : (t[]) -> (t[])
PyAnyTorchListValue list(const PyAnyTorchListValue &l, PyLocation *loc, PyInsertionPoint *ip);

// aten::log1p : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log1p(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::log1p_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log1p_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::log2 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log2(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::log2_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log2_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::log.int : (int) -> (float)
PyTorch_FloatValue log(const PyTorch_IntValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::log : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::log_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::logical_not : (Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_not(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::logical_not_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_not_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue logsumexp(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::lt.float_int : (float, int) -> (bool)
PyTorch_BoolValue lt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::lt.float : (float, float) -> (bool)
PyTorch_BoolValue lt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::lt.int : (int, int) -> (bool)
PyTorch_BoolValue lt(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue lt(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::lt_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::masked_fill_.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_select(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, PyLocation *loc, PyInsertionPoint *ip);

// aten::matmul : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue matmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::max.dim : (Tensor, int, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> max(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::max : (Tensor) -> (Tensor)
PyAnyTorchTensorValue max(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
//PyAnyTorchTensorValue max_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, PyLocation *loc, PyInsertionPoint *ip);

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
PyAnyTorchTensorValue max_pool2d_with_indices_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices, PyLocation *loc, PyInsertionPoint *ip);

// aten::max_pool2d_with_indices : (Tensor, int[], int[], int[], int[], bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> max_pool2d_with_indices(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, PyLocation *loc, PyInsertionPoint *ip);

// aten::maximum : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue maximum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::mean.dim : (Tensor, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::mean : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::minimum : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue minimum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::mish : (Tensor) -> (Tensor)
PyAnyTorchTensorValue mish(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::mm : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2, PyLocation *loc, PyInsertionPoint *ip);

// aten::movedim.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue movedim(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &source, const PyTorch_IntValue &destination, PyLocation *loc, PyInsertionPoint *ip);

// aten::mse_loss_backward : (Tensor, Tensor, Tensor, int) -> (Tensor)
PyAnyTorchTensorValue mse_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction, PyLocation *loc, PyInsertionPoint *ip);

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
PyAnyTorchTensorValue mse_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction, PyLocation *loc, PyInsertionPoint *ip);

// aten::mul.float : (float, float) -> (float)
PyTorch_FloatValue mul(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::mul.int : (int, int) -> (int)
PyTorch_IntValue mul(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue mul(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::mul_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::mv : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec, PyLocation *loc, PyInsertionPoint *ip);

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue narrow(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length, PyLocation *loc, PyInsertionPoint *ip);

// aten::native_batch_norm_backward : (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_batch_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyAnyTorchOptionalTensorValue &save_mean, const PyAnyTorchOptionalTensorValue &save_invstd, const PyTorch_BoolValue &train, const PyTorch_FloatValue &eps, const PyAnyTorchListOfTorchBoolValue &output_mask, PyLocation *loc, PyInsertionPoint *ip);

// aten::native_batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_batch_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, PyLocation *loc, PyInsertionPoint *ip);

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
PyAnyTorchTensorValue native_dropout_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale, PyLocation *loc, PyInsertionPoint *ip);

// aten::native_dropout : (Tensor, float, bool?) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyAnyTorchOptionalBoolValue &train, PyLocation *loc, PyInsertionPoint *ip);

// aten::native_group_norm_backward : (Tensor, Tensor, Tensor, Tensor, Tensor?, int, int, int, int, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_group_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &mean, const PyAnyTorchTensorValue &rstd, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &N, const PyTorch_IntValue &C, const PyTorch_IntValue &HxW, const PyTorch_IntValue &group, const PyAnyTorchListOfTorchBoolValue &output_mask, PyLocation *loc, PyInsertionPoint *ip);

// aten::native_group_norm : (Tensor, Tensor?, Tensor?, int, int, int, int, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_group_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_IntValue &N, const PyTorch_IntValue &C, const PyTorch_IntValue &HxW, const PyTorch_IntValue &group, const PyTorch_FloatValue &eps, PyLocation *loc, PyInsertionPoint *ip);

// aten::native_layer_norm_backward : (Tensor, Tensor, int[], Tensor, Tensor, Tensor?, Tensor?, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_layer_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchTensorValue &mean, const PyAnyTorchTensorValue &rstd, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchBoolValue &output_mask, PyLocation *loc, PyInsertionPoint *ip);

// aten::native_layer_norm : (Tensor, int[], Tensor?, Tensor?, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_layer_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, PyLocation *loc, PyInsertionPoint *ip);

// aten::ne.bool : (bool, bool) -> (bool)
PyTorch_BoolValue ne(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::ne.float_int : (float, int) -> (bool)
PyTorch_BoolValue ne(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::ne.int_list : (int[], int[]) -> (bool)
PyTorch_BoolValue ne(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::ne.int : (int, int) -> (bool)
PyTorch_BoolValue ne(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ne(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ne(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::ne_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::neg.float : (float) -> (float)
PyTorch_FloatValue neg(const PyTorch_FloatValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::neg.int : (int) -> (int)
PyTorch_IntValue neg(const PyTorch_IntValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::neg : (Tensor) -> (Tensor)
PyAnyTorchTensorValue neg(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::neg_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue neg_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_empty(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_empty_strided(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_ones(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_zeros(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::nll_loss2d_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue nll_loss2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight, PyLocation *loc, PyInsertionPoint *ip);

// aten::nll_loss2d_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> nll_loss2d_forward(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, PyLocation *loc, PyInsertionPoint *ip);

// aten::nll_loss_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue nll_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight, PyLocation *loc, PyInsertionPoint *ip);

// aten::nll_loss_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> nll_loss_forward(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, PyLocation *loc, PyInsertionPoint *ip);

// aten::norm.ScalarOpt_dim : (Tensor, Scalar?, int[], bool) -> (Tensor)
PyAnyTorchTensorValue norm(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &p, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::numel : (Tensor) -> (int)
PyTorch_IntValue numel(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::numpy_T : (Tensor) -> (Tensor)
PyAnyTorchTensorValue numpy_T(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::one_hot : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue one_hot(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &num_classes, PyLocation *loc, PyInsertionPoint *ip);

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue ones_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue ones(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
PyAnyTorchTensorValue pad(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue permute_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, PyLocation *loc, PyInsertionPoint *ip);

// aten::permute : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue permute(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, PyLocation *loc, PyInsertionPoint *ip);

// aten::pow.int_float : (int, float) -> (float)
PyTorch_FloatValue pow(const PyTorch_IntValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::pow.Scalar : (Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &exponent, PyLocation *loc, PyInsertionPoint *ip);

// aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &exponent, PyLocation *loc, PyInsertionPoint *ip);

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent, PyLocation *loc, PyInsertionPoint *ip);

// aten::prelu : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue prelu(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight, PyLocation *loc, PyInsertionPoint *ip);

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue rand_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randint(const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::randint : (int, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randint(const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue randn_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::real : (Tensor) -> (Tensor)
PyAnyTorchTensorValue real(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::reciprocal : (Tensor) -> (Tensor)
PyAnyTorchTensorValue reciprocal(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::reciprocal_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue reciprocal_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::relu6 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu6(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::relu6_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu6_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::relu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::relu_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::remainder.int : (int, int) -> (int)
PyTorch_IntValue remainder(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::remainder.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue remainder(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::repeat : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue repeat(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats, PyLocation *loc, PyInsertionPoint *ip);

// aten::reshape : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue reshape(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape, PyLocation *loc, PyInsertionPoint *ip);

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
PyAnyTorchTensorValue resize_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue roll(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims, PyLocation *loc, PyInsertionPoint *ip);

// aten::round : (Tensor) -> (Tensor)
PyAnyTorchTensorValue round(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::round_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue round_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::rsqrt : (Tensor) -> (Tensor)
PyAnyTorchTensorValue rsqrt(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::rsqrt_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue rsqrt_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue rsub(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::scalar_tensor : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue scalar_tensor(const PyAnyTorchScalarValue &s, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::scaled_dot_product_attention : (Tensor, Tensor, Tensor, Tensor?, float, bool, float?) -> (Tensor)
PyAnyTorchTensorValue scaled_dot_product_attention(const PyAnyTorchTensorValue &query, const PyAnyTorchTensorValue &key, const PyAnyTorchTensorValue &value, const PyAnyTorchOptionalTensorValue &attn_mask, const PyTorch_FloatValue &dropout_p, const PyTorch_BoolValue &is_causal, const PyAnyTorchOptionalFloatValue &scale, PyLocation *loc, PyInsertionPoint *ip);

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_add(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, PyLocation *loc, PyInsertionPoint *ip);

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_add_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, PyLocation *loc, PyInsertionPoint *ip);

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
PyAnyTorchTensorValue scatter_reduce(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self, PyLocation *loc, PyInsertionPoint *ip);

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
PyAnyTorchTensorValue scatter_reduce_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self, PyLocation *loc, PyInsertionPoint *ip);

// aten::scatter.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, PyLocation *loc, PyInsertionPoint *ip);

// aten::scatter.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::scatter_.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, PyLocation *loc, PyInsertionPoint *ip);

// aten::scatter_.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue scatter_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, PyLocation *loc, PyInsertionPoint *ip);

// aten::select.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, PyLocation *loc, PyInsertionPoint *ip);

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, PyLocation *loc, PyInsertionPoint *ip);

// aten::sigmoid : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sigmoid(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::sigmoid_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sigmoid_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::sign : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sign(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::sign_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sign_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::silu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue silu(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::silu_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue silu_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::sin : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sin(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::sin_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sin_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::size.int : (Tensor, int) -> (int)
PyTorch_IntValue size(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::size : (Tensor) -> (int[])
PyAnyTorchListOfTorchIntValue size(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip);

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip);

// aten::slice.t : (t[], int?, int?, int) -> (t[])
PyAnyTorchListValue slice(const PyAnyTorchListValue &l, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip);

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip);

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::sort.int : (int[], bool) -> ()
void sort(const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse, PyLocation *loc, PyInsertionPoint *ip);

// aten::sort : (Tensor, int, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> sort(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &descending, PyLocation *loc, PyInsertionPoint *ip);

// aten::split.Tensor : (Tensor, int, int) -> (Tensor[])
PyAnyTorchListOfTensorValue split(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &split_size, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::sqrt.int : (int) -> (float)
PyTorch_FloatValue sqrt(const PyTorch_IntValue &a, PyLocation *loc, PyInsertionPoint *ip);

// aten::sqrt : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sqrt(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::sqrt_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sqrt_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::square : (Tensor) -> (Tensor)
PyAnyTorchTensorValue square(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::square_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue square_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue squeeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::squeeze_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue squeeze_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::squeeze : (Tensor) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::stack : (Tensor[], int) -> (Tensor)
PyAnyTorchTensorValue stack(const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::std.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::std.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::std : (Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, PyLocation *loc, PyInsertionPoint *ip);

// aten::sub.float : (float, float) -> (float)
PyTorch_FloatValue sub(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::sub.int : (int, int) -> (int)
PyTorch_IntValue sub(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::sub_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::sub_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip);

// aten::sum.dim_IntList : (Tensor, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::sum : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::t_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue t_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::t : (Tensor) -> (Tensor)
PyAnyTorchTensorValue t(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, PyLocation *loc, PyInsertionPoint *ip);

// aten::tanh : (Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::tanh_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, PyLocation *loc, PyInsertionPoint *ip);

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, PyLocation *loc, PyInsertionPoint *ip);

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, PyLocation *loc, PyInsertionPoint *ip);

// aten::tensor : (t[], int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyAnyTorchListValue &data, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, PyLocation *loc, PyInsertionPoint *ip);

// aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, PyLocation *loc, PyInsertionPoint *ip);

// aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::threshold_ : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip);

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, PyLocation *loc, PyInsertionPoint *ip);

// aten::topk : (Tensor, int, int, bool, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> topk(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &k, const PyTorch_IntValue &dim, const PyTorch_BoolValue &largest, const PyTorch_BoolValue &sorted, PyLocation *loc, PyInsertionPoint *ip);

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue transpose_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1, PyLocation *loc, PyInsertionPoint *ip);

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue transpose(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1, PyLocation *loc, PyInsertionPoint *ip);

// aten::tril : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue tril(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, PyLocation *loc, PyInsertionPoint *ip);

// aten::tril_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue tril_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, PyLocation *loc, PyInsertionPoint *ip);

// aten::triu : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue triu(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, PyLocation *loc, PyInsertionPoint *ip);

// aten::triu_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue triu_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, PyLocation *loc, PyInsertionPoint *ip);

// aten::type_as : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue type_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::unbind.int : (Tensor, int) -> (Tensor[])
PyAnyTorchListOfTensorValue unbind(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue unfold_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip);

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue uniform(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip);

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue uniform_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip);

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::unsqueeze : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip);

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
PyAnyTorchTensorValue upsample_nearest2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w, PyLocation *loc, PyInsertionPoint *ip);

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
PyAnyTorchTensorValue upsample_nearest2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w, PyLocation *loc, PyInsertionPoint *ip);

// aten::var.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::var.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::var_mean.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::var_mean.dim : (Tensor, int[]?, bool, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip);

// aten::var_mean : (Tensor, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, PyLocation *loc, PyInsertionPoint *ip);

// aten::var : (Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, PyLocation *loc, PyInsertionPoint *ip);

// aten::view_as_complex : (Tensor) -> (Tensor)
PyAnyTorchTensorValue view_as_complex(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue view_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::view_copy : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue view_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, PyLocation *loc, PyInsertionPoint *ip);

// aten::view : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, PyLocation *loc, PyInsertionPoint *ip);

// aten::where.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::where.ScalarOther : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::where.ScalarSelf : (Tensor, Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::zero : (Tensor) -> (Tensor)
PyAnyTorchTensorValue zero(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::zero_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue zero_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue zeros_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue zeros(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip);

// aten::_convolution.deprecated : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, PyLocation *loc, PyInsertionPoint *ip);

// aten::_convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, const PyTorch_BoolValue &allow_tf32, PyLocation *loc, PyInsertionPoint *ip);

// aten::_embedding_bag : (Tensor, Tensor, Tensor, bool, int, bool, Tensor?, bool, int) -> (Tensor, Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> _embedding_bag(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyAnyTorchTensorValue &offsets, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_IntValue &mode, const PyTorch_BoolValue &sparse, const PyAnyTorchOptionalTensorValue &per_sample_weights, const PyTorch_BoolValue &include_last_offset, const PyTorch_IntValue &padding_idx, PyLocation *loc, PyInsertionPoint *ip);

// aten::_index_put_impl : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _index_put_impl(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, const PyTorch_BoolValue &unsafe, PyLocation *loc, PyInsertionPoint *ip);

// aten::_index_put_impl_ : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _index_put_impl_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, const PyTorch_BoolValue &unsafe, PyLocation *loc, PyInsertionPoint *ip);

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue _log_softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue _log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float, PyLocation *loc, PyInsertionPoint *ip);

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue _reshape_alias_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, PyLocation *loc, PyInsertionPoint *ip);

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue _reshape_alias(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, PyLocation *loc, PyInsertionPoint *ip);

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
PyAnyTorchTensorValue _shape_as_tensor(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue _softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype, PyLocation *loc, PyInsertionPoint *ip);

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue _softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float, PyLocation *loc, PyInsertionPoint *ip);

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue _to_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip);

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue _unsafe_view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, PyLocation *loc, PyInsertionPoint *ip);

// aten::__and__.bool : (bool, bool) -> (bool)
PyTorch_BoolValue __and__(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b, PyLocation *loc, PyInsertionPoint *ip);

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __and__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip);

// aten::__contains__.int_list : (int[], int) -> (bool)
PyTorch_BoolValue __contains__(const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item, PyLocation *loc, PyInsertionPoint *ip);

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
PyTorch_BoolValue __contains__(const PyTorch_DictValue &dict, const PyTorch_StringValue &key, PyLocation *loc, PyInsertionPoint *ip);

// aten::__derive_index : (int, int, int) -> (int)
PyTorch_IntValue __derive_index(const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip);

// aten::__not__ : (bool) -> (bool)
PyTorch_BoolValue __not__(const PyTorch_BoolValue &self, PyLocation *loc, PyInsertionPoint *ip);

// aten::__range_length : (int, int, int) -> (int)
PyTorch_IntValue __range_length(const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip);

// prim::device : (Tensor) -> (Device)
PyTorch_DeviceValue device(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// prim::dtype : (Tensor) -> (int)
PyTorch_IntValue dtype(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// prim::layout : (Tensor) -> (int)
PyTorch_IntValue layout(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// prim::max.int : (int, int) -> (int)
PyTorch_IntValue max(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// prim::max.self_int : (int[]) -> (int)
PyTorch_IntValue max(const PyAnyTorchListOfTorchIntValue &self, PyLocation *loc, PyInsertionPoint *ip);

// prim::min.int : (int, int) -> (int)
PyTorch_IntValue min(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip);

// prim::min.self_int : (int[]) -> (int)
PyTorch_IntValue min(const PyAnyTorchListOfTorchIntValue &self, PyLocation *loc, PyInsertionPoint *ip);

// prim::NumToTensor.Scalar : (Scalar) -> (Tensor)
PyAnyTorchTensorValue NumToTensor(const PyAnyTorchScalarValue &a, PyLocation *loc, PyInsertionPoint *ip);

// prim::RaiseException : (str, str?) -> ()
void RaiseException(const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls, PyLocation *loc, PyInsertionPoint *ip);

// prims::convert_element_type : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue convert_element_type(const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype, PyLocation *loc, PyInsertionPoint *ip);

// prims::squeeze : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &a, const PyAnyTorchListOfTorchIntValue &dimensions, PyLocation *loc, PyInsertionPoint *ip);

// prims::var : (Tensor, int[]?, float, int?) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &inp, const PyAnyTorchOptionalListOfTorchIntValue &dims, const PyTorch_FloatValue &correction, const PyAnyTorchOptionalIntValue &output_dtype, PyLocation *loc, PyInsertionPoint *ip);

// prims::view_of : (Tensor) -> (Tensor)
PyAnyTorchTensorValue view_of(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip);

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i, PyLocation *loc, PyInsertionPoint *ip);
