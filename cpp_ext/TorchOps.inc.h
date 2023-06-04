
// aten::abs : (Tensor) -> (Tensor)
PyAnyTorchTensorValue abs(const PyAnyTorchTensorValue &self);

// aten::abs_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue abs_(const PyAnyTorchTensorValue &self);

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue adaptive_avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size);

// aten::add.float_int : (float, int) -> (float)
PyTorch_FloatValue add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::add.int : (int, int) -> (int)
PyTorch_IntValue add(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue add(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha);

// aten::add.str : (str, str) -> (str)
PyTorch_StringValue add(const PyTorch_StringValue &a, const PyTorch_StringValue &b);

// aten::add.t : (t[], t[]) -> (t[])
PyAnyTorchListValue add(const PyAnyTorchListValue &a, const PyAnyTorchListValue &b);

// aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue add(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha);

// aten::add_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue add_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha);

// aten::add_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue add_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha);

// aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcdiv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value);

// aten::addcdiv_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcdiv_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value);

// aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value);

// aten::addcmul_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcmul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value);

// aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue addmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat1, const PyAnyTorchTensorValue &mat2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha);

// aten::alias_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue alias_copy(const PyAnyTorchTensorValue &self);

// aten::all.bool : (bool[]) -> (bool)
PyTorch_BoolValue all(const PyAnyTorchListOfTorchBoolValue &self);

// aten::all : (Tensor) -> (Tensor)
PyAnyTorchTensorValue all(const PyAnyTorchTensorValue &self);

// aten::amax : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue amax(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::any.bool : (bool[]) -> (bool)
PyTorch_BoolValue any(const PyAnyTorchListOfTorchBoolValue &self);

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue any(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::any : (Tensor) -> (Tensor)
PyAnyTorchTensorValue any(const PyAnyTorchTensorValue &self);

// aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::arange.start_out : (Scalar, Scalar, Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchTensorValue &out);

// aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
PyAnyTorchTensorValue argmax(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
PyAnyTorchTensorValue as_strided_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset);

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
PyAnyTorchTensorValue as_strided_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset);

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue atan2(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue atan2_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::atan : (Tensor) -> (Tensor)
PyAnyTorchTensorValue atan(const PyAnyTorchTensorValue &self);

// aten::atan_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue atan_(const PyAnyTorchTensorValue &self);

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override);

// aten::baddbmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue baddbmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha);

// aten::baddbmm_ : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue baddbmm_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha);

// aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)
PyAnyTorchTensorValue batch_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enabled);

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::bincount : (Tensor, Tensor?, int) -> (Tensor)
PyAnyTorchTensorValue bincount(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &weights, const PyTorch_IntValue &minlength);

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_not : (Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_not(const PyAnyTorchTensorValue &self);

// aten::bitwise_not_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_not_(const PyAnyTorchTensorValue &self);

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::bmm : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2);

// aten::Bool.float : (float) -> (bool)
PyTorch_BoolValue Bool(const PyTorch_FloatValue &a);

// aten::Bool.int : (int) -> (bool)
PyTorch_BoolValue Bool(const PyTorch_IntValue &a);

// aten::Bool.Tensor : (Tensor) -> (bool)
PyTorch_BoolValue Bool(const PyAnyTorchTensorValue &a);

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue broadcast_to(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size);

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue bucketize(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right);

// aten::cat : (Tensor[], int) -> (Tensor)
PyAnyTorchTensorValue cat(const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim);

// aten::ceil.float : (float) -> (int)
PyTorch_IntValue ceil(const PyTorch_FloatValue &a);

// aten::ceil : (Tensor) -> (Tensor)
PyAnyTorchTensorValue ceil(const PyAnyTorchTensorValue &self);

// aten::ceil_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue ceil_(const PyAnyTorchTensorValue &self);

// aten::clamp_max : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_max(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max);

// aten::clamp_max_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_max_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max);

// aten::clamp_min : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_min(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min);

// aten::clamp_min_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_min_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min);

// aten::clamp : (Tensor, Scalar?, Scalar?) -> (Tensor)
PyAnyTorchTensorValue clamp(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max);

// aten::clamp.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
PyAnyTorchTensorValue clamp(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max);

// aten::clamp_ : (Tensor, Scalar?, Scalar?) -> (Tensor)
PyAnyTorchTensorValue clamp_(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max);

// aten::clamp_.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
PyAnyTorchTensorValue clamp_(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max);

// aten::clone : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue clone(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format);

// aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)
PyAnyTorchTensorValue constant_pad_nd(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyAnyTorchScalarValue &value);

// aten::contiguous : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue contiguous(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format);

// aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)
PyAnyTorchTensorValue conv2d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_IntValue &groups);

// aten::conv_transpose1d : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose1d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation);

// aten::conv_transpose2d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose2d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation);

// aten::conv_transpose3d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose3d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation);

// aten::convolution_backward : (Tensor, Tensor, Tensor, int[]?, int[], int[], int[], bool, int[], int, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> convolution_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalListOfTorchIntValue &bias_sizes, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchBoolValue &output_mask);

// aten::convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)
PyAnyTorchTensorValue convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups);

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue copy(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking);

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue copy_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking);

// aten::cos : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cos(const PyAnyTorchTensorValue &self);

// aten::cos_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cos_(const PyAnyTorchTensorValue &self);

// aten::cpu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cpu(const PyAnyTorchTensorValue &self);

// aten::cross_entropy_loss : (Tensor, Tensor, Tensor?, int, int, float) -> (Tensor)
PyAnyTorchTensorValue cross_entropy_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyTorch_FloatValue &label_smoothing);

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue cumsum(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype);

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
void Delete(const PyTorch_DictValue &self, const PyTorch_StringValue &key);

// aten::detach_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue detach_copy(const PyAnyTorchTensorValue &self);

// aten::detach : (Tensor) -> (Tensor)
PyAnyTorchTensorValue detach(const PyAnyTorchTensorValue &self);

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue diagonal_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2);

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue diagonal_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2);

// aten::dim : (Tensor) -> (int)
PyTorch_IntValue dim(const PyAnyTorchTensorValue &self);

// aten::div.float : (float, float) -> (float)
PyTorch_FloatValue div(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::div.int : (int, int) -> (float)
PyTorch_FloatValue div(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::div : (Scalar, Scalar) -> (float)
PyTorch_FloatValue div(const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b);

// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode);

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::div_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode);

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::dropout : (Tensor, float, bool) -> (Tensor)
PyAnyTorchTensorValue dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train);

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
PyAnyTorchTensorValue dropout_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train);

// aten::embedding_bag.padding_idx : (Tensor, Tensor, Tensor, bool, int, bool, Tensor?, bool, int?) -> (Tensor, Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> embedding_bag(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyAnyTorchTensorValue &offsets, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_IntValue &mode, const PyTorch_BoolValue &sparse, const PyAnyTorchOptionalTensorValue &per_sample_weights, const PyTorch_BoolValue &include_last_offset, const PyAnyTorchOptionalIntValue &padding_idx);

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
PyAnyTorchTensorValue embedding_dense_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq);

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
PyAnyTorchTensorValue embedding(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse);

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue empty_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue empty(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::eq.device : (Device, Device) -> (bool)
PyTorch_BoolValue eq(const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b);

// aten::eq.float : (float, float) -> (bool)
PyTorch_BoolValue eq(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::eq.int_list : (int[], int[]) -> (bool)
PyTorch_BoolValue eq(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b);

// aten::eq.int : (int, int) -> (bool)
PyTorch_BoolValue eq(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue eq(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::eq.str : (str, str) -> (bool)
PyTorch_BoolValue eq(const PyTorch_StringValue &a, const PyTorch_StringValue &b);

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue eq(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::eq_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::erf : (Tensor) -> (Tensor)
PyAnyTorchTensorValue erf(const PyAnyTorchTensorValue &self);

// aten::erf_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue erf_(const PyAnyTorchTensorValue &self);

// aten::exp : (Tensor) -> (Tensor)
PyAnyTorchTensorValue exp(const PyAnyTorchTensorValue &self);

// aten::exp_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue exp_(const PyAnyTorchTensorValue &self);

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue expand_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue expand_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit);

// aten::expand : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue expand(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit);

// aten::expm1 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue expm1(const PyAnyTorchTensorValue &self);

// aten::expm1_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue expm1_(const PyAnyTorchTensorValue &self);

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
PyAnyTorchTensorValue fft_fft(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm);

// aten::fill.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fill(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value);

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value);

// aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value);

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value);

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue flatten(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim);

// aten::flip : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue flip(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims);

// aten::FloatImplicit : (Tensor) -> (float)
PyTorch_FloatValue FloatImplicit(const PyAnyTorchTensorValue &a);

// aten::Float.Scalar : (Scalar) -> (float)
PyTorch_FloatValue Float(const PyAnyTorchScalarValue &a);

// aten::Float.str : (str) -> (float)
PyTorch_FloatValue Float(const PyTorch_StringValue &a);

// aten::Float.Tensor : (Tensor) -> (float)
PyTorch_FloatValue Float(const PyAnyTorchTensorValue &a);

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::floor : (Tensor) -> (Tensor)
PyAnyTorchTensorValue floor(const PyAnyTorchTensorValue &self);

// aten::floor_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue floor_(const PyAnyTorchTensorValue &self);

// aten::floordiv.int : (int, int) -> (int)
PyTorch_IntValue floordiv(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fmod(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::fmod_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fmod_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue frobenius_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::full_like : (Tensor, Scalar, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue full_like(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::full : (int[], Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue full(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue gather(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad);

// aten::ge.float_int : (float, int) -> (bool)
PyTorch_BoolValue ge(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::ge.float : (float, float) -> (bool)
PyTorch_BoolValue ge(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::ge.int : (int, int) -> (bool)
PyTorch_BoolValue ge(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ge(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ge(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::ge_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
PyAnyTorchTensorValue gelu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate);

// aten::gelu : (Tensor, str) -> (Tensor)
PyAnyTorchTensorValue gelu(const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate);

// aten::gt.float_int : (float, int) -> (bool)
PyTorch_BoolValue gt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::gt.float : (float, float) -> (bool)
PyTorch_BoolValue gt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::gt.int : (int, int) -> (bool)
PyTorch_BoolValue gt(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue gt(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue gt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::gt_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::hardsigmoid : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardsigmoid(const PyAnyTorchTensorValue &self);

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardsigmoid_(const PyAnyTorchTensorValue &self);

// aten::hardswish : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardswish(const PyAnyTorchTensorValue &self);

// aten::hardswish_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardswish_(const PyAnyTorchTensorValue &self);

// aten::hardtanh_backward : (Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val);

// aten::hardtanh : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val);

// aten::hardtanh_ : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val);

// aten::imag : (Tensor) -> (Tensor)
PyAnyTorchTensorValue imag(const PyAnyTorchTensorValue &self);

// aten::index_put.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate);

// aten::index_put : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate);

// aten::index_put_.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate);

// aten::index_put_ : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate);

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue index_select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index);

// aten::index.Tensor_hacked_twin : (Tensor, Tensor[]) -> (Tensor)
PyAnyTorchTensorValue index(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices);

// aten::index.Tensor : (Tensor, Tensor?[]) -> (Tensor)
PyAnyTorchTensorValue index(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices);

// aten::Int.bool : (bool) -> (int)
PyTorch_IntValue Int(const PyTorch_BoolValue &a);

// aten::Int.float : (float) -> (int)
PyTorch_IntValue Int(const PyTorch_FloatValue &a);

// aten::IntImplicit : (Tensor) -> (int)
PyTorch_IntValue IntImplicit(const PyAnyTorchTensorValue &a);

// aten::Int.Scalar : (Scalar) -> (int)
PyTorch_IntValue Int(const PyAnyTorchScalarValue &a);

// aten::Int.Tensor : (Tensor) -> (int)
PyTorch_IntValue Int(const PyAnyTorchTensorValue &a);

// aten::is_floating_point : (Tensor) -> (bool)
PyTorch_BoolValue is_floating_point(const PyAnyTorchTensorValue &self);

// aten::join : (str, str[]) -> (str)
PyTorch_StringValue join(const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values);

// aten::keys.str : (Dict(str, t)) -> (str[])
PyAnyTorchListOfTorchStringValue keys(const PyTorch_DictValue &self);

// aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)
PyAnyTorchTensorValue layer_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enable);

// aten::le.int : (int, int) -> (bool)
PyTorch_BoolValue le(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue le(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue le(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::le_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue le_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue le_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::leaky_relu_backward : (Tensor, Tensor, Scalar, bool) -> (Tensor)
PyAnyTorchTensorValue leaky_relu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, const PyTorch_BoolValue &self_is_result);

// aten::leaky_relu : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue leaky_relu(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope);

// aten::leaky_relu_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue leaky_relu_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope);

// aten::len.str : (str) -> (int)
PyTorch_IntValue len(const PyTorch_StringValue &s);

// aten::len.t : (t[]) -> (int)
PyTorch_IntValue len(const PyAnyTorchListValue &a);

// aten::len.Tensor : (Tensor) -> (int)
PyTorch_IntValue len(const PyAnyTorchTensorValue &t);

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lerp(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight);

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lerp_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight);

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue lift_fresh_copy(const PyAnyTorchTensorValue &self);

// aten::linalg_vector_norm : (Tensor, Scalar, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue linalg_vector_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &ord, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype);

// aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias);

// aten::list.t : (t[]) -> (t[])
PyAnyTorchListValue list(const PyAnyTorchListValue &l);

// aten::log1p : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log1p(const PyAnyTorchTensorValue &self);

// aten::log1p_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log1p_(const PyAnyTorchTensorValue &self);

// aten::log2 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log2(const PyAnyTorchTensorValue &self);

// aten::log2_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log2_(const PyAnyTorchTensorValue &self);

// aten::log.int : (int) -> (float)
PyTorch_FloatValue log(const PyTorch_IntValue &a);

// aten::log : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log(const PyAnyTorchTensorValue &self);

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype);

// aten::log_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log_(const PyAnyTorchTensorValue &self);

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_not : (Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_not(const PyAnyTorchTensorValue &self);

// aten::logical_not_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_not_(const PyAnyTorchTensorValue &self);

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue logsumexp(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::lt.float_int : (float, int) -> (bool)
PyTorch_BoolValue lt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::lt.float : (float, float) -> (bool)
PyTorch_BoolValue lt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::lt.int : (int, int) -> (bool)
PyTorch_BoolValue lt(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue lt(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::lt_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value);

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value);

// aten::masked_fill_.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value);

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value);

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_select(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask);

// aten::matmul : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue matmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::max.dim : (Tensor, int, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> max(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::max : (Tensor) -> (Tensor)
PyAnyTorchTensorValue max(const PyAnyTorchTensorValue &self);

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
PyAnyTorchTensorValue max_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode);

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
PyAnyTorchTensorValue max_pool2d_with_indices_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices);

// aten::max_pool2d_with_indices : (Tensor, int[], int[], int[], int[], bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> max_pool2d_with_indices(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode);

// aten::maximum : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue maximum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::mean.dim : (Tensor, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype);

// aten::mean : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype);

// aten::minimum : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue minimum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::mish : (Tensor) -> (Tensor)
PyAnyTorchTensorValue mish(const PyAnyTorchTensorValue &self);

// aten::mm : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2);

// aten::movedim.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue movedim(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &source, const PyTorch_IntValue &destination);

// aten::mse_loss_backward : (Tensor, Tensor, Tensor, int) -> (Tensor)
PyAnyTorchTensorValue mse_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction);

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
PyAnyTorchTensorValue mse_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction);

// aten::mul.float : (float, float) -> (float)
PyTorch_FloatValue mul(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::mul.int : (int, int) -> (int)
PyTorch_IntValue mul(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue mul(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::mul_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::mv : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec);

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue narrow(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length);

// aten::native_batch_norm_backward : (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_batch_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyAnyTorchOptionalTensorValue &save_mean, const PyAnyTorchOptionalTensorValue &save_invstd, const PyTorch_BoolValue &train, const PyTorch_FloatValue &eps, const PyAnyTorchListOfTorchBoolValue &output_mask);

// aten::native_batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_batch_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps);

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
PyAnyTorchTensorValue native_dropout_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale);

// aten::native_dropout : (Tensor, float, bool?) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyAnyTorchOptionalBoolValue &train);

// aten::native_group_norm_backward : (Tensor, Tensor, Tensor, Tensor, Tensor?, int, int, int, int, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_group_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &mean, const PyAnyTorchTensorValue &rstd, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &N, const PyTorch_IntValue &C, const PyTorch_IntValue &HxW, const PyTorch_IntValue &group, const PyAnyTorchListOfTorchBoolValue &output_mask);

// aten::native_group_norm : (Tensor, Tensor?, Tensor?, int, int, int, int, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_group_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_IntValue &N, const PyTorch_IntValue &C, const PyTorch_IntValue &HxW, const PyTorch_IntValue &group, const PyTorch_FloatValue &eps);

// aten::native_layer_norm_backward : (Tensor, Tensor, int[], Tensor, Tensor, Tensor?, Tensor?, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_layer_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchTensorValue &mean, const PyAnyTorchTensorValue &rstd, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchBoolValue &output_mask);

// aten::native_layer_norm : (Tensor, int[], Tensor?, Tensor?, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_layer_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps);

// aten::ne.bool : (bool, bool) -> (bool)
PyTorch_BoolValue ne(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b);

// aten::ne.float_int : (float, int) -> (bool)
PyTorch_BoolValue ne(const PyTorch_FloatValue &a, const PyTorch_IntValue &b);

// aten::ne.int_list : (int[], int[]) -> (bool)
PyTorch_BoolValue ne(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b);

// aten::ne.int : (int, int) -> (bool)
PyTorch_BoolValue ne(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ne(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ne(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::ne_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::neg.float : (float) -> (float)
PyTorch_FloatValue neg(const PyTorch_FloatValue &a);

// aten::neg.int : (int) -> (int)
PyTorch_IntValue neg(const PyTorch_IntValue &a);

// aten::neg : (Tensor) -> (Tensor)
PyAnyTorchTensorValue neg(const PyAnyTorchTensorValue &self);

// aten::neg_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue neg_(const PyAnyTorchTensorValue &self);

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_empty(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_empty_strided(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_ones(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_zeros(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::nll_loss2d_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue nll_loss2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight);

// aten::nll_loss2d_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> nll_loss2d_forward(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index);

// aten::nll_loss_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue nll_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight);

// aten::nll_loss_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> nll_loss_forward(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index);

// aten::norm.ScalarOpt_dim : (Tensor, Scalar?, int[], bool) -> (Tensor)
PyAnyTorchTensorValue norm(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &p, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim);

// aten::numel : (Tensor) -> (int)
PyTorch_IntValue numel(const PyAnyTorchTensorValue &self);

// aten::numpy_T : (Tensor) -> (Tensor)
PyAnyTorchTensorValue numpy_T(const PyAnyTorchTensorValue &self);

// aten::one_hot : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue one_hot(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &num_classes);

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue ones_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue ones(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
PyAnyTorchTensorValue pad(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value);

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue permute_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims);

// aten::permute : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue permute(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims);

// aten::pow.int_float : (int, float) -> (float)
PyTorch_FloatValue pow(const PyTorch_IntValue &a, const PyTorch_FloatValue &b);

// aten::pow.Scalar : (Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &exponent);

// aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &exponent);

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent);

// aten::prelu : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue prelu(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight);

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue rand_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randint(const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::randint : (int, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randint(const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue randn_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::real : (Tensor) -> (Tensor)
PyAnyTorchTensorValue real(const PyAnyTorchTensorValue &self);

// aten::reciprocal : (Tensor) -> (Tensor)
PyAnyTorchTensorValue reciprocal(const PyAnyTorchTensorValue &self);

// aten::reciprocal_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue reciprocal_(const PyAnyTorchTensorValue &self);

// aten::relu6 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu6(const PyAnyTorchTensorValue &self);

// aten::relu6_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu6_(const PyAnyTorchTensorValue &self);

// aten::relu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu(const PyAnyTorchTensorValue &self);

// aten::relu_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu_(const PyAnyTorchTensorValue &self);

// aten::remainder.int : (int, int) -> (int)
PyTorch_IntValue remainder(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::remainder.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue remainder(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::repeat : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue repeat(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats);

// aten::reshape : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue reshape(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape);

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
PyAnyTorchTensorValue resize_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format);

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue roll(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims);

// aten::round : (Tensor) -> (Tensor)
PyAnyTorchTensorValue round(const PyAnyTorchTensorValue &self);

// aten::round_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue round_(const PyAnyTorchTensorValue &self);

// aten::rsqrt : (Tensor) -> (Tensor)
PyAnyTorchTensorValue rsqrt(const PyAnyTorchTensorValue &self);

// aten::rsqrt_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue rsqrt_(const PyAnyTorchTensorValue &self);

// aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue rsub(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha);

// aten::scaled_dot_product_attention : (Tensor, Tensor, Tensor, Tensor?, float, bool, float?) -> (Tensor)
PyAnyTorchTensorValue scaled_dot_product_attention(const PyAnyTorchTensorValue &query, const PyAnyTorchTensorValue &key, const PyAnyTorchTensorValue &value, const PyAnyTorchOptionalTensorValue &attn_mask, const PyTorch_FloatValue &dropout_p, const PyTorch_BoolValue &is_causal, const PyAnyTorchOptionalFloatValue &scale);

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_add(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src);

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_add_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src);

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
PyAnyTorchTensorValue scatter_reduce(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self);

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
PyAnyTorchTensorValue scatter_reduce_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self);

// aten::scatter.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src);

// aten::scatter.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchScalarValue &value);

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index);

// aten::select.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index);

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index);

// aten::sigmoid : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sigmoid(const PyAnyTorchTensorValue &self);

// aten::sigmoid_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sigmoid_(const PyAnyTorchTensorValue &self);

// aten::silu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue silu(const PyAnyTorchTensorValue &self);

// aten::silu_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue silu_(const PyAnyTorchTensorValue &self);

// aten::sin : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sin(const PyAnyTorchTensorValue &self);

// aten::sin_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sin_(const PyAnyTorchTensorValue &self);

// aten::size.int : (Tensor, int) -> (int)
PyTorch_IntValue size(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::size : (Tensor) -> (int[])
PyAnyTorchListOfTorchIntValue size(const PyAnyTorchTensorValue &self);

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step);

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step);

// aten::slice.t : (t[], int?, int?, int) -> (t[])
PyAnyTorchListValue slice(const PyAnyTorchListValue &l, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step);

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step);

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype);

// aten::softplus : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue softplus(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &threshold__);

// aten::sort.int : (int[], bool) -> ()
void sort(const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse);

// aten::sort : (Tensor, int, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> sort(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &descending);

// aten::sqrt.int : (int) -> (float)
PyTorch_FloatValue sqrt(const PyTorch_IntValue &a);

// aten::sqrt : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sqrt(const PyAnyTorchTensorValue &self);

// aten::sqrt_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sqrt_(const PyAnyTorchTensorValue &self);

// aten::square : (Tensor) -> (Tensor)
PyAnyTorchTensorValue square(const PyAnyTorchTensorValue &self);

// aten::square_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue square_(const PyAnyTorchTensorValue &self);

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue squeeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::squeeze_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue squeeze_copy(const PyAnyTorchTensorValue &self);

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::squeeze : (Tensor) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &self);

// aten::stack : (Tensor[], int) -> (Tensor)
PyAnyTorchTensorValue stack(const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim);

// aten::std.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim);

// aten::std.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim);

// aten::std : (Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased);

// aten::sub.float : (float, float) -> (float)
PyTorch_FloatValue sub(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b);

// aten::sub.int : (int, int) -> (int)
PyTorch_IntValue sub(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha);

// aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha);

// aten::sub_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha);

// aten::sub_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha);

// aten::sum.dim_IntList : (Tensor, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype);

// aten::sum : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype);

// aten::t_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue t_copy(const PyAnyTorchTensorValue &self);

// aten::t : (Tensor) -> (Tensor)
PyAnyTorchTensorValue t(const PyAnyTorchTensorValue &self);

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output);

// aten::tanh : (Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh(const PyAnyTorchTensorValue &self);

// aten::tanh_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh_(const PyAnyTorchTensorValue &self);

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad);

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad);

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad);

// aten::tensor : (t[], int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyAnyTorchListValue &data, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad);

// aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__);

// aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value);

// aten::threshold_ : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value);

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format);

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format);

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format);

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format);

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy);

// aten::topk : (Tensor, int, int, bool, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> topk(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &k, const PyTorch_IntValue &dim, const PyTorch_BoolValue &largest, const PyTorch_BoolValue &sorted);

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue transpose_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1);

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue transpose(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1);

// aten::triu : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue triu(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal);

// aten::triu_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue triu_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal);

// aten::type_as : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue type_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue unfold_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step);

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue uniform(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue uniform_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator);

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::unsqueeze : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim);

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
PyAnyTorchTensorValue upsample_nearest2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w);

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
PyAnyTorchTensorValue upsample_nearest2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w);

// aten::var.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim);

// aten::var.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim);

// aten::var_mean.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim);

// aten::var_mean.dim : (Tensor, int[]?, bool, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim);

// aten::var_mean : (Tensor, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased);

// aten::var : (Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased);

// aten::view_as_complex : (Tensor) -> (Tensor)
PyAnyTorchTensorValue view_as_complex(const PyAnyTorchTensorValue &self);

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue view_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype);

// aten::view_copy : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue view_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size);

// aten::view : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size);

// aten::where.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchScalarValue &other);

// aten::where.ScalarOther : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other);

// aten::where.ScalarSelf : (Tensor, Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &other);

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::zero : (Tensor) -> (Tensor)
PyAnyTorchTensorValue zero(const PyAnyTorchTensorValue &self);

// aten::zero_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue zero_(const PyAnyTorchTensorValue &self);

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue zeros_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format);

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue zeros(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory);

// aten::_convolution.deprecated : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled);

// aten::_convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, const PyTorch_BoolValue &allow_tf32);

// aten::_embedding_bag : (Tensor, Tensor, Tensor, bool, int, bool, Tensor?, bool, int) -> (Tensor, Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> _embedding_bag(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyAnyTorchTensorValue &offsets, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_IntValue &mode, const PyTorch_BoolValue &sparse, const PyAnyTorchOptionalTensorValue &per_sample_weights, const PyTorch_BoolValue &include_last_offset, const PyTorch_IntValue &padding_idx);

// aten::_index_put_impl : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _index_put_impl(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, const PyTorch_BoolValue &unsafe);

// aten::_index_put_impl_ : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _index_put_impl_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, const PyTorch_BoolValue &unsafe);

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue _log_softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype);

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue _log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float);

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue _reshape_alias_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride);

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue _reshape_alias(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride);

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
PyAnyTorchTensorValue _shape_as_tensor(const PyAnyTorchTensorValue &self);

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue _softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype);

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue _softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float);

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue _to_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format);

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue _unsafe_view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size);

// aten::__and__.bool : (bool, bool) -> (bool)
PyTorch_BoolValue __and__(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b);

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __and__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other);

// aten::__contains__.int_list : (int[], int) -> (bool)
PyTorch_BoolValue __contains__(const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item);

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
PyTorch_BoolValue __contains__(const PyTorch_DictValue &dict, const PyTorch_StringValue &key);

// aten::__derive_index : (int, int, int) -> (int)
PyTorch_IntValue __derive_index(const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step);

// aten::__not__ : (bool) -> (bool)
PyTorch_BoolValue __not__(const PyTorch_BoolValue &self);

// aten::__range_length : (int, int, int) -> (int)
PyTorch_IntValue __range_length(const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step);

// prim::device : (Tensor) -> (Device)
PyTorch_DeviceValue device(const PyAnyTorchTensorValue &a);

// prim::dtype : (Tensor) -> (int)
PyTorch_IntValue dtype(const PyAnyTorchTensorValue &a);

// prim::layout : (Tensor) -> (int)
PyTorch_IntValue layout(const PyAnyTorchTensorValue &a);

// prim::max.int : (int, int) -> (int)
PyTorch_IntValue max(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// prim::max.self_int : (int[]) -> (int)
PyTorch_IntValue max(const PyAnyTorchListOfTorchIntValue &self);

// prim::min.int : (int, int) -> (int)
PyTorch_IntValue min(const PyTorch_IntValue &a, const PyTorch_IntValue &b);

// prim::min.self_int : (int[]) -> (int)
PyTorch_IntValue min(const PyAnyTorchListOfTorchIntValue &self);

// prim::NumToTensor.Scalar : (Scalar) -> (Tensor)
PyAnyTorchTensorValue NumToTensor(const PyAnyTorchScalarValue &a);

// prim::RaiseException : (str, str?) -> ()
void RaiseException(const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls);

// prims::convert_element_type : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue convert_element_type(const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype);

// prims::squeeze : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &a, const PyAnyTorchListOfTorchIntValue &dimensions);

// prims::var : (Tensor, int[]?, float, int?) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &inp, const PyAnyTorchOptionalListOfTorchIntValue &dims, const PyTorch_FloatValue &correction, const PyAnyTorchOptionalIntValue &output_dtype);

// prims::view_of : (Tensor) -> (Tensor)
PyAnyTorchTensorValue view_of(const PyAnyTorchTensorValue &a);

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i);
