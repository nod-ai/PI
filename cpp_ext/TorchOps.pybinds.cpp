
// aten::Bool.Tensor : (Tensor) -> (bool)
m.def("Bool", py::overload_cast<const PyAnyTorchTensorValue &>(&Bool));

// aten::Bool.float : (float) -> (bool)
m.def("Bool", py::overload_cast<const PyTorch_FloatValue &>(&Bool));

// aten::Bool.int : (int) -> (bool)
m.def("Bool", py::overload_cast<const PyTorch_IntValue &>(&Bool));

// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
m.def("Delete", py::overload_cast<const PyTorch_DictValue &, const PyTorch_StringValue &>(&Delete));

// aten::Float.Tensor : (Tensor) -> (float)
m.def("Float", py::overload_cast<const PyAnyTorchTensorValue &>(&Float));

// aten::Float.str : (str) -> (float)
m.def("Float", py::overload_cast<const PyTorch_StringValue &>(&Float));

// aten::FloatImplicit : (Tensor) -> (float)
m.def("FloatImplicit", py::overload_cast<const PyAnyTorchTensorValue &>(&FloatImplicit));

// aten::Int.Tensor : (Tensor) -> (int)
m.def("Int", py::overload_cast<const PyAnyTorchTensorValue &>(&Int));

// aten::Int.float : (float) -> (int)
m.def("Int", py::overload_cast<const PyTorch_FloatValue &>(&Int));

// aten::Int.bool : (bool) -> (int)
m.def("Int", py::overload_cast<const PyTorch_BoolValue &>(&Int));

// aten::IntImplicit : (Tensor) -> (int)
m.def("IntImplicit", py::overload_cast<const PyAnyTorchTensorValue &>(&IntImplicit));

// prim::RaiseException : (str, str?) -> ()
m.def("RaiseException", py::overload_cast<const PyTorch_StringValue &, const PyAnyTorchOptionalStringValue &>(&RaiseException));

// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("__and__", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&__and__));

// aten::__and__.bool : (bool, bool) -> (bool)
m.def("__and__", py::overload_cast<const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&__and__));

// aten::__contains__.str : (Dict(str, t), str) -> (bool)
m.def("__contains__", py::overload_cast<const PyTorch_DictValue &, const PyTorch_StringValue &>(&__contains__));

// aten::__contains__.int_list : (int[], int) -> (bool)
m.def("__contains__", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyTorch_IntValue &>(&__contains__));

// aten::__derive_index : (int, int, int) -> (int)
m.def("__derive_index", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&__derive_index));

// aten::__not__ : (bool) -> (bool)
m.def("__not__", py::overload_cast<const PyTorch_BoolValue &>(&__not__));

// aten::__range_length : (int, int, int) -> (int)
m.def("__range_length", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&__range_length));

// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
m.def("_log_softmax", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &>(&_log_softmax));

// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
m.def("_log_softmax_backward_data", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&_log_softmax_backward_data));

// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
m.def("_reshape_alias", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&_reshape_alias));

// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
m.def("_reshape_alias_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&_reshape_alias_copy));

// aten::_shape_as_tensor : (Tensor) -> (Tensor)
m.def("_shape_as_tensor", py::overload_cast<const PyAnyTorchTensorValue &>(&_shape_as_tensor));

// aten::_softmax : (Tensor, int, bool) -> (Tensor)
m.def("_softmax", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &>(&_softmax));

// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
m.def("_softmax_backward_data", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&_softmax_backward_data));

// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
m.def("_to_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &, const PyTorch_BoolValue &, const PyAnyTorchOptionalIntValue &>(&_to_copy));

// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
m.def("_unsafe_view", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&_unsafe_view));

// aten::abs : (Tensor) -> (Tensor)
m.def("abs", py::overload_cast<const PyAnyTorchTensorValue &>(&abs));

// aten::abs_ : (Tensor) -> (Tensor)
m.def("abs_", py::overload_cast<const PyAnyTorchTensorValue &>(&abs_));

// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
m.def("adaptive_avg_pool2d", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&adaptive_avg_pool2d));

// aten::add.str : (str, str) -> (str)
m.def("add", py::overload_cast<const PyTorch_StringValue &, const PyTorch_StringValue &>(&add));

// aten::add.int : (int, int) -> (int)
m.def("add", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&add));

// aten::add.float_int : (float, int) -> (float)
m.def("add", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&add));

// aten::alias_copy : (Tensor) -> (Tensor)
m.def("alias_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&alias_copy));

// aten::all : (Tensor) -> (Tensor)
m.def("all", py::overload_cast<const PyAnyTorchTensorValue &>(&all));

// aten::all.bool : (bool[]) -> (bool)
m.def("all", py::overload_cast<const PyAnyTorchListOfTorchBoolValue &>(&all));

// aten::amax : (Tensor, int[], bool) -> (Tensor)
m.def("amax", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&amax));

// aten::any : (Tensor) -> (Tensor)
m.def("any", py::overload_cast<const PyAnyTorchTensorValue &>(&any));

// aten::any.dim : (Tensor, int, bool) -> (Tensor)
m.def("any", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &>(&any));

// aten::any.bool : (bool[]) -> (bool)
m.def("any", py::overload_cast<const PyAnyTorchListOfTorchBoolValue &>(&any));

// aten::argmax : (Tensor, int?, bool) -> (Tensor)
m.def("argmax", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyTorch_BoolValue &>(&argmax));

// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
m.def("as_strided_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &>(&as_strided_copy));

// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
m.def("as_strided_scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &>(&as_strided_scatter));

// aten::atan : (Tensor) -> (Tensor)
m.def("atan", py::overload_cast<const PyAnyTorchTensorValue &>(&atan));

// aten::atan2 : (Tensor, Tensor) -> (Tensor)
m.def("atan2", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&atan2));

// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
m.def("atan2_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&atan2_));

// aten::atan_ : (Tensor) -> (Tensor)
m.def("atan_", py::overload_cast<const PyAnyTorchTensorValue &>(&atan_));

// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
m.def("avg_pool2d", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &, const PyAnyTorchOptionalIntValue &>(&avg_pool2d));

// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
m.def("bernoulli", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalGeneratorValue &>(&bernoulli));

// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
m.def("bernoulli", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_FloatValue &, const PyAnyTorchOptionalGeneratorValue &>(&bernoulli));

// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
m.def("bernoulli", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchOptionalGeneratorValue &>(&bernoulli));

// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
m.def("bernoulli_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_FloatValue &, const PyAnyTorchOptionalGeneratorValue &>(&bernoulli_));

// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
m.def("bernoulli_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchOptionalGeneratorValue &>(&bernoulli_));

// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_and", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_and));

// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_and_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_and_));

// aten::bitwise_not : (Tensor) -> (Tensor)
m.def("bitwise_not", py::overload_cast<const PyAnyTorchTensorValue &>(&bitwise_not));

// aten::bitwise_not_ : (Tensor) -> (Tensor)
m.def("bitwise_not_", py::overload_cast<const PyAnyTorchTensorValue &>(&bitwise_not_));

// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_or", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_or));

// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_or_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_or_));

// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_xor", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_xor));

// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("bitwise_xor_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bitwise_xor_));

// aten::bmm : (Tensor, Tensor) -> (Tensor)
m.def("bmm", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&bmm));

// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
m.def("broadcast_to", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&broadcast_to));

// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
m.def("bucketize", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&bucketize));

// aten::ceil : (Tensor) -> (Tensor)
m.def("ceil", py::overload_cast<const PyAnyTorchTensorValue &>(&ceil));

// aten::ceil.float : (float) -> (int)
m.def("ceil", py::overload_cast<const PyTorch_FloatValue &>(&ceil));

// aten::ceil_ : (Tensor) -> (Tensor)
m.def("ceil_", py::overload_cast<const PyAnyTorchTensorValue &>(&ceil_));

// aten::clone : (Tensor, int?) -> (Tensor)
m.def("clone", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &>(&clone));

// aten::contiguous : (Tensor, int) -> (Tensor)
m.def("contiguous", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&contiguous));

// prims::convert_element_type : (Tensor, int) -> (Tensor)
m.def("convert_element_type", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&convert_element_type));

// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
m.def("copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&copy));

// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
m.def("copy_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&copy_));

// aten::cos : (Tensor) -> (Tensor)
m.def("cos", py::overload_cast<const PyAnyTorchTensorValue &>(&cos));

// aten::cos_ : (Tensor) -> (Tensor)
m.def("cos_", py::overload_cast<const PyAnyTorchTensorValue &>(&cos_));

// aten::cpu : (Tensor) -> (Tensor)
m.def("cpu", py::overload_cast<const PyAnyTorchTensorValue &>(&cpu));

// aten::cumsum : (Tensor, int, int?) -> (Tensor)
m.def("cumsum", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchOptionalIntValue &>(&cumsum));

// aten::detach : (Tensor) -> (Tensor)
m.def("detach", py::overload_cast<const PyAnyTorchTensorValue &>(&detach));

// aten::detach_copy : (Tensor) -> (Tensor)
m.def("detach_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&detach_copy));

// prim::device : (Tensor) -> (Device)
m.def("device", py::overload_cast<const PyAnyTorchTensorValue &>(&device));

// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
m.def("diagonal_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&diagonal_copy));

// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
m.def("diagonal_scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&diagonal_scatter));

// aten::dim : (Tensor) -> (int)
m.def("dim", py::overload_cast<const PyAnyTorchTensorValue &>(&dim));

// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("div", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&div));

// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
m.def("div", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchOptionalStringValue &>(&div));

// aten::div.int : (int, int) -> (float)
m.def("div", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&div));

// aten::div.float : (float, float) -> (float)
m.def("div", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&div));

// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("div_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&div_));

// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
m.def("div_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchOptionalStringValue &>(&div_));

// aten::dropout : (Tensor, float, bool) -> (Tensor)
m.def("dropout", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_FloatValue &, const PyTorch_BoolValue &>(&dropout));

// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
m.def("dropout_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_FloatValue &, const PyTorch_BoolValue &>(&dropout_));

// prim::dtype : (Tensor) -> (int)
m.def("dtype", py::overload_cast<const PyAnyTorchTensorValue &>(&dtype));

// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
m.def("embedding", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&embedding));

// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
m.def("embedding_dense_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &>(&embedding_dense_backward));

// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("empty", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &, const PyAnyTorchOptionalIntValue &>(&empty));

// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("empty_like", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &, const PyAnyTorchOptionalIntValue &>(&empty_like));

// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("eq", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&eq));

// aten::eq.int_list : (int[], int[]) -> (bool)
m.def("eq", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&eq));

// aten::eq.str : (str, str) -> (bool)
m.def("eq", py::overload_cast<const PyTorch_StringValue &, const PyTorch_StringValue &>(&eq));

// aten::eq.int : (int, int) -> (bool)
m.def("eq", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&eq));

// aten::eq.float : (float, float) -> (bool)
m.def("eq", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&eq));

// aten::eq.device : (Device, Device) -> (bool)
m.def("eq", py::overload_cast<const PyTorch_DeviceValue &, const PyTorch_DeviceValue &>(&eq));

// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("eq_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&eq_));

// aten::erf : (Tensor) -> (Tensor)
m.def("erf", py::overload_cast<const PyAnyTorchTensorValue &>(&erf));

// aten::erf_ : (Tensor) -> (Tensor)
m.def("erf_", py::overload_cast<const PyAnyTorchTensorValue &>(&erf_));

// aten::exp : (Tensor) -> (Tensor)
m.def("exp", py::overload_cast<const PyAnyTorchTensorValue &>(&exp));

// aten::exp_ : (Tensor) -> (Tensor)
m.def("exp_", py::overload_cast<const PyAnyTorchTensorValue &>(&exp_));

// aten::expand : (Tensor, int[], bool) -> (Tensor)
m.def("expand", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&expand));

// aten::expand_as : (Tensor, Tensor) -> (Tensor)
m.def("expand_as", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&expand_as));

// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
m.def("expand_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&expand_copy));

// aten::expm1 : (Tensor) -> (Tensor)
m.def("expm1", py::overload_cast<const PyAnyTorchTensorValue &>(&expm1));

// aten::expm1_ : (Tensor) -> (Tensor)
m.def("expm1_", py::overload_cast<const PyAnyTorchTensorValue &>(&expm1_));

// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
m.def("fft_fft", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyTorch_IntValue &, const PyAnyTorchOptionalStringValue &>(&fft_fft));

// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("fill", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&fill));

// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("fill_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&fill_));

// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
m.def("flatten", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&flatten));

// aten::flip : (Tensor, int[]) -> (Tensor)
m.def("flip", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&flip));

// aten::floor : (Tensor) -> (Tensor)
m.def("floor", py::overload_cast<const PyAnyTorchTensorValue &>(&floor));

// aten::floor_ : (Tensor) -> (Tensor)
m.def("floor_", py::overload_cast<const PyAnyTorchTensorValue &>(&floor_));

// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
m.def("floor_divide", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&floor_divide));

// aten::floordiv.int : (int, int) -> (int)
m.def("floordiv", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&floordiv));

// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
m.def("frobenius_norm", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&frobenius_norm));

// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
m.def("gather", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&gather));

// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ge", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&ge));

// aten::ge.int : (int, int) -> (bool)
m.def("ge", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&ge));

// aten::ge.float : (float, float) -> (bool)
m.def("ge", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&ge));

// aten::ge.float_int : (float, int) -> (bool)
m.def("ge", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&ge));

// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ge_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&ge_));

// aten::gelu : (Tensor, str) -> (Tensor)
m.def("gelu", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_StringValue &>(&gelu));

// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
m.def("gelu_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_StringValue &>(&gelu_backward));

// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("gt", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&gt));

// aten::gt.int : (int, int) -> (bool)
m.def("gt", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&gt));

// aten::gt.float : (float, float) -> (bool)
m.def("gt", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&gt));

// aten::gt.float_int : (float, int) -> (bool)
m.def("gt", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&gt));

// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("gt_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&gt_));

// aten::hardsigmoid : (Tensor) -> (Tensor)
m.def("hardsigmoid", py::overload_cast<const PyAnyTorchTensorValue &>(&hardsigmoid));

// aten::hardsigmoid_ : (Tensor) -> (Tensor)
m.def("hardsigmoid_", py::overload_cast<const PyAnyTorchTensorValue &>(&hardsigmoid_));

// aten::hardswish : (Tensor) -> (Tensor)
m.def("hardswish", py::overload_cast<const PyAnyTorchTensorValue &>(&hardswish));

// aten::hardswish_ : (Tensor) -> (Tensor)
m.def("hardswish_", py::overload_cast<const PyAnyTorchTensorValue &>(&hardswish_));

// aten::imag : (Tensor) -> (Tensor)
m.def("imag", py::overload_cast<const PyAnyTorchTensorValue &>(&imag));

// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
m.def("index_select", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &>(&index_select));

// aten::is_floating_point : (Tensor) -> (bool)
m.def("is_floating_point", py::overload_cast<const PyAnyTorchTensorValue &>(&is_floating_point));

// aten::join : (str, str[]) -> (str)
m.def("join", py::overload_cast<const PyTorch_StringValue &, const PyAnyTorchListOfTorchStringValue &>(&join));

// aten::keys.str : (Dict(str, t)) -> (str[])
m.def("keys", py::overload_cast<const PyTorch_DictValue &>(&keys));

// prim::layout : (Tensor) -> (int)
m.def("layout", py::overload_cast<const PyAnyTorchTensorValue &>(&layout));

// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("le", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&le));

// aten::le.int : (int, int) -> (bool)
m.def("le", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&le));

// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("le_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&le_));

// aten::len.Tensor : (Tensor) -> (int)
m.def("len", py::overload_cast<const PyAnyTorchTensorValue &>(&len));

// aten::len.str : (str) -> (int)
m.def("len", py::overload_cast<const PyTorch_StringValue &>(&len));

// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("lerp", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&lerp));

// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("lerp_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&lerp_));

// aten::lift_fresh_copy : (Tensor) -> (Tensor)
m.def("lift_fresh_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&lift_fresh_copy));

// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
m.def("linear", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_LinearParamsValue &, const PyTorch_FloatValue &, const PyTorch_IntValue &>(&linear));

// aten::log : (Tensor) -> (Tensor)
m.def("log", py::overload_cast<const PyAnyTorchTensorValue &>(&log));

// aten::log.int : (int) -> (float)
m.def("log", py::overload_cast<const PyTorch_IntValue &>(&log));

// aten::log1p : (Tensor) -> (Tensor)
m.def("log1p", py::overload_cast<const PyAnyTorchTensorValue &>(&log1p));

// aten::log1p_ : (Tensor) -> (Tensor)
m.def("log1p_", py::overload_cast<const PyAnyTorchTensorValue &>(&log1p_));

// aten::log2 : (Tensor) -> (Tensor)
m.def("log2", py::overload_cast<const PyAnyTorchTensorValue &>(&log2));

// aten::log2_ : (Tensor) -> (Tensor)
m.def("log2_", py::overload_cast<const PyAnyTorchTensorValue &>(&log2_));

// aten::log_ : (Tensor) -> (Tensor)
m.def("log_", py::overload_cast<const PyAnyTorchTensorValue &>(&log_));

// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
m.def("log_softmax", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchOptionalIntValue &>(&log_softmax));

// aten::logical_and : (Tensor, Tensor) -> (Tensor)
m.def("logical_and", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_and));

// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_and_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_and_));

// aten::logical_not : (Tensor) -> (Tensor)
m.def("logical_not", py::overload_cast<const PyAnyTorchTensorValue &>(&logical_not));

// aten::logical_not_ : (Tensor) -> (Tensor)
m.def("logical_not_", py::overload_cast<const PyAnyTorchTensorValue &>(&logical_not_));

// aten::logical_or : (Tensor, Tensor) -> (Tensor)
m.def("logical_or", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_or));

// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_or_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_or_));

// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
m.def("logical_xor", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_xor));

// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
m.def("logical_xor_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&logical_xor_));

// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
m.def("logsumexp", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&logsumexp));

// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("lt", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&lt));

// aten::lt.int : (int, int) -> (bool)
m.def("lt", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&lt));

// aten::lt.float : (float, float) -> (bool)
m.def("lt", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&lt));

// aten::lt.float_int : (float, int) -> (bool)
m.def("lt", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&lt));

// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("lt_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&lt_));

// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("masked_fill", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&masked_fill));

// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("masked_fill_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&masked_fill_));

// aten::masked_select : (Tensor, Tensor) -> (Tensor)
m.def("masked_select", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&masked_select));

// aten::matmul : (Tensor, Tensor) -> (Tensor)
m.def("matmul", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&matmul));

// aten::max : (Tensor) -> (Tensor)
m.def("max", py::overload_cast<const PyAnyTorchTensorValue &>(&max));

// prim::max.self_int : (int[]) -> (int)
m.def("max", py::overload_cast<const PyAnyTorchListOfTorchIntValue &>(&max));

// prim::max.int : (int, int) -> (int)
m.def("max", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&max));

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
m.def("max_pool2d", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&max_pool2d));

// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
m.def("max_pool2d_with_indices_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &, const PyAnyTorchTensorValue &>(&max_pool2d_with_indices_backward));

// aten::maximum : (Tensor, Tensor) -> (Tensor)
m.def("maximum", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&maximum));

// aten::mean : (Tensor, int?) -> (Tensor)
m.def("mean", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &>(&mean));

// prim::min.self_int : (int[]) -> (int)
m.def("min", py::overload_cast<const PyAnyTorchListOfTorchIntValue &>(&min));

// prim::min.int : (int, int) -> (int)
m.def("min", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&min));

// aten::minimum : (Tensor, Tensor) -> (Tensor)
m.def("minimum", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&minimum));

// aten::mish : (Tensor) -> (Tensor)
m.def("mish", py::overload_cast<const PyAnyTorchTensorValue &>(&mish));

// aten::mm : (Tensor, Tensor) -> (Tensor)
m.def("mm", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&mm));

// aten::movedim.int : (Tensor, int, int) -> (Tensor)
m.def("movedim", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&movedim));

// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
m.def("mse_loss", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&mse_loss));

// aten::mse_loss_backward : (Tensor, Tensor, Tensor, int) -> (Tensor)
m.def("mse_loss_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&mse_loss_backward));

// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("mul", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&mul));

// aten::mul.int : (int, int) -> (int)
m.def("mul", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&mul));

// aten::mul.float : (float, float) -> (float)
m.def("mul", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&mul));

// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("mul_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&mul_));

// aten::mv : (Tensor, Tensor) -> (Tensor)
m.def("mv", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&mv));

// aten::narrow : (Tensor, int, int, int) -> (Tensor)
m.def("narrow", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&narrow));

// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
m.def("native_dropout_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_FloatValue &>(&native_dropout_backward));

// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ne", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&ne));

// aten::ne.int_list : (int[], int[]) -> (bool)
m.def("ne", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&ne));

// aten::ne.int : (int, int) -> (bool)
m.def("ne", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&ne));

// aten::ne.float_int : (float, int) -> (bool)
m.def("ne", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(&ne));

// aten::ne.bool : (bool, bool) -> (bool)
m.def("ne", py::overload_cast<const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&ne));

// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
m.def("ne_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&ne_));

// aten::neg : (Tensor) -> (Tensor)
m.def("neg", py::overload_cast<const PyAnyTorchTensorValue &>(&neg));

// aten::neg.int : (int) -> (int)
m.def("neg", py::overload_cast<const PyTorch_IntValue &>(&neg));

// aten::neg.float : (float) -> (float)
m.def("neg", py::overload_cast<const PyTorch_FloatValue &>(&neg));

// aten::neg_ : (Tensor) -> (Tensor)
m.def("neg_", py::overload_cast<const PyAnyTorchTensorValue &>(&neg_));

// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_empty", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&new_empty));

// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_empty_strided", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&new_empty_strided));

// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_ones", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&new_ones));

// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("new_zeros", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&new_zeros));

// aten::numel : (Tensor) -> (int)
m.def("numel", py::overload_cast<const PyAnyTorchTensorValue &>(&numel));

// aten::numpy_T : (Tensor) -> (Tensor)
m.def("numpy_T", py::overload_cast<const PyAnyTorchTensorValue &>(&numpy_T));

// aten::one_hot : (Tensor, int) -> (Tensor)
m.def("one_hot", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&one_hot));

// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("ones", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&ones));

// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("ones_like", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &, const PyAnyTorchOptionalIntValue &>(&ones_like));

// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
m.def("pad", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyTorch_StringValue &, const PyAnyTorchOptionalFloatValue &>(&pad));

// aten::permute : (Tensor, int[]) -> (Tensor)
m.def("permute", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&permute));

// aten::permute_copy : (Tensor, int[]) -> (Tensor)
m.def("permute_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&permute_copy));

// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
m.def("pow", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&pow));

// aten::pow.int_float : (int, float) -> (float)
m.def("pow", py::overload_cast<const PyTorch_IntValue &, const PyTorch_FloatValue &>(&pow));

// aten::prelu : (Tensor, Tensor) -> (Tensor)
m.def("prelu", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&prelu));

// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("rand_like", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &, const PyAnyTorchOptionalIntValue &>(&rand_like));

// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randint", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&randint));

// aten::randint : (int, int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randint", py::overload_cast<const PyTorch_IntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&randint));

// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("randn", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&randn));

// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
m.def("randn", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalGeneratorValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&randn));

// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("randn_like", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &, const PyAnyTorchOptionalIntValue &>(&randn_like));

// aten::real : (Tensor) -> (Tensor)
m.def("real", py::overload_cast<const PyAnyTorchTensorValue &>(&real));

// aten::reciprocal : (Tensor) -> (Tensor)
m.def("reciprocal", py::overload_cast<const PyAnyTorchTensorValue &>(&reciprocal));

// aten::reciprocal_ : (Tensor) -> (Tensor)
m.def("reciprocal_", py::overload_cast<const PyAnyTorchTensorValue &>(&reciprocal_));

// aten::relu : (Tensor) -> (Tensor)
m.def("relu", py::overload_cast<const PyAnyTorchTensorValue &>(&relu));

// aten::relu6 : (Tensor) -> (Tensor)
m.def("relu6", py::overload_cast<const PyAnyTorchTensorValue &>(&relu6));

// aten::relu6_ : (Tensor) -> (Tensor)
m.def("relu6_", py::overload_cast<const PyAnyTorchTensorValue &>(&relu6_));

// aten::relu_ : (Tensor) -> (Tensor)
m.def("relu_", py::overload_cast<const PyAnyTorchTensorValue &>(&relu_));

// aten::remainder.int : (int, int) -> (int)
m.def("remainder", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&remainder));

// aten::repeat : (Tensor, int[]) -> (Tensor)
m.def("repeat", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&repeat));

// aten::reshape : (Tensor, int[]) -> (Tensor)
m.def("reshape", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&reshape));

// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
m.def("resize_", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &>(&resize_));

// aten::roll : (Tensor, int[], int[]) -> (Tensor)
m.def("roll", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &>(&roll));

// aten::round : (Tensor) -> (Tensor)
m.def("round", py::overload_cast<const PyAnyTorchTensorValue &>(&round));

// aten::round_ : (Tensor) -> (Tensor)
m.def("round_", py::overload_cast<const PyAnyTorchTensorValue &>(&round_));

// aten::rsqrt : (Tensor) -> (Tensor)
m.def("rsqrt", py::overload_cast<const PyAnyTorchTensorValue &>(&rsqrt));

// aten::rsqrt_ : (Tensor) -> (Tensor)
m.def("rsqrt_", py::overload_cast<const PyAnyTorchTensorValue &>(&rsqrt_));

// aten::scatter.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&scatter));

// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_add", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&scatter_add));

// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
m.def("scatter_add_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&scatter_add_));

// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
m.def("scatter_reduce", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_StringValue &, const PyTorch_BoolValue &>(&scatter_reduce));

// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
m.def("scatter_reduce_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_StringValue &, const PyTorch_BoolValue &>(&scatter_reduce_));

// aten::select.int : (Tensor, int, int) -> (Tensor)
m.def("select", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&select));

// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
m.def("select_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&select_copy));

// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
m.def("select_scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&select_scatter));

// aten::sigmoid : (Tensor) -> (Tensor)
m.def("sigmoid", py::overload_cast<const PyAnyTorchTensorValue &>(&sigmoid));

// aten::sigmoid_ : (Tensor) -> (Tensor)
m.def("sigmoid_", py::overload_cast<const PyAnyTorchTensorValue &>(&sigmoid_));

// aten::silu : (Tensor) -> (Tensor)
m.def("silu", py::overload_cast<const PyAnyTorchTensorValue &>(&silu));

// aten::silu_ : (Tensor) -> (Tensor)
m.def("silu_", py::overload_cast<const PyAnyTorchTensorValue &>(&silu_));

// aten::sin : (Tensor) -> (Tensor)
m.def("sin", py::overload_cast<const PyAnyTorchTensorValue &>(&sin));

// aten::sin_ : (Tensor) -> (Tensor)
m.def("sin_", py::overload_cast<const PyAnyTorchTensorValue &>(&sin_));

// aten::size : (Tensor) -> (int[])
m.def("size", py::overload_cast<const PyAnyTorchTensorValue &>(&size));

// aten::size.int : (Tensor, int) -> (int)
m.def("size", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&size));

// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyTorch_IntValue &>(&slice));

// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyTorch_IntValue &>(&slice_copy));

// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
m.def("slice_scatter", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyTorch_IntValue &>(&slice_scatter));

// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
m.def("softmax", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyAnyTorchOptionalIntValue &>(&softmax));

// aten::sort.int : (int[], bool) -> ()
m.def("sort", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyTorch_BoolValue &>(&sort));

// aten::sqrt : (Tensor) -> (Tensor)
m.def("sqrt", py::overload_cast<const PyAnyTorchTensorValue &>(&sqrt));

// aten::sqrt.int : (int) -> (float)
m.def("sqrt", py::overload_cast<const PyTorch_IntValue &>(&sqrt));

// aten::sqrt_ : (Tensor) -> (Tensor)
m.def("sqrt_", py::overload_cast<const PyAnyTorchTensorValue &>(&sqrt_));

// aten::square : (Tensor) -> (Tensor)
m.def("square", py::overload_cast<const PyAnyTorchTensorValue &>(&square));

// aten::square_ : (Tensor) -> (Tensor)
m.def("square_", py::overload_cast<const PyAnyTorchTensorValue &>(&square_));

// aten::squeeze.dim : (Tensor, int) -> (Tensor)
m.def("squeeze", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&squeeze));

// aten::squeeze : (Tensor) -> (Tensor)
m.def("squeeze", py::overload_cast<const PyAnyTorchTensorValue &>(&squeeze));

// prims::squeeze : (Tensor, int[]) -> (Tensor)
m.def("squeeze", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&squeeze));

// aten::squeeze_copy : (Tensor) -> (Tensor)
m.def("squeeze_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&squeeze_copy));

// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
m.def("squeeze_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&squeeze_copy));

// aten::std : (Tensor, bool) -> (Tensor)
m.def("std", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&std));

// aten::sub.int : (int, int) -> (int)
m.def("sub", py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(&sub));

// aten::sub.float : (float, float) -> (float)
m.def("sub", py::overload_cast<const PyTorch_FloatValue &, const PyTorch_FloatValue &>(&sub));

// aten::sum : (Tensor, int?) -> (Tensor)
m.def("sum", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &>(&sum));

// aten::t : (Tensor) -> (Tensor)
m.def("t", py::overload_cast<const PyAnyTorchTensorValue &>(&t));

// aten::t_copy : (Tensor) -> (Tensor)
m.def("t_copy", py::overload_cast<const PyAnyTorchTensorValue &>(&t_copy));

// aten::tanh : (Tensor) -> (Tensor)
m.def("tanh", py::overload_cast<const PyAnyTorchTensorValue &>(&tanh));

// aten::tanh_ : (Tensor) -> (Tensor)
m.def("tanh_", py::overload_cast<const PyAnyTorchTensorValue &>(&tanh_));

// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
m.def("tanh_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&tanh_backward));

// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
m.def("tensor", py::overload_cast<const PyTorch_BoolValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyTorch_BoolValue &>(&tensor));

// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
m.def("tensor", py::overload_cast<const PyTorch_IntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyTorch_BoolValue &>(&tensor));

// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
m.def("tensor", py::overload_cast<const PyTorch_FloatValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyTorch_BoolValue &>(&tensor));

// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
m.def("to", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &, const PyAnyTorchOptionalIntValue &>(&to));

// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
m.def("to", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &, const PyAnyTorchOptionalIntValue &>(&to));

// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
m.def("to", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &, const PyAnyTorchOptionalIntValue &>(&to));

// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
m.def("to", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalIntValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &>(&to));

// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
m.def("to", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_DeviceValue &, const PyTorch_IntValue &, const PyTorch_BoolValue &, const PyTorch_BoolValue &, const PyAnyTorchOptionalIntValue &>(&to));

// aten::transpose.int : (Tensor, int, int) -> (Tensor)
m.def("transpose", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&transpose));

// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
m.def("transpose_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&transpose_copy));

// aten::triu : (Tensor, int) -> (Tensor)
m.def("triu", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&triu));

// aten::triu_ : (Tensor, int) -> (Tensor)
m.def("triu_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&triu_));

// aten::type_as : (Tensor, Tensor) -> (Tensor)
m.def("type_as", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&type_as));

// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
m.def("unfold_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &, const PyTorch_IntValue &, const PyTorch_IntValue &>(&unfold_copy));

// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
m.def("uniform", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_FloatValue &, const PyTorch_FloatValue &, const PyAnyTorchOptionalGeneratorValue &>(&uniform));

// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
m.def("uniform_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_FloatValue &, const PyTorch_FloatValue &, const PyAnyTorchOptionalGeneratorValue &>(&uniform_));

// aten::unsqueeze : (Tensor, int) -> (Tensor)
m.def("unsqueeze", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&unsqueeze));

// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
m.def("unsqueeze_", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&unsqueeze_));

// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
m.def("unsqueeze_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&unsqueeze_copy));

// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
m.def("upsample_nearest2d", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalFloatValue &, const PyAnyTorchOptionalFloatValue &>(&upsample_nearest2d));

// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
m.def("upsample_nearest2d_backward", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalFloatValue &, const PyAnyTorchOptionalFloatValue &>(&upsample_nearest2d_backward));

// aten::var : (Tensor, bool) -> (Tensor)
m.def("var", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_BoolValue &>(&var));

// aten::view : (Tensor, int[]) -> (Tensor)
m.def("view", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&view));

// aten::view_as_complex : (Tensor) -> (Tensor)
m.def("view_as_complex", py::overload_cast<const PyAnyTorchTensorValue &>(&view_as_complex));

// aten::view_copy : (Tensor, int[]) -> (Tensor)
m.def("view_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchListOfTorchIntValue &>(&view_copy));

// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
m.def("view_copy", py::overload_cast<const PyAnyTorchTensorValue &, const PyTorch_IntValue &>(&view_copy));

// prims::view_of : (Tensor) -> (Tensor)
m.def("view_of", py::overload_cast<const PyAnyTorchTensorValue &>(&view_of));

// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
m.def("where", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &, const PyAnyTorchTensorValue &>(&where));

// aten::zero : (Tensor) -> (Tensor)
m.def("zero", py::overload_cast<const PyAnyTorchTensorValue &>(&zero));

// aten::zero_ : (Tensor) -> (Tensor)
m.def("zero_", py::overload_cast<const PyAnyTorchTensorValue &>(&zero_));

// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
m.def("zeros", py::overload_cast<const PyAnyTorchListOfTorchIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &>(&zeros));

// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
m.def("zeros_like", py::overload_cast<const PyAnyTorchTensorValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalIntValue &, const PyAnyTorchOptionalDeviceValue &, const PyAnyTorchOptionalBoolValue &, const PyAnyTorchOptionalIntValue &>(&zeros_like));
