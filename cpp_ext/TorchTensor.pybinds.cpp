
// __abs__(self) -> Tensor
// aten::abs : (Tensor) -> (Tensor)
c.def("__abs__", py::overload_cast<const PyAnyTorchTensorValue&>(&abs));

// @overload __and__(self, other Tensor) -> Tensor
// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__and__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&__and__));

// __bool__(self) -> builtins.bool
c.def("__bool__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__bool__ with signature __bool__(self) -> builtins.bool"); });

// __complex__(self) -> builtins.complex
c.def("__complex__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__complex__ with signature __complex__(self) -> builtins.complex"); });

// __div__(self, other Any) -> Tensor
// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__div__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&div));

// __eq__(self, other Any) -> Tensor
// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__eq__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&eq));

// __float__(self) -> builtins.float
c.def("__float__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__float__ with signature __float__(self) -> builtins.float"); });

// __ge__(self, other Any) -> Tensor
// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__ge__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&ge));

// __getitem__(self, indices Union[None, _int, slice, Tensor, List, Tuple]) -> Tensor
c.def("__getitem__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__getitem__ with signature __getitem__(self, indices Union[None, _int, slice, Tensor, List, Tuple]) -> Tensor"); });

// __gt__(self, other Any) -> Tensor
// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__gt__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&gt));

// __iadd__(self, other Any) -> Tensor
c.def("__iadd__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__iadd__ with signature __iadd__(self, other Any) -> Tensor"); });

// @overload __iand__(self, other Tensor) -> Tensor
c.def("__iand__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__iand__ with signature @overload __iand__(self, other Tensor) -> Tensor"); });

// @overload __iand__(self, other Number) -> Tensor
c.def("__iand__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__iand__ with signature @overload __iand__(self, other Number) -> Tensor"); });

// @overload __iand__(self, other Any) -> Tensor
c.def("__iand__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__iand__ with signature @overload __iand__(self, other Any) -> Tensor"); });

// __idiv__(self, other Any) -> Tensor
c.def("__idiv__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__idiv__ with signature __idiv__(self, other Any) -> Tensor"); });

// __ifloordiv__(self, other Any) -> Tensor
c.def("__ifloordiv__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ifloordiv__ with signature __ifloordiv__(self, other Any) -> Tensor"); });

// @overload __ilshift__(self, other Tensor) -> Tensor
c.def("__ilshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ilshift__ with signature @overload __ilshift__(self, other Tensor) -> Tensor"); });

// @overload __ilshift__(self, other Number) -> Tensor
c.def("__ilshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ilshift__ with signature @overload __ilshift__(self, other Number) -> Tensor"); });

// @overload __ilshift__(self, other Any) -> Tensor
c.def("__ilshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ilshift__ with signature @overload __ilshift__(self, other Any) -> Tensor"); });

// __imod__(self, other Any) -> Tensor
c.def("__imod__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__imod__ with signature __imod__(self, other Any) -> Tensor"); });

// __imul__(self, other Any) -> Tensor
c.def("__imul__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__imul__ with signature __imul__(self, other Any) -> Tensor"); });

// __index__(self) -> builtins.int
c.def("__index__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__index__ with signature __index__(self) -> builtins.int"); });

// __int__(self) -> builtins.int
c.def("__int__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__int__ with signature __int__(self) -> builtins.int"); });

// __invert__(self) -> Tensor
c.def("__invert__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__invert__ with signature __invert__(self) -> Tensor"); });

// @overload __ior__(self, other Tensor) -> Tensor
c.def("__ior__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ior__ with signature @overload __ior__(self, other Tensor) -> Tensor"); });

// @overload __ior__(self, other Number) -> Tensor
c.def("__ior__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ior__ with signature @overload __ior__(self, other Number) -> Tensor"); });

// @overload __ior__(self, other Any) -> Tensor
c.def("__ior__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ior__ with signature @overload __ior__(self, other Any) -> Tensor"); });

// @overload __irshift__(self, other Tensor) -> Tensor
c.def("__irshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__irshift__ with signature @overload __irshift__(self, other Tensor) -> Tensor"); });

// @overload __irshift__(self, other Number) -> Tensor
c.def("__irshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__irshift__ with signature @overload __irshift__(self, other Number) -> Tensor"); });

// @overload __irshift__(self, other Any) -> Tensor
c.def("__irshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__irshift__ with signature @overload __irshift__(self, other Any) -> Tensor"); });

// __isub__(self, other Any) -> Tensor
c.def("__isub__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__isub__ with signature __isub__(self, other Any) -> Tensor"); });

// @overload __ixor__(self, other Tensor) -> Tensor
c.def("__ixor__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ixor__ with signature @overload __ixor__(self, other Tensor) -> Tensor"); });

// @overload __ixor__(self, other Number) -> Tensor
c.def("__ixor__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ixor__ with signature @overload __ixor__(self, other Number) -> Tensor"); });

// @overload __ixor__(self, other Any) -> Tensor
c.def("__ixor__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ixor__ with signature @overload __ixor__(self, other Any) -> Tensor"); });

// __le__(self, other Any) -> Tensor
// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__le__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&le));

// __long__(self) -> builtins.int
c.def("__long__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__long__ with signature __long__(self) -> builtins.int"); });

// @overload __lshift__(self, other Tensor) -> Tensor
c.def("__lshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__lshift__ with signature @overload __lshift__(self, other Tensor) -> Tensor"); });

// @overload __lshift__(self, other Number) -> Tensor
c.def("__lshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__lshift__ with signature @overload __lshift__(self, other Number) -> Tensor"); });

// @overload __lshift__(self, other Any) -> Tensor
c.def("__lshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__lshift__ with signature @overload __lshift__(self, other Any) -> Tensor"); });

// __lt__(self, other Any) -> Tensor
// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__lt__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&lt));

// __matmul__(self, other Any) -> Tensor
// aten::matmul : (Tensor, Tensor) -> (Tensor)
c.def("__matmul__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&matmul));

// __mod__(self, other Any) -> Tensor
c.def("__mod__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__mod__ with signature __mod__(self, other Any) -> Tensor"); });

// __mul__(self, other Any) -> Tensor
// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__mul__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&mul));

// __ne__(self, other Any) -> Tensor
// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("__ne__", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&ne));

// __neg__(self) -> Tensor
// aten::neg : (Tensor) -> (Tensor)
c.def("__neg__", py::overload_cast<const PyAnyTorchTensorValue&>(&neg));

// __nonzero__(self) -> builtins.bool
c.def("__nonzero__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__nonzero__ with signature __nonzero__(self) -> builtins.bool"); });

// @overload __or__(self, other Tensor) -> Tensor
c.def("__or__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__or__ with signature @overload __or__(self, other Tensor) -> Tensor"); });

// @overload __or__(self, other Number) -> Tensor
c.def("__or__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__or__ with signature @overload __or__(self, other Number) -> Tensor"); });

// @overload __or__(self, other Any) -> Tensor
c.def("__or__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__or__ with signature @overload __or__(self, other Any) -> Tensor"); });

// __radd__(self, other Any) -> Tensor
c.def("__radd__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__radd__ with signature __radd__(self, other Any) -> Tensor"); });

// __rand__(self, other Any) -> Tensor
c.def("__rand__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rand__ with signature __rand__(self, other Any) -> Tensor"); });

// __rfloordiv__(self, other Any) -> Tensor
c.def("__rfloordiv__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rfloordiv__ with signature __rfloordiv__(self, other Any) -> Tensor"); });

// __rmul__(self, other Any) -> Tensor
c.def("__rmul__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rmul__ with signature __rmul__(self, other Any) -> Tensor"); });

// __ror__(self, other Any) -> Tensor
c.def("__ror__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__ror__ with signature __ror__(self, other Any) -> Tensor"); });

// __rpow__(self, other Any) -> Tensor
c.def("__rpow__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rpow__ with signature __rpow__(self, other Any) -> Tensor"); });

// @overload __rshift__(self, other Tensor) -> Tensor
c.def("__rshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rshift__ with signature @overload __rshift__(self, other Tensor) -> Tensor"); });

// @overload __rshift__(self, other Number) -> Tensor
c.def("__rshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rshift__ with signature @overload __rshift__(self, other Number) -> Tensor"); });

// @overload __rshift__(self, other Any) -> Tensor
c.def("__rshift__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rshift__ with signature @overload __rshift__(self, other Any) -> Tensor"); });

// __rtruediv__(self, other Any) -> Tensor
c.def("__rtruediv__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rtruediv__ with signature __rtruediv__(self, other Any) -> Tensor"); });

// __rxor__(self, other Any) -> Tensor
c.def("__rxor__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__rxor__ with signature __rxor__(self, other Any) -> Tensor"); });

// __setitem__(self, indices Union[None, _int, slice, Tensor, List, Tuple], val Union[Tensor, Number]) -> None
c.def("__setitem__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__setitem__ with signature __setitem__(self, indices Union[None, _int, slice, Tensor, List, Tuple], val Union[Tensor, Number]) -> None"); });

// __truediv__(self, other Any) -> Tensor
c.def("__truediv__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__truediv__ with signature __truediv__(self, other Any) -> Tensor"); });

// @overload __xor__(self, other Tensor) -> Tensor
c.def("__xor__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__xor__ with signature @overload __xor__(self, other Tensor) -> Tensor"); });

// @overload __xor__(self, other Number) -> Tensor
c.def("__xor__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__xor__ with signature @overload __xor__(self, other Number) -> Tensor"); });

// @overload __xor__(self, other Any) -> Tensor
c.def("__xor__", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("__xor__ with signature @overload __xor__(self, other Any) -> Tensor"); });

// _addmm_activation(self, mat1 Tensor, mat2 Tensor, *, beta Number=1, alpha Number=1, use_gelu _bool=False) -> Tensor
c.def("_addmm_activation", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_addmm_activation with signature _addmm_activation(self, mat1 Tensor, mat2 Tensor, *, beta Number=1, alpha Number=1, use_gelu _bool=False) -> Tensor"); });

// _autocast_to_full_precision(self, cuda_enabled _bool, cpu_enabled _bool) -> Tensor
c.def("_autocast_to_full_precision", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_autocast_to_full_precision with signature _autocast_to_full_precision(self, cuda_enabled _bool, cpu_enabled _bool) -> Tensor"); });

// _autocast_to_reduced_precision(self, cuda_enabled _bool, cpu_enabled _bool, cuda_dtype _dtype, cpu_dtype _dtype) -> Tensor
c.def("_autocast_to_reduced_precision", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_autocast_to_reduced_precision with signature _autocast_to_reduced_precision(self, cuda_enabled _bool, cpu_enabled _bool, cuda_dtype _dtype, cpu_dtype _dtype) -> Tensor"); });

// _coalesced_(self, coalesced _bool) -> Tensor
c.def("_coalesced_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_coalesced_ with signature _coalesced_(self, coalesced _bool) -> Tensor"); });

// _conj(self) -> Tensor
c.def("_conj", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_conj with signature _conj(self) -> Tensor"); });

// _conj_physical(self) -> Tensor
c.def("_conj_physical", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_conj_physical with signature _conj_physical(self) -> Tensor"); });

// _dimI(self) -> _int
c.def("_dimI", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_dimI with signature _dimI(self) -> _int"); });

// _dimV(self) -> _int
c.def("_dimV", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_dimV with signature _dimV(self) -> _int"); });

// _indices(self) -> Tensor
c.def("_indices", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_indices with signature _indices(self) -> Tensor"); });

// _is_all_true(self) -> Tensor
c.def("_is_all_true", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_is_all_true with signature _is_all_true(self) -> Tensor"); });

// _is_any_true(self) -> Tensor
c.def("_is_any_true", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_is_any_true with signature _is_any_true(self) -> Tensor"); });

// _is_view(self) -> _bool
c.def("_is_view", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_is_view with signature _is_view(self) -> _bool"); });

// _is_zerotensor(self) -> _bool
c.def("_is_zerotensor", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_is_zerotensor with signature _is_zerotensor(self) -> _bool"); });

// @staticmethod _make_subclass(cls Type[S], data Tensor, require_grad _bool=False, dispatch_strides _bool=False, dispatch_device _bool=False, device_for_backend_keys Optional[_device]=None) -> S
c.def("_make_subclass", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_make_subclass with signature @staticmethod _make_subclass(cls Type[S], data Tensor, require_grad _bool=False, dispatch_strides _bool=False, dispatch_device _bool=False, device_for_backend_keys Optional[_device]=None) -> S"); });

// _neg_view(self) -> Tensor
c.def("_neg_view", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_neg_view with signature _neg_view(self) -> Tensor"); });

// _nested_tensor_size(self) -> Tensor
c.def("_nested_tensor_size", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_nested_tensor_size with signature _nested_tensor_size(self) -> Tensor"); });

// _nested_tensor_storage_offsets(self) -> Tensor
c.def("_nested_tensor_storage_offsets", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_nested_tensor_storage_offsets with signature _nested_tensor_storage_offsets(self) -> Tensor"); });

// _nested_tensor_strides(self) -> Tensor
c.def("_nested_tensor_strides", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_nested_tensor_strides with signature _nested_tensor_strides(self) -> Tensor"); });

// _nnz(self) -> _int
c.def("_nnz", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_nnz with signature _nnz(self) -> _int"); });

// _to_dense(self, dtype Optional[_dtype]=None, masked_grad Optional[_bool]=None) -> Tensor
c.def("_to_dense", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_to_dense with signature _to_dense(self, dtype Optional[_dtype]=None, masked_grad Optional[_bool]=None) -> Tensor"); });

// _values(self) -> Tensor
c.def("_values", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("_values with signature _values(self) -> Tensor"); });

// abs_(self) -> Tensor
// aten::abs_ : (Tensor) -> (Tensor)
c.def("abs_", py::overload_cast<const PyAnyTorchTensorValue&>(&abs_));

// absolute(self) -> Tensor
c.def("absolute", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("absolute with signature absolute(self) -> Tensor"); });

// absolute_(self) -> Tensor
c.def("absolute_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("absolute_ with signature absolute_(self) -> Tensor"); });

// acos(self) -> Tensor
c.def("acos", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("acos with signature acos(self) -> Tensor"); });

// acos_(self) -> Tensor
c.def("acos_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("acos_ with signature acos_(self) -> Tensor"); });

// acosh(self) -> Tensor
c.def("acosh", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("acosh with signature acosh(self) -> Tensor"); });

// acosh_(self) -> Tensor
c.def("acosh_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("acosh_ with signature acosh_(self) -> Tensor"); });

// add_(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat], *, alpha Optional[Number]=1) -> Tensor
// aten::add_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
c.def("add_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&add_));

// addbmm(self, batch1 Tensor, batch2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
c.def("addbmm", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("addbmm with signature addbmm(self, batch1 Tensor, batch2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor"); });

// addbmm_(self, batch1 Tensor, batch2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
c.def("addbmm_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("addbmm_ with signature addbmm_(self, batch1 Tensor, batch2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor"); });

// addcdiv(self, tensor1 Tensor, tensor2 Tensor, *, value Number=1) -> Tensor
// aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
c.def("addcdiv", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&addcdiv));

// addcdiv_(self, tensor1 Tensor, tensor2 Tensor, *, value Number=1) -> Tensor
// aten::addcdiv_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
c.def("addcdiv_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&addcdiv_));

// addcmul(self, tensor1 Tensor, tensor2 Tensor, *, value Number=1) -> Tensor
// aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
c.def("addcmul", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&addcmul));

// addcmul_(self, tensor1 Tensor, tensor2 Tensor, *, value Number=1) -> Tensor
// aten::addcmul_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
c.def("addcmul_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&addcmul_));

// addmm(self, mat1 Tensor, mat2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
// aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
c.def("addmm", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&, const PyAnyTorchScalarValue&>(&addmm));

// addmm_(self, mat1 Tensor, mat2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
c.def("addmm_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("addmm_ with signature addmm_(self, mat1 Tensor, mat2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor"); });

// addmv(self, mat Tensor, vec Tensor, *, beta Number=1, alpha Number=1) -> Tensor
c.def("addmv", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("addmv with signature addmv(self, mat Tensor, vec Tensor, *, beta Number=1, alpha Number=1) -> Tensor"); });

// addmv_(self, mat Tensor, vec Tensor, *, beta Number=1, alpha Number=1) -> Tensor
c.def("addmv_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("addmv_ with signature addmv_(self, mat Tensor, vec Tensor, *, beta Number=1, alpha Number=1) -> Tensor"); });

// addr(self, vec1 Tensor, vec2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
c.def("addr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("addr with signature addr(self, vec1 Tensor, vec2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor"); });

// addr_(self, vec1 Tensor, vec2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
c.def("addr_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("addr_ with signature addr_(self, vec1 Tensor, vec2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor"); });

// adjoint(self) -> Tensor
c.def("adjoint", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("adjoint with signature adjoint(self) -> Tensor"); });

// align_as(self, other Tensor) -> Tensor
c.def("align_as", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("align_as with signature align_as(self, other Tensor) -> Tensor"); });

// @overload align_to(self, order Sequence[Union[str, ellipsis, None]], ellipsis_idx _int) -> Tensor
c.def("align_to", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("align_to with signature @overload align_to(self, order Sequence[Union[str, ellipsis, None]], ellipsis_idx _int) -> Tensor"); });

// @overload align_to(self, names Sequence[Union[str, ellipsis, None]]) -> Tensor
c.def("align_to", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("align_to with signature @overload align_to(self, names Sequence[Union[str, ellipsis, None]]) -> Tensor"); });

// @overload all(self) -> Tensor
// aten::all : (Tensor) -> (Tensor)
c.def("all", py::overload_cast<const PyAnyTorchTensorValue&>(&all));

// allclose(self, other Tensor, rtol _float=1e-05, atol _float=1e-08, equal_nan _bool=False) -> _bool
c.def("allclose", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("allclose with signature allclose(self, other Tensor, rtol _float=1e-05, atol _float=1e-08, equal_nan _bool=False) -> _bool"); });

// amax(self, dim Union[_int, _size]=(), keepdim _bool=False) -> Tensor
// aten::amax : (Tensor, int[], bool) -> (Tensor)
c.def("amax", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&, const PyTorch_BoolValue&>(&amax));

// amin(self, dim Union[_int, _size]=(), keepdim _bool=False) -> Tensor
c.def("amin", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("amin with signature amin(self, dim Union[_int, _size]=(), keepdim _bool=False) -> Tensor"); });

// aminmax(self, *, dim Optional[_int]=None, keepdim _bool=False) -> torch.return_types.aminmax
c.def("aminmax", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("aminmax with signature aminmax(self, *, dim Optional[_int]=None, keepdim _bool=False) -> torch.return_types.aminmax"); });

// angle(self) -> Tensor
c.def("angle", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("angle with signature angle(self) -> Tensor"); });

// @overload any(self) -> Tensor
// aten::any : (Tensor) -> (Tensor)
c.def("any", py::overload_cast<const PyAnyTorchTensorValue&>(&any));

// @overload any(self, dim _int, keepdim _bool=False) -> Tensor
// aten::any.dim : (Tensor, int, bool) -> (Tensor)
c.def("any", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyTorch_BoolValue&>(&any));

// apply_(self, callable Callable) -> Tensor
c.def("apply_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("apply_ with signature apply_(self, callable Callable) -> Tensor"); });

// arccos(self) -> Tensor
c.def("arccos", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arccos with signature arccos(self) -> Tensor"); });

// arccos_(self) -> Tensor
c.def("arccos_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arccos_ with signature arccos_(self) -> Tensor"); });

// arccosh(self) -> Tensor
c.def("arccosh", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arccosh with signature arccosh(self) -> Tensor"); });

// arccosh_(self) -> Tensor
c.def("arccosh_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arccosh_ with signature arccosh_(self) -> Tensor"); });

// arcsin(self) -> Tensor
c.def("arcsin", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arcsin with signature arcsin(self) -> Tensor"); });

// arcsin_(self) -> Tensor
c.def("arcsin_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arcsin_ with signature arcsin_(self) -> Tensor"); });

// arcsinh(self) -> Tensor
c.def("arcsinh", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arcsinh with signature arcsinh(self) -> Tensor"); });

// arcsinh_(self) -> Tensor
c.def("arcsinh_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arcsinh_ with signature arcsinh_(self) -> Tensor"); });

// arctan(self) -> Tensor
c.def("arctan", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arctan with signature arctan(self) -> Tensor"); });

// arctan2(self, other Tensor) -> Tensor
c.def("arctan2", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arctan2 with signature arctan2(self, other Tensor) -> Tensor"); });

// arctan2_(self, other Tensor) -> Tensor
c.def("arctan2_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arctan2_ with signature arctan2_(self, other Tensor) -> Tensor"); });

// arctan_(self) -> Tensor
c.def("arctan_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arctan_ with signature arctan_(self) -> Tensor"); });

// arctanh(self) -> Tensor
c.def("arctanh", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arctanh with signature arctanh(self) -> Tensor"); });

// arctanh_(self) -> Tensor
c.def("arctanh_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("arctanh_ with signature arctanh_(self) -> Tensor"); });

// argmax(self, dim Optional[_int]=None, keepdim _bool=False) -> Tensor
// aten::argmax : (Tensor, int?, bool) -> (Tensor)
c.def("argmax", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchOptionalIntValue&, const PyTorch_BoolValue&>(&argmax));

// argmin(self, dim Optional[_int]=None, keepdim _bool=False) -> Tensor
c.def("argmin", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("argmin with signature argmin(self, dim Optional[_int]=None, keepdim _bool=False) -> Tensor"); });

// @overload argsort(self, *, stable _bool, dim _int=-1, descending _bool=False) -> Tensor
c.def("argsort", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("argsort with signature @overload argsort(self, *, stable _bool, dim _int=-1, descending _bool=False) -> Tensor"); });

// @overload argsort(self, dim _int=-1, descending _bool=False) -> Tensor
c.def("argsort", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("argsort with signature @overload argsort(self, dim _int=-1, descending _bool=False) -> Tensor"); });

// @overload argsort(self, dim Union[str, ellipsis, None], descending _bool=False) -> Tensor
c.def("argsort", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("argsort with signature @overload argsort(self, dim Union[str, ellipsis, None], descending _bool=False) -> Tensor"); });

// argwhere(self) -> Tensor
c.def("argwhere", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("argwhere with signature argwhere(self) -> Tensor"); });

// as_strided(self, size Sequence[Union[_int, SymInt]], stride Sequence[Union[_int, SymInt]], storage_offset Optional[Union[_int, SymInt]]=None) -> Tensor
c.def("as_strided", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("as_strided with signature as_strided(self, size Sequence[Union[_int, SymInt]], stride Sequence[Union[_int, SymInt]], storage_offset Optional[Union[_int, SymInt]]=None) -> Tensor"); });

// as_strided_(self, size Sequence[Union[_int, SymInt]], stride Sequence[Union[_int, SymInt]], storage_offset Optional[Union[_int, SymInt]]=None) -> Tensor
c.def("as_strided_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("as_strided_ with signature as_strided_(self, size Sequence[Union[_int, SymInt]], stride Sequence[Union[_int, SymInt]], storage_offset Optional[Union[_int, SymInt]]=None) -> Tensor"); });

// as_strided_scatter(self, src Tensor, size Sequence[Union[_int, SymInt]], stride Sequence[Union[_int, SymInt]], storage_offset Optional[Union[_int, SymInt]]=None) -> Tensor
// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
c.def("as_strided_scatter", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&, const PyAnyTorchListOfTorchIntValue&, const PyAnyTorchOptionalIntValue&>(&as_strided_scatter));

// as_subclass(self, cls Type[S]) -> S
c.def("as_subclass", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("as_subclass with signature as_subclass(self, cls Type[S]) -> S"); });

// asin(self) -> Tensor
c.def("asin", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("asin with signature asin(self) -> Tensor"); });

// asin_(self) -> Tensor
c.def("asin_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("asin_ with signature asin_(self) -> Tensor"); });

// asinh(self) -> Tensor
c.def("asinh", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("asinh with signature asinh(self) -> Tensor"); });

// asinh_(self) -> Tensor
c.def("asinh_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("asinh_ with signature asinh_(self) -> Tensor"); });

// atan(self) -> Tensor
c.def("atan", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("atan with signature atan(self) -> Tensor"); });

// atan2(self, other Tensor) -> Tensor
// aten::atan2 : (Tensor, Tensor) -> (Tensor)
c.def("atan2", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&atan2));

// atan2_(self, other Tensor) -> Tensor
// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
c.def("atan2_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&atan2_));

// atan_(self) -> Tensor
c.def("atan_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("atan_ with signature atan_(self) -> Tensor"); });

// atanh(self) -> Tensor
c.def("atanh", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("atanh with signature atanh(self) -> Tensor"); });

// atanh_(self) -> Tensor
c.def("atanh_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("atanh_ with signature atanh_(self) -> Tensor"); });

// baddbmm(self, batch1 Tensor, batch2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
// aten::baddbmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
c.def("baddbmm", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&, const PyAnyTorchScalarValue&>(&baddbmm));

// baddbmm_(self, batch1 Tensor, batch2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
// aten::baddbmm_ : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
c.def("baddbmm_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&, const PyAnyTorchScalarValue&>(&baddbmm_));

// @overload bernoulli(self, *, generator Optional[Generator]=None) -> Tensor
// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
c.def("bernoulli", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchOptionalGeneratorValue&>(&bernoulli));

// @overload bernoulli(self, p _float, *, generator Optional[Generator]=None) -> Tensor
// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
c.def("bernoulli", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_FloatValue&, const PyAnyTorchOptionalGeneratorValue&>(&bernoulli));

// @overload bernoulli_(self, p Tensor, *, generator Optional[Generator]=None) -> Tensor
// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
c.def("bernoulli_", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_FloatValue&, const PyAnyTorchOptionalGeneratorValue&>(&bernoulli_));

// bfloat16(self) -> Tensor
c.def("bfloat16", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bfloat16 with signature bfloat16(self) -> Tensor"); });

// bincount(self, weights Optional[Tensor]=None, minlength _int=0) -> Tensor
c.def("bincount", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bincount with signature bincount(self, weights Optional[Tensor]=None, minlength _int=0) -> Tensor"); });

// @overload bitwise_and(self, other Tensor) -> Tensor
// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("bitwise_and", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&bitwise_and));

// @overload bitwise_and_(self, other Tensor) -> Tensor
// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("bitwise_and_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&bitwise_and_));

// @overload bitwise_left_shift(self, other Tensor) -> Tensor
c.def("bitwise_left_shift", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bitwise_left_shift with signature @overload bitwise_left_shift(self, other Tensor) -> Tensor"); });

// @overload bitwise_left_shift(self, other Number) -> Tensor
c.def("bitwise_left_shift", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bitwise_left_shift with signature @overload bitwise_left_shift(self, other Number) -> Tensor"); });

// @overload bitwise_left_shift_(self, other Tensor) -> Tensor
c.def("bitwise_left_shift_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bitwise_left_shift_ with signature @overload bitwise_left_shift_(self, other Tensor) -> Tensor"); });

// @overload bitwise_left_shift_(self, other Number) -> Tensor
c.def("bitwise_left_shift_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bitwise_left_shift_ with signature @overload bitwise_left_shift_(self, other Number) -> Tensor"); });

// bitwise_not(self) -> Tensor
// aten::bitwise_not : (Tensor) -> (Tensor)
c.def("bitwise_not", py::overload_cast<const PyAnyTorchTensorValue&>(&bitwise_not));

// bitwise_not_(self) -> Tensor
// aten::bitwise_not_ : (Tensor) -> (Tensor)
c.def("bitwise_not_", py::overload_cast<const PyAnyTorchTensorValue&>(&bitwise_not_));

// @overload bitwise_or(self, other Tensor) -> Tensor
// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("bitwise_or", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&bitwise_or));

// @overload bitwise_or_(self, other Tensor) -> Tensor
// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("bitwise_or_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&bitwise_or_));

// @overload bitwise_right_shift(self, other Tensor) -> Tensor
c.def("bitwise_right_shift", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bitwise_right_shift with signature @overload bitwise_right_shift(self, other Tensor) -> Tensor"); });

// @overload bitwise_right_shift(self, other Number) -> Tensor
c.def("bitwise_right_shift", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bitwise_right_shift with signature @overload bitwise_right_shift(self, other Number) -> Tensor"); });

// @overload bitwise_right_shift_(self, other Tensor) -> Tensor
c.def("bitwise_right_shift_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bitwise_right_shift_ with signature @overload bitwise_right_shift_(self, other Tensor) -> Tensor"); });

// @overload bitwise_right_shift_(self, other Number) -> Tensor
c.def("bitwise_right_shift_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bitwise_right_shift_ with signature @overload bitwise_right_shift_(self, other Number) -> Tensor"); });

// @overload bitwise_xor(self, other Tensor) -> Tensor
// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("bitwise_xor", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&bitwise_xor));

// @overload bitwise_xor_(self, other Tensor) -> Tensor
// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("bitwise_xor_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&bitwise_xor_));

// bmm(self, mat2 Tensor) -> Tensor
// aten::bmm : (Tensor, Tensor) -> (Tensor)
c.def("bmm", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&bmm));

// bool(self) -> Tensor
c.def("bool", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("bool with signature bool(self) -> Tensor"); });

// @overload broadcast_to(self, size Sequence[Union[_int, SymInt]]) -> Tensor
// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
c.def("broadcast_to", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&>(&broadcast_to));

// byte(self) -> Tensor
c.def("byte", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("byte with signature byte(self) -> Tensor"); });

// cauchy_(self, median _float=0, sigma _float=1, *, generator Optional[Generator]=None) -> Tensor
c.def("cauchy_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cauchy_ with signature cauchy_(self, median _float=0, sigma _float=1, *, generator Optional[Generator]=None) -> Tensor"); });

// ccol_indices(self) -> Tensor
c.def("ccol_indices", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("ccol_indices with signature ccol_indices(self) -> Tensor"); });

// ceil(self) -> Tensor
// aten::ceil : (Tensor) -> (Tensor)
c.def("ceil", py::overload_cast<const PyAnyTorchTensorValue&>(&ceil));

// ceil_(self) -> Tensor
// aten::ceil_ : (Tensor) -> (Tensor)
c.def("ceil_", py::overload_cast<const PyAnyTorchTensorValue&>(&ceil_));

// chalf(self, *, memory_format Optional[memory_format]=None) -> Tensor
c.def("chalf", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("chalf with signature chalf(self, *, memory_format Optional[memory_format]=None) -> Tensor"); });

// char(self) -> Tensor
c.def("char", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("char with signature char(self) -> Tensor"); });

// cholesky(self, upper _bool=False) -> Tensor
c.def("cholesky", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cholesky with signature cholesky(self, upper _bool=False) -> Tensor"); });

// cholesky_inverse(self, upper _bool=False) -> Tensor
c.def("cholesky_inverse", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cholesky_inverse with signature cholesky_inverse(self, upper _bool=False) -> Tensor"); });

// cholesky_solve(self, input2 Tensor, upper _bool=False) -> Tensor
c.def("cholesky_solve", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cholesky_solve with signature cholesky_solve(self, input2 Tensor, upper _bool=False) -> Tensor"); });

// chunk(self, chunks _int, dim _int=0) -> List[Tensor]
c.def("chunk", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("chunk with signature chunk(self, chunks _int, dim _int=0) -> List[Tensor]"); });

// @overload clamp(self, min Optional[Tensor]=None, max Optional[Tensor]=None) -> Tensor
c.def("clamp", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("clamp with signature @overload clamp(self, min Optional[Tensor]=None, max Optional[Tensor]=None) -> Tensor"); });

// @overload clamp(self, min Optional[Number]=None, max Optional[Number]=None) -> Tensor
c.def("clamp", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("clamp with signature @overload clamp(self, min Optional[Number]=None, max Optional[Number]=None) -> Tensor"); });

// @overload clamp_(self, min Optional[Tensor]=None, max Optional[Tensor]=None) -> Tensor
c.def("clamp_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("clamp_ with signature @overload clamp_(self, min Optional[Tensor]=None, max Optional[Tensor]=None) -> Tensor"); });

// @overload clamp_(self, min Optional[Number]=None, max Optional[Number]=None) -> Tensor
c.def("clamp_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("clamp_ with signature @overload clamp_(self, min Optional[Number]=None, max Optional[Number]=None) -> Tensor"); });

// @overload clamp_max(self, max Tensor) -> Tensor
// aten::clamp_max : (Tensor, Scalar) -> (Tensor)
c.def("clamp_max", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&clamp_max));

// @overload clamp_max_(self, max Tensor) -> Tensor
// aten::clamp_max_ : (Tensor, Scalar) -> (Tensor)
c.def("clamp_max_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&clamp_max_));

// @overload clamp_min(self, min Tensor) -> Tensor
// aten::clamp_min : (Tensor, Scalar) -> (Tensor)
c.def("clamp_min", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&clamp_min));

// @overload clamp_min_(self, min Tensor) -> Tensor
// aten::clamp_min_ : (Tensor, Scalar) -> (Tensor)
c.def("clamp_min_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&clamp_min_));

// @overload clip(self, min Optional[Tensor]=None, max Optional[Tensor]=None) -> Tensor
c.def("clip", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("clip with signature @overload clip(self, min Optional[Tensor]=None, max Optional[Tensor]=None) -> Tensor"); });

// @overload clip(self, min Optional[Number]=None, max Optional[Number]=None) -> Tensor
c.def("clip", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("clip with signature @overload clip(self, min Optional[Number]=None, max Optional[Number]=None) -> Tensor"); });

// @overload clip_(self, min Optional[Tensor]=None, max Optional[Tensor]=None) -> Tensor
c.def("clip_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("clip_ with signature @overload clip_(self, min Optional[Tensor]=None, max Optional[Tensor]=None) -> Tensor"); });

// @overload clip_(self, min Optional[Number]=None, max Optional[Number]=None) -> Tensor
c.def("clip_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("clip_ with signature @overload clip_(self, min Optional[Number]=None, max Optional[Number]=None) -> Tensor"); });

// clone(self, *, memory_format Optional[memory_format]=None) -> Tensor
// aten::clone : (Tensor, int?) -> (Tensor)
c.def("clone", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchOptionalIntValue&>(&clone));

// coalesce(self) -> Tensor
c.def("coalesce", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("coalesce with signature coalesce(self) -> Tensor"); });

// col_indices(self) -> Tensor
c.def("col_indices", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("col_indices with signature col_indices(self) -> Tensor"); });

// conj(self) -> Tensor
c.def("conj", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("conj with signature conj(self) -> Tensor"); });

// conj_physical(self) -> Tensor
c.def("conj_physical", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("conj_physical with signature conj_physical(self) -> Tensor"); });

// conj_physical_(self) -> Tensor
c.def("conj_physical_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("conj_physical_ with signature conj_physical_(self) -> Tensor"); });

// contiguous(self, memory_format=torch.contiguous_format) -> Tensor
// aten::contiguous : (Tensor, int) -> (Tensor)
c.def("contiguous", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&>(&contiguous));

// copy_(self, src Tensor, non_blocking _bool=False) -> Tensor
// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
c.def("copy_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyTorch_BoolValue&>(&copy_));

// @overload copysign(self, other Tensor) -> Tensor
c.def("copysign", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("copysign with signature @overload copysign(self, other Tensor) -> Tensor"); });

// @overload copysign(self, other Number) -> Tensor
c.def("copysign", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("copysign with signature @overload copysign(self, other Number) -> Tensor"); });

// @overload copysign_(self, other Tensor) -> Tensor
c.def("copysign_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("copysign_ with signature @overload copysign_(self, other Tensor) -> Tensor"); });

// @overload copysign_(self, other Number) -> Tensor
c.def("copysign_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("copysign_ with signature @overload copysign_(self, other Number) -> Tensor"); });

// corrcoef(self) -> Tensor
c.def("corrcoef", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("corrcoef with signature corrcoef(self) -> Tensor"); });

// cos(self) -> Tensor
// aten::cos : (Tensor) -> (Tensor)
c.def("cos", py::overload_cast<const PyAnyTorchTensorValue&>(&cos));

// cos_(self) -> Tensor
// aten::cos_ : (Tensor) -> (Tensor)
c.def("cos_", py::overload_cast<const PyAnyTorchTensorValue&>(&cos_));

// cosh(self) -> Tensor
c.def("cosh", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cosh with signature cosh(self) -> Tensor"); });

// cosh_(self) -> Tensor
c.def("cosh_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cosh_ with signature cosh_(self) -> Tensor"); });

// @overload count_nonzero(self, dim Optional[_int]=None) -> Tensor
c.def("count_nonzero", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("count_nonzero with signature @overload count_nonzero(self, dim Optional[_int]=None) -> Tensor"); });

// @overload count_nonzero(self, dim _size) -> Tensor
c.def("count_nonzero", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("count_nonzero with signature @overload count_nonzero(self, dim _size) -> Tensor"); });

// @overload count_nonzero(self, *dim _int) -> Tensor
c.def("count_nonzero", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("count_nonzero with signature @overload count_nonzero(self, *dim _int) -> Tensor"); });

// cov(self, *, correction _int=1, fweights Optional[Tensor]=None, aweights Optional[Tensor]=None) -> Tensor
c.def("cov", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cov with signature cov(self, *, correction _int=1, fweights Optional[Tensor]=None, aweights Optional[Tensor]=None) -> Tensor"); });

// cpu(self) -> Tensor
// aten::cpu : (Tensor) -> (Tensor)
c.def("cpu", py::overload_cast<const PyAnyTorchTensorValue&>(&cpu));

// cross(self, other Tensor, dim Optional[_int]=None) -> Tensor
c.def("cross", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cross with signature cross(self, other Tensor, dim Optional[_int]=None) -> Tensor"); });

// crow_indices(self) -> Tensor
c.def("crow_indices", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("crow_indices with signature crow_indices(self) -> Tensor"); });

// cuda(self, device Optional[Union[_device, _int, str]]=None, non_blocking _bool=False) -> Tensor
c.def("cuda", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cuda with signature cuda(self, device Optional[Union[_device, _int, str]]=None, non_blocking _bool=False) -> Tensor"); });

// @overload cummax(self, dim _int) -> torch.return_types.cummax
c.def("cummax", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cummax with signature @overload cummax(self, dim _int) -> torch.return_types.cummax"); });

// @overload cummax(self, dim Union[str, ellipsis, None]) -> torch.return_types.cummax
c.def("cummax", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cummax with signature @overload cummax(self, dim Union[str, ellipsis, None]) -> torch.return_types.cummax"); });

// @overload cummin(self, dim _int) -> torch.return_types.cummin
c.def("cummin", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cummin with signature @overload cummin(self, dim _int) -> torch.return_types.cummin"); });

// @overload cummin(self, dim Union[str, ellipsis, None]) -> torch.return_types.cummin
c.def("cummin", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cummin with signature @overload cummin(self, dim Union[str, ellipsis, None]) -> torch.return_types.cummin"); });

// @overload cumprod(self, dim _int, *, dtype Optional[_dtype]=None) -> Tensor
c.def("cumprod", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cumprod with signature @overload cumprod(self, dim _int, *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload cumprod(self, dim Union[str, ellipsis, None], *, dtype Optional[_dtype]=None) -> Tensor
c.def("cumprod", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cumprod with signature @overload cumprod(self, dim Union[str, ellipsis, None], *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload cumprod_(self, dim _int, *, dtype Optional[_dtype]=None) -> Tensor
c.def("cumprod_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cumprod_ with signature @overload cumprod_(self, dim _int, *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload cumprod_(self, dim Union[str, ellipsis, None], *, dtype Optional[_dtype]=None) -> Tensor
c.def("cumprod_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cumprod_ with signature @overload cumprod_(self, dim Union[str, ellipsis, None], *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload cumsum(self, dim _int, *, dtype Optional[_dtype]=None) -> Tensor
// aten::cumsum : (Tensor, int, int?) -> (Tensor)
c.def("cumsum", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchOptionalIntValue&>(&cumsum));

// @overload cumsum_(self, dim _int, *, dtype Optional[_dtype]=None) -> Tensor
c.def("cumsum_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cumsum_ with signature @overload cumsum_(self, dim _int, *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload cumsum_(self, dim Union[str, ellipsis, None], *, dtype Optional[_dtype]=None) -> Tensor
c.def("cumsum_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("cumsum_ with signature @overload cumsum_(self, dim Union[str, ellipsis, None], *, dtype Optional[_dtype]=None) -> Tensor"); });

// data_ptr(self) -> _int
c.def("data_ptr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("data_ptr with signature data_ptr(self) -> _int"); });

// deg2rad(self) -> Tensor
c.def("deg2rad", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("deg2rad with signature deg2rad(self) -> Tensor"); });

// deg2rad_(self) -> Tensor
c.def("deg2rad_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("deg2rad_ with signature deg2rad_(self) -> Tensor"); });

// dense_dim(self) -> _int
c.def("dense_dim", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("dense_dim with signature dense_dim(self) -> _int"); });

// dequantize(self) -> Tensor
c.def("dequantize", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("dequantize with signature dequantize(self) -> Tensor"); });

// det(self) -> Tensor
c.def("det", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("det with signature det(self) -> Tensor"); });

// detach(self) -> Tensor
// aten::detach : (Tensor) -> (Tensor)
c.def("detach", py::overload_cast<const PyAnyTorchTensorValue&>(&detach));

// detach_(self) -> Tensor
c.def("detach_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("detach_ with signature detach_(self) -> Tensor"); });

// diag(self, diagonal _int=0) -> Tensor
c.def("diag", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("diag with signature diag(self, diagonal _int=0) -> Tensor"); });

// diag_embed(self, offset _int=0, dim1 _int=-2, dim2 _int=-1) -> Tensor
c.def("diag_embed", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("diag_embed with signature diag_embed(self, offset _int=0, dim1 _int=-2, dim2 _int=-1) -> Tensor"); });

// diagflat(self, offset _int=0) -> Tensor
c.def("diagflat", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("diagflat with signature diagflat(self, offset _int=0) -> Tensor"); });

// @overload diagonal(self, *, outdim Union[str, ellipsis, None], dim1 Union[str, ellipsis, None], dim2 Union[str, ellipsis, None], offset _int=0) -> Tensor
c.def("diagonal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("diagonal with signature @overload diagonal(self, *, outdim Union[str, ellipsis, None], dim1 Union[str, ellipsis, None], dim2 Union[str, ellipsis, None], offset _int=0) -> Tensor"); });

// @overload diagonal(self, offset _int=0, dim1 _int=0, dim2 _int=1) -> Tensor
c.def("diagonal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("diagonal with signature @overload diagonal(self, offset _int=0, dim1 _int=0, dim2 _int=1) -> Tensor"); });

// diagonal_scatter(self, src Tensor, offset _int=0, dim1 _int=0, dim2 _int=1) -> Tensor
// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
c.def("diagonal_scatter", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyTorch_IntValue&, const PyTorch_IntValue&>(&diagonal_scatter));

// diff(self, n _int=1, dim _int=-1, prepend Optional[Tensor]=None, append Optional[Tensor]=None) -> Tensor
c.def("diff", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("diff with signature diff(self, n _int=1, dim _int=-1, prepend Optional[Tensor]=None, append Optional[Tensor]=None) -> Tensor"); });

// digamma(self) -> Tensor
c.def("digamma", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("digamma with signature digamma(self) -> Tensor"); });

// digamma_(self) -> Tensor
c.def("digamma_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("digamma_ with signature digamma_(self) -> Tensor"); });

// dim(self) -> _int
// aten::dim : (Tensor) -> (int)
c.def("dim", py::overload_cast<const PyAnyTorchTensorValue&>(&dim));

// dist(self, other Tensor, p Number=2) -> Tensor
c.def("dist", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("dist with signature dist(self, other Tensor, p Number=2) -> Tensor"); });

// div(self, other Union[Tensor, Number], *, rounding_mode Optional[str]=None) -> Tensor
// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
c.def("div", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchOptionalStringValue&>(&div));

// div_(self, other Union[Tensor, Number], *, rounding_mode Optional[str]=None) -> Tensor
// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
c.def("div_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchOptionalStringValue&>(&div_));

// @overload divide(self, other Tensor) -> Tensor
c.def("divide", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("divide with signature @overload divide(self, other Tensor) -> Tensor"); });

// @overload divide(self, other Tensor, *, rounding_mode Optional[str]) -> Tensor
c.def("divide", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("divide with signature @overload divide(self, other Tensor, *, rounding_mode Optional[str]) -> Tensor"); });

// @overload divide(self, other Number, *, rounding_mode Optional[str]) -> Tensor
c.def("divide", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("divide with signature @overload divide(self, other Number, *, rounding_mode Optional[str]) -> Tensor"); });

// @overload divide(self, other Number) -> Tensor
c.def("divide", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("divide with signature @overload divide(self, other Number) -> Tensor"); });

// @overload divide_(self, other Tensor) -> Tensor
c.def("divide_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("divide_ with signature @overload divide_(self, other Tensor) -> Tensor"); });

// @overload divide_(self, other Tensor, *, rounding_mode Optional[str]) -> Tensor
c.def("divide_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("divide_ with signature @overload divide_(self, other Tensor, *, rounding_mode Optional[str]) -> Tensor"); });

// @overload divide_(self, other Number, *, rounding_mode Optional[str]) -> Tensor
c.def("divide_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("divide_ with signature @overload divide_(self, other Number, *, rounding_mode Optional[str]) -> Tensor"); });

// @overload divide_(self, other Number) -> Tensor
c.def("divide_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("divide_ with signature @overload divide_(self, other Number) -> Tensor"); });

// dot(self, tensor Tensor) -> Tensor
c.def("dot", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("dot with signature dot(self, tensor Tensor) -> Tensor"); });

// double(self) -> Tensor
c.def("double", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("double with signature double(self) -> Tensor"); });

// @overload dsplit(self, sections _int) -> List[Tensor]
c.def("dsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("dsplit with signature @overload dsplit(self, sections _int) -> List[Tensor]"); });

// @overload dsplit(self, indices _size) -> List[Tensor]
c.def("dsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("dsplit with signature @overload dsplit(self, indices _size) -> List[Tensor]"); });

// @overload dsplit(self, *indices _int) -> List[Tensor]
c.def("dsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("dsplit with signature @overload dsplit(self, *indices _int) -> List[Tensor]"); });

// element_size(self) -> _int
c.def("element_size", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("element_size with signature element_size(self) -> _int"); });

// @overload eq_(self, other Tensor) -> Tensor
// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("eq_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&eq_));

// equal(self, other Tensor) -> _bool
c.def("equal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("equal with signature equal(self, other Tensor) -> _bool"); });

// erf(self) -> Tensor
// aten::erf : (Tensor) -> (Tensor)
c.def("erf", py::overload_cast<const PyAnyTorchTensorValue&>(&erf));

// erf_(self) -> Tensor
// aten::erf_ : (Tensor) -> (Tensor)
c.def("erf_", py::overload_cast<const PyAnyTorchTensorValue&>(&erf_));

// erfc(self) -> Tensor
c.def("erfc", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("erfc with signature erfc(self) -> Tensor"); });

// erfc_(self) -> Tensor
c.def("erfc_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("erfc_ with signature erfc_(self) -> Tensor"); });

// erfinv(self) -> Tensor
c.def("erfinv", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("erfinv with signature erfinv(self) -> Tensor"); });

// erfinv_(self) -> Tensor
c.def("erfinv_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("erfinv_ with signature erfinv_(self) -> Tensor"); });

// exp(self) -> Tensor
// aten::exp : (Tensor) -> (Tensor)
c.def("exp", py::overload_cast<const PyAnyTorchTensorValue&>(&exp));

// exp2(self) -> Tensor
c.def("exp2", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("exp2 with signature exp2(self) -> Tensor"); });

// exp2_(self) -> Tensor
c.def("exp2_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("exp2_ with signature exp2_(self) -> Tensor"); });

// exp_(self) -> Tensor
// aten::exp_ : (Tensor) -> (Tensor)
c.def("exp_", py::overload_cast<const PyAnyTorchTensorValue&>(&exp_));

// @overload expand(self, size Sequence[Union[_int, SymInt]], *, implicit _bool=False) -> Tensor
// aten::expand : (Tensor, int[], bool) -> (Tensor)
c.def("expand", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&, const PyTorch_BoolValue&>(&expand));

// expand_as(self, other Tensor) -> Tensor
// aten::expand_as : (Tensor, Tensor) -> (Tensor)
c.def("expand_as", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&expand_as));

// expm1(self) -> Tensor
// aten::expm1 : (Tensor) -> (Tensor)
c.def("expm1", py::overload_cast<const PyAnyTorchTensorValue&>(&expm1));

// expm1_(self) -> Tensor
// aten::expm1_ : (Tensor) -> (Tensor)
c.def("expm1_", py::overload_cast<const PyAnyTorchTensorValue&>(&expm1_));

// exponential_(self, lambd _float=1, *, generator Optional[Generator]=None) -> Tensor
c.def("exponential_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("exponential_ with signature exponential_(self, lambd _float=1, *, generator Optional[Generator]=None) -> Tensor"); });

// @overload fill_(self, value Tensor) -> Tensor
// aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)
c.def("fill_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&fill_));

// fill_diagonal_(self, fill_value Number, wrap _bool=False) -> Tensor
c.def("fill_diagonal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("fill_diagonal_ with signature fill_diagonal_(self, fill_value Number, wrap _bool=False) -> Tensor"); });

// fix(self) -> Tensor
c.def("fix", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("fix with signature fix(self) -> Tensor"); });

// fix_(self) -> Tensor
c.def("fix_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("fix_ with signature fix_(self) -> Tensor"); });

// @overload flatten(self, start_dim _int=0, end_dim _int=-1) -> Tensor
// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
c.def("flatten", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyTorch_IntValue&>(&flatten));

// @overload flip(self, dims _size) -> Tensor
// aten::flip : (Tensor, int[]) -> (Tensor)
c.def("flip", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&>(&flip));

// fliplr(self) -> Tensor
c.def("fliplr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("fliplr with signature fliplr(self) -> Tensor"); });

// flipud(self) -> Tensor
c.def("flipud", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("flipud with signature flipud(self) -> Tensor"); });

// float(self) -> Tensor
c.def("float", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("float with signature float(self) -> Tensor"); });

// @overload float_power(self, exponent Tensor) -> Tensor
c.def("float_power", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("float_power with signature @overload float_power(self, exponent Tensor) -> Tensor"); });

// @overload float_power(self, exponent Number) -> Tensor
c.def("float_power", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("float_power with signature @overload float_power(self, exponent Number) -> Tensor"); });

// @overload float_power_(self, exponent Tensor) -> Tensor
c.def("float_power_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("float_power_ with signature @overload float_power_(self, exponent Tensor) -> Tensor"); });

// @overload float_power_(self, exponent Number) -> Tensor
c.def("float_power_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("float_power_ with signature @overload float_power_(self, exponent Number) -> Tensor"); });

// floor(self) -> Tensor
// aten::floor : (Tensor) -> (Tensor)
c.def("floor", py::overload_cast<const PyAnyTorchTensorValue&>(&floor));

// floor_(self) -> Tensor
// aten::floor_ : (Tensor) -> (Tensor)
c.def("floor_", py::overload_cast<const PyAnyTorchTensorValue&>(&floor_));

// floor_divide_(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat]) -> Tensor
c.def("floor_divide_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("floor_divide_ with signature floor_divide_(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat]) -> Tensor"); });

// fmax(self, other Tensor) -> Tensor
c.def("fmax", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("fmax with signature fmax(self, other Tensor) -> Tensor"); });

// fmin(self, other Tensor) -> Tensor
c.def("fmin", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("fmin with signature fmin(self, other Tensor) -> Tensor"); });

// @overload fmod(self, other Tensor) -> Tensor
// aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)
c.def("fmod", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&fmod));

// @overload fmod_(self, other Tensor) -> Tensor
// aten::fmod_.Scalar : (Tensor, Scalar) -> (Tensor)
c.def("fmod_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&fmod_));

// frac(self) -> Tensor
c.def("frac", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("frac with signature frac(self) -> Tensor"); });

// frac_(self) -> Tensor
c.def("frac_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("frac_ with signature frac_(self) -> Tensor"); });

// frexp(self) -> torch.return_types.frexp
c.def("frexp", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("frexp with signature frexp(self) -> torch.return_types.frexp"); });

// @overload gather(self, dim _int, index Tensor, *, sparse_grad _bool=False) -> Tensor
// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
c.def("gather", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchTensorValue&, const PyTorch_BoolValue&>(&gather));

// gcd(self, other Tensor) -> Tensor
c.def("gcd", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("gcd with signature gcd(self, other Tensor) -> Tensor"); });

// gcd_(self, other Tensor) -> Tensor
c.def("gcd_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("gcd_ with signature gcd_(self, other Tensor) -> Tensor"); });

// @overload ge_(self, other Tensor) -> Tensor
// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("ge_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&ge_));

// geometric_(self, p _float, *, generator Optional[Generator]=None) -> Tensor
c.def("geometric_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("geometric_ with signature geometric_(self, p _float, *, generator Optional[Generator]=None) -> Tensor"); });

// geqrf(self) -> torch.return_types.geqrf
c.def("geqrf", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("geqrf with signature geqrf(self) -> torch.return_types.geqrf"); });

// ger(self, vec2 Tensor) -> Tensor
c.def("ger", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("ger with signature ger(self, vec2 Tensor) -> Tensor"); });

// get_device(self) -> _int
c.def("get_device", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("get_device with signature get_device(self) -> _int"); });

// @overload greater(self, other Tensor) -> Tensor
c.def("greater", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("greater with signature @overload greater(self, other Tensor) -> Tensor"); });

// @overload greater(self, other Number) -> Tensor
c.def("greater", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("greater with signature @overload greater(self, other Number) -> Tensor"); });

// @overload greater_(self, other Tensor) -> Tensor
c.def("greater_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("greater_ with signature @overload greater_(self, other Tensor) -> Tensor"); });

// @overload greater_(self, other Number) -> Tensor
c.def("greater_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("greater_ with signature @overload greater_(self, other Number) -> Tensor"); });

// @overload greater_equal(self, other Tensor) -> Tensor
c.def("greater_equal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("greater_equal with signature @overload greater_equal(self, other Tensor) -> Tensor"); });

// @overload greater_equal(self, other Number) -> Tensor
c.def("greater_equal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("greater_equal with signature @overload greater_equal(self, other Number) -> Tensor"); });

// @overload greater_equal_(self, other Tensor) -> Tensor
c.def("greater_equal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("greater_equal_ with signature @overload greater_equal_(self, other Tensor) -> Tensor"); });

// @overload greater_equal_(self, other Number) -> Tensor
c.def("greater_equal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("greater_equal_ with signature @overload greater_equal_(self, other Number) -> Tensor"); });

// @overload gt_(self, other Tensor) -> Tensor
// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("gt_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&gt_));

// half(self) -> Tensor
c.def("half", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("half with signature half(self) -> Tensor"); });

// hardshrink(self, lambd Number=0.5) -> Tensor
c.def("hardshrink", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("hardshrink with signature hardshrink(self, lambd Number=0.5) -> Tensor"); });

// has_names(self) -> _bool
c.def("has_names", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("has_names with signature has_names(self) -> _bool"); });

// heaviside(self, values Tensor) -> Tensor
c.def("heaviside", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("heaviside with signature heaviside(self, values Tensor) -> Tensor"); });

// heaviside_(self, values Tensor) -> Tensor
c.def("heaviside_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("heaviside_ with signature heaviside_(self, values Tensor) -> Tensor"); });

// histc(self, bins _int=100, min Number=0, max Number=0) -> Tensor
c.def("histc", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("histc with signature histc(self, bins _int=100, min Number=0, max Number=0) -> Tensor"); });

// @overload histogram(self, bins Tensor, *, weight Optional[Tensor]=None, density _bool=False) -> torch.return_types.histogram
c.def("histogram", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("histogram with signature @overload histogram(self, bins Tensor, *, weight Optional[Tensor]=None, density _bool=False) -> torch.return_types.histogram"); });

// @overload histogram(self, bins _int=100, *, range Optional[Sequence[_float]]=None, weight Optional[Tensor]=None, density _bool=False) -> torch.return_types.histogram
c.def("histogram", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("histogram with signature @overload histogram(self, bins _int=100, *, range Optional[Sequence[_float]]=None, weight Optional[Tensor]=None, density _bool=False) -> torch.return_types.histogram"); });

// @overload hsplit(self, sections _int) -> List[Tensor]
c.def("hsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("hsplit with signature @overload hsplit(self, sections _int) -> List[Tensor]"); });

// @overload hsplit(self, indices _size) -> List[Tensor]
c.def("hsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("hsplit with signature @overload hsplit(self, indices _size) -> List[Tensor]"); });

// @overload hsplit(self, *indices _int) -> List[Tensor]
c.def("hsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("hsplit with signature @overload hsplit(self, *indices _int) -> List[Tensor]"); });

// hypot(self, other Tensor) -> Tensor
c.def("hypot", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("hypot with signature hypot(self, other Tensor) -> Tensor"); });

// hypot_(self, other Tensor) -> Tensor
c.def("hypot_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("hypot_ with signature hypot_(self, other Tensor) -> Tensor"); });

// i0(self) -> Tensor
c.def("i0", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("i0 with signature i0(self) -> Tensor"); });

// i0_(self) -> Tensor
c.def("i0_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("i0_ with signature i0_(self) -> Tensor"); });

// igamma(self, other Tensor) -> Tensor
c.def("igamma", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("igamma with signature igamma(self, other Tensor) -> Tensor"); });

// igamma_(self, other Tensor) -> Tensor
c.def("igamma_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("igamma_ with signature igamma_(self, other Tensor) -> Tensor"); });

// igammac(self, other Tensor) -> Tensor
c.def("igammac", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("igammac with signature igammac(self, other Tensor) -> Tensor"); });

// igammac_(self, other Tensor) -> Tensor
c.def("igammac_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("igammac_ with signature igammac_(self, other Tensor) -> Tensor"); });

// @overload index_add(self, dim _int, index Tensor, source Tensor, *, alpha Number=1) -> Tensor
c.def("index_add", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_add with signature @overload index_add(self, dim _int, index Tensor, source Tensor, *, alpha Number=1) -> Tensor"); });

// @overload index_add(self, dim Union[str, ellipsis, None], index Tensor, source Tensor, *, alpha Number=1) -> Tensor
c.def("index_add", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_add with signature @overload index_add(self, dim Union[str, ellipsis, None], index Tensor, source Tensor, *, alpha Number=1) -> Tensor"); });

// index_add_(self, dim _int, index Tensor, source Tensor, *, alpha Number=1) -> Tensor
c.def("index_add_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_add_ with signature index_add_(self, dim _int, index Tensor, source Tensor, *, alpha Number=1) -> Tensor"); });

// @overload index_copy(self, dim _int, index Tensor, source Tensor) -> Tensor
c.def("index_copy", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_copy with signature @overload index_copy(self, dim _int, index Tensor, source Tensor) -> Tensor"); });

// @overload index_copy(self, dim Union[str, ellipsis, None], index Tensor, source Tensor) -> Tensor
c.def("index_copy", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_copy with signature @overload index_copy(self, dim Union[str, ellipsis, None], index Tensor, source Tensor) -> Tensor"); });

// @overload index_copy_(self, dim _int, index Tensor, source Tensor) -> Tensor
c.def("index_copy_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_copy_ with signature @overload index_copy_(self, dim _int, index Tensor, source Tensor) -> Tensor"); });

// @overload index_copy_(self, dim Union[str, ellipsis, None], index Tensor, source Tensor) -> Tensor
c.def("index_copy_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_copy_ with signature @overload index_copy_(self, dim Union[str, ellipsis, None], index Tensor, source Tensor) -> Tensor"); });

// @overload index_fill(self, dim _int, index Tensor, value Tensor) -> Tensor
c.def("index_fill", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_fill with signature @overload index_fill(self, dim _int, index Tensor, value Tensor) -> Tensor"); });

// @overload index_fill(self, dim Union[str, ellipsis, None], index Tensor, value Tensor) -> Tensor
c.def("index_fill", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_fill with signature @overload index_fill(self, dim Union[str, ellipsis, None], index Tensor, value Tensor) -> Tensor"); });

// @overload index_fill(self, dim _int, index Tensor, value Number) -> Tensor
c.def("index_fill", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_fill with signature @overload index_fill(self, dim _int, index Tensor, value Number) -> Tensor"); });

// @overload index_fill(self, dim Union[str, ellipsis, None], index Tensor, value Number) -> Tensor
c.def("index_fill", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_fill with signature @overload index_fill(self, dim Union[str, ellipsis, None], index Tensor, value Number) -> Tensor"); });

// @overload index_fill_(self, dim _int, index Tensor, value Tensor) -> Tensor
c.def("index_fill_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_fill_ with signature @overload index_fill_(self, dim _int, index Tensor, value Tensor) -> Tensor"); });

// @overload index_fill_(self, dim Union[str, ellipsis, None], index Tensor, value Tensor) -> Tensor
c.def("index_fill_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_fill_ with signature @overload index_fill_(self, dim Union[str, ellipsis, None], index Tensor, value Tensor) -> Tensor"); });

// @overload index_fill_(self, dim _int, index Tensor, value Number) -> Tensor
c.def("index_fill_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_fill_ with signature @overload index_fill_(self, dim _int, index Tensor, value Number) -> Tensor"); });

// @overload index_fill_(self, dim Union[str, ellipsis, None], index Tensor, value Number) -> Tensor
c.def("index_fill_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_fill_ with signature @overload index_fill_(self, dim Union[str, ellipsis, None], index Tensor, value Number) -> Tensor"); });

// index_put(self, indices Optional[Union[Tuple[Tensor, ], List[Tensor]]], values Tensor, accumulate _bool=False) -> Tensor
c.def("index_put", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_put with signature index_put(self, indices Optional[Union[Tuple[Tensor, ], List[Tensor]]], values Tensor, accumulate _bool=False) -> Tensor"); });

// index_put_(self, indices Optional[Union[Tuple[Tensor, ], List[Tensor]]], values Tensor, accumulate _bool=False) -> Tensor
c.def("index_put_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_put_ with signature index_put_(self, indices Optional[Union[Tuple[Tensor, ], List[Tensor]]], values Tensor, accumulate _bool=False) -> Tensor"); });

// index_reduce(self, dim _int, index Tensor, source Tensor, reduce str, *, include_self _bool=True) -> Tensor
c.def("index_reduce", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_reduce with signature index_reduce(self, dim _int, index Tensor, source Tensor, reduce str, *, include_self _bool=True) -> Tensor"); });

// index_reduce_(self, dim _int, index Tensor, source Tensor, reduce str, *, include_self _bool=True) -> Tensor
c.def("index_reduce_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("index_reduce_ with signature index_reduce_(self, dim _int, index Tensor, source Tensor, reduce str, *, include_self _bool=True) -> Tensor"); });

// @overload index_select(self, dim _int, index Tensor) -> Tensor
// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
c.def("index_select", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchTensorValue&>(&index_select));

// indices(self) -> Tensor
c.def("indices", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("indices with signature indices(self) -> Tensor"); });

// inner(self, other Tensor) -> Tensor
c.def("inner", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("inner with signature inner(self, other Tensor) -> Tensor"); });

// int(self) -> Tensor
c.def("int", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("int with signature int(self) -> Tensor"); });

// int_repr(self) -> Tensor
c.def("int_repr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("int_repr with signature int_repr(self) -> Tensor"); });

// inverse(self) -> Tensor
c.def("inverse", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("inverse with signature inverse(self) -> Tensor"); });

// is_coalesced(self) -> _bool
c.def("is_coalesced", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_coalesced with signature is_coalesced(self) -> _bool"); });

// is_complex(self) -> _bool
c.def("is_complex", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_complex with signature is_complex(self) -> _bool"); });

// is_conj(self) -> _bool
c.def("is_conj", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_conj with signature is_conj(self) -> _bool"); });

// is_contiguous(self, memory_format=torch.contiguous_format) -> _bool
c.def("is_contiguous", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_contiguous with signature is_contiguous(self, memory_format=torch.contiguous_format) -> _bool"); });

// is_distributed(self) -> _bool
c.def("is_distributed", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_distributed with signature is_distributed(self) -> _bool"); });

// is_floating_point(self) -> _bool
// aten::is_floating_point : (Tensor) -> (bool)
c.def("is_floating_point", py::overload_cast<const PyAnyTorchTensorValue&>(&is_floating_point));

// is_inference(self) -> _bool
c.def("is_inference", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_inference with signature is_inference(self) -> _bool"); });

// is_neg(self) -> _bool
c.def("is_neg", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_neg with signature is_neg(self) -> _bool"); });

// is_nonzero(self) -> _bool
c.def("is_nonzero", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_nonzero with signature is_nonzero(self) -> _bool"); });

// is_pinned(self, device Optional[Union[_device, str, None]]=None) -> _bool
c.def("is_pinned", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_pinned with signature is_pinned(self, device Optional[Union[_device, str, None]]=None) -> _bool"); });

// is_same_size(self, other Tensor) -> _bool
c.def("is_same_size", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_same_size with signature is_same_size(self, other Tensor) -> _bool"); });

// is_set_to(self, tensor Tensor) -> _bool
c.def("is_set_to", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_set_to with signature is_set_to(self, tensor Tensor) -> _bool"); });

// is_signed(self) -> _bool
c.def("is_signed", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("is_signed with signature is_signed(self) -> _bool"); });

// isclose(self, other Tensor, rtol _float=1e-05, atol _float=1e-08, equal_nan _bool=False) -> Tensor
c.def("isclose", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("isclose with signature isclose(self, other Tensor, rtol _float=1e-05, atol _float=1e-08, equal_nan _bool=False) -> Tensor"); });

// isfinite(self) -> Tensor
c.def("isfinite", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("isfinite with signature isfinite(self) -> Tensor"); });

// isinf(self) -> Tensor
c.def("isinf", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("isinf with signature isinf(self) -> Tensor"); });

// isnan(self) -> Tensor
c.def("isnan", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("isnan with signature isnan(self) -> Tensor"); });

// isneginf(self) -> Tensor
c.def("isneginf", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("isneginf with signature isneginf(self) -> Tensor"); });

// isposinf(self) -> Tensor
c.def("isposinf", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("isposinf with signature isposinf(self) -> Tensor"); });

// isreal(self) -> Tensor
c.def("isreal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("isreal with signature isreal(self) -> Tensor"); });

// istft(self, n_fft _int, hop_length Optional[_int]=None, win_length Optional[_int]=None, window Optional[Tensor]=None, center _bool=True, normalized _bool=False, onesided Optional[_bool]=None, length Optional[_int]=None, return_complex _bool=False) -> Tensor
c.def("istft", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("istft with signature istft(self, n_fft _int, hop_length Optional[_int]=None, win_length Optional[_int]=None, window Optional[Tensor]=None, center _bool=True, normalized _bool=False, onesided Optional[_bool]=None, length Optional[_int]=None, return_complex _bool=False) -> Tensor"); });

// item(self) -> Number
c.def("item", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("item with signature item(self) -> Number"); });

// kron(self, other Tensor) -> Tensor
c.def("kron", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("kron with signature kron(self, other Tensor) -> Tensor"); });

// @overload kthvalue(self, k _int, dim _int=-1, keepdim _bool=False) -> torch.return_types.kthvalue
c.def("kthvalue", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("kthvalue with signature @overload kthvalue(self, k _int, dim _int=-1, keepdim _bool=False) -> torch.return_types.kthvalue"); });

// @overload kthvalue(self, k _int, dim Union[str, ellipsis, None], keepdim _bool=False) -> torch.return_types.kthvalue
c.def("kthvalue", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("kthvalue with signature @overload kthvalue(self, k _int, dim Union[str, ellipsis, None], keepdim _bool=False) -> torch.return_types.kthvalue"); });

// lcm(self, other Tensor) -> Tensor
c.def("lcm", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("lcm with signature lcm(self, other Tensor) -> Tensor"); });

// lcm_(self, other Tensor) -> Tensor
c.def("lcm_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("lcm_ with signature lcm_(self, other Tensor) -> Tensor"); });

// ldexp(self, other Tensor) -> Tensor
c.def("ldexp", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("ldexp with signature ldexp(self, other Tensor) -> Tensor"); });

// ldexp_(self, other Tensor) -> Tensor
c.def("ldexp_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("ldexp_ with signature ldexp_(self, other Tensor) -> Tensor"); });

// @overload le_(self, other Tensor) -> Tensor
// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("le_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&le_));

// @overload lerp(self, end Tensor, weight Tensor) -> Tensor
// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
c.def("lerp", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&lerp));

// @overload lerp_(self, end Tensor, weight Tensor) -> Tensor
// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
c.def("lerp_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&lerp_));

// @overload less(self, other Tensor) -> Tensor
c.def("less", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("less with signature @overload less(self, other Tensor) -> Tensor"); });

// @overload less(self, other Number) -> Tensor
c.def("less", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("less with signature @overload less(self, other Number) -> Tensor"); });

// @overload less_(self, other Tensor) -> Tensor
c.def("less_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("less_ with signature @overload less_(self, other Tensor) -> Tensor"); });

// @overload less_(self, other Number) -> Tensor
c.def("less_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("less_ with signature @overload less_(self, other Number) -> Tensor"); });

// @overload less_equal(self, other Tensor) -> Tensor
c.def("less_equal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("less_equal with signature @overload less_equal(self, other Tensor) -> Tensor"); });

// @overload less_equal(self, other Number) -> Tensor
c.def("less_equal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("less_equal with signature @overload less_equal(self, other Number) -> Tensor"); });

// @overload less_equal_(self, other Tensor) -> Tensor
c.def("less_equal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("less_equal_ with signature @overload less_equal_(self, other Tensor) -> Tensor"); });

// @overload less_equal_(self, other Number) -> Tensor
c.def("less_equal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("less_equal_ with signature @overload less_equal_(self, other Number) -> Tensor"); });

// lgamma(self) -> Tensor
c.def("lgamma", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("lgamma with signature lgamma(self) -> Tensor"); });

// lgamma_(self) -> Tensor
c.def("lgamma_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("lgamma_ with signature lgamma_(self) -> Tensor"); });

// log(self) -> Tensor
// aten::log : (Tensor) -> (Tensor)
c.def("log", py::overload_cast<const PyAnyTorchTensorValue&>(&log));

// log10(self) -> Tensor
c.def("log10", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("log10 with signature log10(self) -> Tensor"); });

// log10_(self) -> Tensor
c.def("log10_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("log10_ with signature log10_(self) -> Tensor"); });

// log1p(self) -> Tensor
// aten::log1p : (Tensor) -> (Tensor)
c.def("log1p", py::overload_cast<const PyAnyTorchTensorValue&>(&log1p));

// log1p_(self) -> Tensor
// aten::log1p_ : (Tensor) -> (Tensor)
c.def("log1p_", py::overload_cast<const PyAnyTorchTensorValue&>(&log1p_));

// log2(self) -> Tensor
// aten::log2 : (Tensor) -> (Tensor)
c.def("log2", py::overload_cast<const PyAnyTorchTensorValue&>(&log2));

// log2_(self) -> Tensor
// aten::log2_ : (Tensor) -> (Tensor)
c.def("log2_", py::overload_cast<const PyAnyTorchTensorValue&>(&log2_));

// log_(self) -> Tensor
// aten::log_ : (Tensor) -> (Tensor)
c.def("log_", py::overload_cast<const PyAnyTorchTensorValue&>(&log_));

// log_normal_(self, mean _float=1, std _float=2, *, generator Optional[Generator]=None) -> Tensor
c.def("log_normal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("log_normal_ with signature log_normal_(self, mean _float=1, std _float=2, *, generator Optional[Generator]=None) -> Tensor"); });

// @overload log_softmax(self, dim _int, dtype Optional[_dtype]=None) -> Tensor
// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
c.def("log_softmax", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchOptionalIntValue&>(&log_softmax));

// logaddexp(self, other Tensor) -> Tensor
c.def("logaddexp", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("logaddexp with signature logaddexp(self, other Tensor) -> Tensor"); });

// logaddexp2(self, other Tensor) -> Tensor
c.def("logaddexp2", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("logaddexp2 with signature logaddexp2(self, other Tensor) -> Tensor"); });

// @overload logcumsumexp(self, dim _int) -> Tensor
c.def("logcumsumexp", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("logcumsumexp with signature @overload logcumsumexp(self, dim _int) -> Tensor"); });

// @overload logcumsumexp(self, dim Union[str, ellipsis, None]) -> Tensor
c.def("logcumsumexp", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("logcumsumexp with signature @overload logcumsumexp(self, dim Union[str, ellipsis, None]) -> Tensor"); });

// logdet(self) -> Tensor
c.def("logdet", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("logdet with signature logdet(self) -> Tensor"); });

// logical_and(self, other Tensor) -> Tensor
// aten::logical_and : (Tensor, Tensor) -> (Tensor)
c.def("logical_and", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&logical_and));

// logical_and_(self, other Tensor) -> Tensor
// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
c.def("logical_and_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&logical_and_));

// logical_not(self) -> Tensor
// aten::logical_not : (Tensor) -> (Tensor)
c.def("logical_not", py::overload_cast<const PyAnyTorchTensorValue&>(&logical_not));

// logical_not_(self) -> Tensor
// aten::logical_not_ : (Tensor) -> (Tensor)
c.def("logical_not_", py::overload_cast<const PyAnyTorchTensorValue&>(&logical_not_));

// logical_or(self, other Tensor) -> Tensor
// aten::logical_or : (Tensor, Tensor) -> (Tensor)
c.def("logical_or", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&logical_or));

// logical_or_(self, other Tensor) -> Tensor
// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
c.def("logical_or_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&logical_or_));

// logical_xor(self, other Tensor) -> Tensor
// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
c.def("logical_xor", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&logical_xor));

// logical_xor_(self, other Tensor) -> Tensor
// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
c.def("logical_xor_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&logical_xor_));

// logit(self, eps Optional[_float]=None) -> Tensor
c.def("logit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("logit with signature logit(self, eps Optional[_float]=None) -> Tensor"); });

// logit_(self, eps Optional[_float]=None) -> Tensor
c.def("logit_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("logit_ with signature logit_(self, eps Optional[_float]=None) -> Tensor"); });

// @overload logsumexp(self, dim Union[_int, _size], keepdim _bool=False) -> Tensor
// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
c.def("logsumexp", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&, const PyTorch_BoolValue&>(&logsumexp));

// long(self) -> Tensor
c.def("long", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("long with signature long(self) -> Tensor"); });

// @overload lt_(self, other Tensor) -> Tensor
// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("lt_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&lt_));

// lu_solve(self, LU_data Tensor, LU_pivots Tensor) -> Tensor
c.def("lu_solve", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("lu_solve with signature lu_solve(self, LU_data Tensor, LU_pivots Tensor) -> Tensor"); });

// map2_(self, x Tensor, y Tensor, callable Callable) -> Tensor
c.def("map2_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("map2_ with signature map2_(self, x Tensor, y Tensor, callable Callable) -> Tensor"); });

// map_(self, tensor Tensor, callable Callable) -> Tensor
c.def("map_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("map_ with signature map_(self, tensor Tensor, callable Callable) -> Tensor"); });

// @overload masked_fill(self, mask Tensor, value Tensor) -> Tensor
// aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
c.def("masked_fill", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&masked_fill));

// @overload masked_fill_(self, mask Tensor, value Tensor) -> Tensor
// aten::masked_fill_.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
c.def("masked_fill_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&masked_fill_));

// masked_scatter(self, mask Tensor, source Tensor) -> Tensor
c.def("masked_scatter", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("masked_scatter with signature masked_scatter(self, mask Tensor, source Tensor) -> Tensor"); });

// masked_scatter_(self, mask Tensor, source Tensor) -> Tensor
c.def("masked_scatter_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("masked_scatter_ with signature masked_scatter_(self, mask Tensor, source Tensor) -> Tensor"); });

// masked_select(self, mask Tensor) -> Tensor
// aten::masked_select : (Tensor, Tensor) -> (Tensor)
c.def("masked_select", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&masked_select));

// matrix_exp(self) -> Tensor
c.def("matrix_exp", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("matrix_exp with signature matrix_exp(self) -> Tensor"); });

// matrix_power(self, n _int) -> Tensor
c.def("matrix_power", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("matrix_power with signature matrix_power(self, n _int) -> Tensor"); });

// @overload max(self) -> Tensor
// aten::max : (Tensor) -> (Tensor)
c.def("max", py::overload_cast<const PyAnyTorchTensorValue&>(&max));

// maximum(self, other Tensor) -> Tensor
// aten::maximum : (Tensor, Tensor) -> (Tensor)
c.def("maximum", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&maximum));

// @overload mean(self, *, dtype Optional[_dtype]=None) -> Tensor
// aten::mean : (Tensor, int?) -> (Tensor)
c.def("mean", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchOptionalIntValue&>(&mean));

// @overload median(self) -> Tensor
c.def("median", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("median with signature @overload median(self) -> Tensor"); });

// @overload median(self, dim _int, keepdim _bool=False) -> torch.return_types.median
c.def("median", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("median with signature @overload median(self, dim _int, keepdim _bool=False) -> torch.return_types.median"); });

// @overload median(self, dim Union[str, ellipsis, None], keepdim _bool=False) -> torch.return_types.median
c.def("median", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("median with signature @overload median(self, dim Union[str, ellipsis, None], keepdim _bool=False) -> torch.return_types.median"); });

// minimum(self, other Tensor) -> Tensor
// aten::minimum : (Tensor, Tensor) -> (Tensor)
c.def("minimum", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&minimum));

// mm(self, mat2 Tensor) -> Tensor
// aten::mm : (Tensor, Tensor) -> (Tensor)
c.def("mm", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&mm));

// @overload mode(self, dim _int=-1, keepdim _bool=False) -> torch.return_types.mode
c.def("mode", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("mode with signature @overload mode(self, dim _int=-1, keepdim _bool=False) -> torch.return_types.mode"); });

// @overload mode(self, dim Union[str, ellipsis, None], keepdim _bool=False) -> torch.return_types.mode
c.def("mode", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("mode with signature @overload mode(self, dim Union[str, ellipsis, None], keepdim _bool=False) -> torch.return_types.mode"); });

// @overload moveaxis(self, source _int, destination _int) -> Tensor
c.def("moveaxis", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("moveaxis with signature @overload moveaxis(self, source _int, destination _int) -> Tensor"); });

// @overload moveaxis(self, source _size, destination _size) -> Tensor
c.def("moveaxis", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("moveaxis with signature @overload moveaxis(self, source _size, destination _size) -> Tensor"); });

// @overload movedim(self, source _int, destination _int) -> Tensor
c.def("movedim", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("movedim with signature @overload movedim(self, source _int, destination _int) -> Tensor"); });

// @overload movedim(self, source _size, destination _size) -> Tensor
c.def("movedim", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("movedim with signature @overload movedim(self, source _size, destination _size) -> Tensor"); });

// msort(self) -> Tensor
c.def("msort", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("msort with signature msort(self) -> Tensor"); });

// mul_(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat]) -> Tensor
// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("mul_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&mul_));

// multinomial(self, num_samples _int, replacement _bool=False, *, generator Optional[Generator]=None) -> Tensor
c.def("multinomial", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("multinomial with signature multinomial(self, num_samples _int, replacement _bool=False, *, generator Optional[Generator]=None) -> Tensor"); });

// @overload multiply(self, other Tensor) -> Tensor
c.def("multiply", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("multiply with signature @overload multiply(self, other Tensor) -> Tensor"); });

// @overload multiply(self, other Number) -> Tensor
c.def("multiply", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("multiply with signature @overload multiply(self, other Number) -> Tensor"); });

// @overload multiply_(self, other Tensor) -> Tensor
c.def("multiply_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("multiply_ with signature @overload multiply_(self, other Tensor) -> Tensor"); });

// @overload multiply_(self, other Number) -> Tensor
c.def("multiply_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("multiply_ with signature @overload multiply_(self, other Number) -> Tensor"); });

// mv(self, vec Tensor) -> Tensor
// aten::mv : (Tensor, Tensor) -> (Tensor)
c.def("mv", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&mv));

// mvlgamma(self, p _int) -> Tensor
c.def("mvlgamma", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("mvlgamma with signature mvlgamma(self, p _int) -> Tensor"); });

// mvlgamma_(self, p _int) -> Tensor
c.def("mvlgamma_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("mvlgamma_ with signature mvlgamma_(self, p _int) -> Tensor"); });

// nan_to_num(self, nan Optional[_float]=None, posinf Optional[_float]=None, neginf Optional[_float]=None) -> Tensor
c.def("nan_to_num", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nan_to_num with signature nan_to_num(self, nan Optional[_float]=None, posinf Optional[_float]=None, neginf Optional[_float]=None) -> Tensor"); });

// nan_to_num_(self, nan Optional[_float]=None, posinf Optional[_float]=None, neginf Optional[_float]=None) -> Tensor
c.def("nan_to_num_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nan_to_num_ with signature nan_to_num_(self, nan Optional[_float]=None, posinf Optional[_float]=None, neginf Optional[_float]=None) -> Tensor"); });

// nanmean(self, dim Optional[Union[_int, _size]]=None, keepdim _bool=False, *, dtype Optional[_dtype]=None) -> Tensor
c.def("nanmean", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nanmean with signature nanmean(self, dim Optional[Union[_int, _size]]=None, keepdim _bool=False, *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload nanmedian(self) -> Tensor
c.def("nanmedian", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nanmedian with signature @overload nanmedian(self) -> Tensor"); });

// @overload nanmedian(self, dim _int, keepdim _bool=False) -> torch.return_types.nanmedian
c.def("nanmedian", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nanmedian with signature @overload nanmedian(self, dim _int, keepdim _bool=False) -> torch.return_types.nanmedian"); });

// @overload nanmedian(self, dim Union[str, ellipsis, None], keepdim _bool=False) -> torch.return_types.nanmedian
c.def("nanmedian", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nanmedian with signature @overload nanmedian(self, dim Union[str, ellipsis, None], keepdim _bool=False) -> torch.return_types.nanmedian"); });

// @overload nanquantile(self, q Tensor, dim Optional[_int]=None, keepdim _bool=False, *, interpolation str='linear') -> Tensor
c.def("nanquantile", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nanquantile with signature @overload nanquantile(self, q Tensor, dim Optional[_int]=None, keepdim _bool=False, *, interpolation str='linear') -> Tensor"); });

// @overload nanquantile(self, q _float, dim Optional[_int]=None, keepdim _bool=False, *, interpolation str='linear') -> Tensor
c.def("nanquantile", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nanquantile with signature @overload nanquantile(self, q _float, dim Optional[_int]=None, keepdim _bool=False, *, interpolation str='linear') -> Tensor"); });

// nansum(self, dim Optional[Union[_int, _size]]=None, keepdim _bool=False, *, dtype Optional[_dtype]=None) -> Tensor
c.def("nansum", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nansum with signature nansum(self, dim Optional[Union[_int, _size]]=None, keepdim _bool=False, *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload narrow(self, dim _int, start Tensor, length Union[_int, SymInt]) -> Tensor
// aten::narrow : (Tensor, int, int, int) -> (Tensor)
c.def("narrow", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyTorch_IntValue&, const PyTorch_IntValue&>(&narrow));

// narrow_copy(self, dim _int, start Union[_int, SymInt], length Union[_int, SymInt]) -> Tensor
c.def("narrow_copy", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("narrow_copy with signature narrow_copy(self, dim _int, start Union[_int, SymInt], length Union[_int, SymInt]) -> Tensor"); });

// ndimension(self) -> _int
c.def("ndimension", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("ndimension with signature ndimension(self) -> _int"); });

// @overload ne_(self, other Tensor) -> Tensor
// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
c.def("ne_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&ne_));

// neg_(self) -> Tensor
// aten::neg_ : (Tensor) -> (Tensor)
c.def("neg_", py::overload_cast<const PyAnyTorchTensorValue&>(&neg_));

// negative(self) -> Tensor
c.def("negative", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("negative with signature negative(self) -> Tensor"); });

// negative_(self) -> Tensor
c.def("negative_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("negative_ with signature negative_(self) -> Tensor"); });

// nelement(self) -> _int
c.def("nelement", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nelement with signature nelement(self) -> _int"); });

// @overload new(self, *args Any, device Device=None) -> Tensor
c.def("new", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("new with signature @overload new(self, *args Any, device Device=None) -> Tensor"); });

// @overload new(self, storage Storage) -> Tensor
c.def("new", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("new with signature @overload new(self, storage Storage) -> Tensor"); });

// @overload new(self, other Tensor) -> Tensor
c.def("new", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("new with signature @overload new(self, other Tensor) -> Tensor"); });

// @overload new(self, size _size, *, device Device=None) -> Tensor
c.def("new", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("new with signature @overload new(self, size _size, *, device Device=None) -> Tensor"); });

// new_full(self, size Sequence[Union[_int, SymInt]], fill_value Number, *, dtype Optional[_dtype]=None, layout Optional[_layout]=None, device Optional[Union[_device, str, None]]=None, pin_memory Optional[_bool]=False, requires_grad Optional[_bool]=False) -> Tensor
c.def("new_full", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("new_full with signature new_full(self, size Sequence[Union[_int, SymInt]], fill_value Number, *, dtype Optional[_dtype]=None, layout Optional[_layout]=None, device Optional[Union[_device, str, None]]=None, pin_memory Optional[_bool]=False, requires_grad Optional[_bool]=False) -> Tensor"); });

// new_tensor(self, data Any, dtype Optional[_dtype]=None, device Device=None, requires_grad _bool=False) -> Tensor
c.def("new_tensor", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("new_tensor with signature new_tensor(self, data Any, dtype Optional[_dtype]=None, device Device=None, requires_grad _bool=False) -> Tensor"); });

// nextafter(self, other Tensor) -> Tensor
c.def("nextafter", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nextafter with signature nextafter(self, other Tensor) -> Tensor"); });

// nextafter_(self, other Tensor) -> Tensor
c.def("nextafter_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nextafter_ with signature nextafter_(self, other Tensor) -> Tensor"); });

// @overload nonzero(self, *, as_tuple Literal[False]=False) -> Tensor
c.def("nonzero", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nonzero with signature @overload nonzero(self, *, as_tuple Literal[False]=False) -> Tensor"); });

// @overload nonzero(self, *, as_tuple Literal[True]) -> Tuple[Tensor, ]
c.def("nonzero", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nonzero with signature @overload nonzero(self, *, as_tuple Literal[True]) -> Tuple[Tensor, ]"); });

// nonzero_static(self, *, size _int, fill_value _int=-1) -> Tensor
c.def("nonzero_static", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("nonzero_static with signature nonzero_static(self, *, size _int, fill_value _int=-1) -> Tensor"); });

// normal_(self, mean _float=0, std _float=1, *, generator Optional[Generator]=None) -> Tensor
c.def("normal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("normal_ with signature normal_(self, mean _float=0, std _float=1, *, generator Optional[Generator]=None) -> Tensor"); });

// @overload not_equal(self, other Tensor) -> Tensor
c.def("not_equal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("not_equal with signature @overload not_equal(self, other Tensor) -> Tensor"); });

// @overload not_equal(self, other Number) -> Tensor
c.def("not_equal", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("not_equal with signature @overload not_equal(self, other Number) -> Tensor"); });

// @overload not_equal_(self, other Tensor) -> Tensor
c.def("not_equal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("not_equal_ with signature @overload not_equal_(self, other Tensor) -> Tensor"); });

// @overload not_equal_(self, other Number) -> Tensor
c.def("not_equal_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("not_equal_ with signature @overload not_equal_(self, other Number) -> Tensor"); });

// numel(self) -> _int
// aten::numel : (Tensor) -> (int)
c.def("numel", py::overload_cast<const PyAnyTorchTensorValue&>(&numel));

// numpy(self, *, force _bool=False) -> Any
c.def("numpy", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("numpy with signature numpy(self, *, force _bool=False) -> Any"); });

// orgqr(self, input2 Tensor) -> Tensor
c.def("orgqr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("orgqr with signature orgqr(self, input2 Tensor) -> Tensor"); });

// ormqr(self, input2 Tensor, input3 Tensor, left _bool=True, transpose _bool=False) -> Tensor
c.def("ormqr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("ormqr with signature ormqr(self, input2 Tensor, input3 Tensor, left _bool=True, transpose _bool=False) -> Tensor"); });

// outer(self, vec2 Tensor) -> Tensor
c.def("outer", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("outer with signature outer(self, vec2 Tensor) -> Tensor"); });

// @overload permute(self, dims _size) -> Tensor
// aten::permute : (Tensor, int[]) -> (Tensor)
c.def("permute", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&>(&permute));

// pin_memory(self, device Optional[Union[_device, str, None]]=None) -> Tensor
c.def("pin_memory", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("pin_memory with signature pin_memory(self, device Optional[Union[_device, str, None]]=None) -> Tensor"); });

// pinverse(self, rcond _float=1e-15) -> Tensor
c.def("pinverse", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("pinverse with signature pinverse(self, rcond _float=1e-15) -> Tensor"); });

// polygamma(self, n _int) -> Tensor
c.def("polygamma", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("polygamma with signature polygamma(self, n _int) -> Tensor"); });

// polygamma_(self, n _int) -> Tensor
c.def("polygamma_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("polygamma_ with signature polygamma_(self, n _int) -> Tensor"); });

// positive(self) -> Tensor
c.def("positive", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("positive with signature positive(self) -> Tensor"); });

// @overload pow(self, exponent Tensor) -> Tensor
// aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)
c.def("pow", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&pow));

// @overload pow_(self, exponent Tensor) -> Tensor
c.def("pow_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("pow_ with signature @overload pow_(self, exponent Tensor) -> Tensor"); });

// @overload pow_(self, exponent Number) -> Tensor
c.def("pow_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("pow_ with signature @overload pow_(self, exponent Number) -> Tensor"); });

// prelu(self, weight Tensor) -> Tensor
// aten::prelu : (Tensor, Tensor) -> (Tensor)
c.def("prelu", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&prelu));

// @overload prod(self, *, dtype Optional[_dtype]=None) -> Tensor
c.def("prod", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("prod with signature @overload prod(self, *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload prod(self, dim _int, keepdim _bool=False, *, dtype Optional[_dtype]=None) -> Tensor
c.def("prod", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("prod with signature @overload prod(self, dim _int, keepdim _bool=False, *, dtype Optional[_dtype]=None) -> Tensor"); });

// @overload prod(self, dim Union[str, ellipsis, None], keepdim _bool=False, *, dtype Optional[_dtype]=None) -> Tensor
c.def("prod", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("prod with signature @overload prod(self, dim Union[str, ellipsis, None], keepdim _bool=False, *, dtype Optional[_dtype]=None) -> Tensor"); });

// put(self, index Tensor, source Tensor, accumulate _bool=False) -> Tensor
c.def("put", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("put with signature put(self, index Tensor, source Tensor, accumulate _bool=False) -> Tensor"); });

// put_(self, index Tensor, source Tensor, accumulate _bool=False) -> Tensor
c.def("put_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("put_ with signature put_(self, index Tensor, source Tensor, accumulate _bool=False) -> Tensor"); });

// q_per_channel_axis(self) -> _int
c.def("q_per_channel_axis", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("q_per_channel_axis with signature q_per_channel_axis(self) -> _int"); });

// q_per_channel_scales(self) -> Tensor
c.def("q_per_channel_scales", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("q_per_channel_scales with signature q_per_channel_scales(self) -> Tensor"); });

// q_per_channel_zero_points(self) -> Tensor
c.def("q_per_channel_zero_points", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("q_per_channel_zero_points with signature q_per_channel_zero_points(self) -> Tensor"); });

// q_scale(self) -> _float
c.def("q_scale", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("q_scale with signature q_scale(self) -> _float"); });

// q_zero_point(self) -> _int
c.def("q_zero_point", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("q_zero_point with signature q_zero_point(self) -> _int"); });

// qr(self, some _bool=True) -> torch.return_types.qr
c.def("qr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("qr with signature qr(self, some _bool=True) -> torch.return_types.qr"); });

// qscheme(self) -> _qscheme
c.def("qscheme", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("qscheme with signature qscheme(self) -> _qscheme"); });

// @overload quantile(self, q Tensor, dim Optional[_int]=None, keepdim _bool=False, *, interpolation str='linear') -> Tensor
c.def("quantile", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("quantile with signature @overload quantile(self, q Tensor, dim Optional[_int]=None, keepdim _bool=False, *, interpolation str='linear') -> Tensor"); });

// @overload quantile(self, q _float, dim Optional[_int]=None, keepdim _bool=False, *, interpolation str='linear') -> Tensor
c.def("quantile", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("quantile with signature @overload quantile(self, q _float, dim Optional[_int]=None, keepdim _bool=False, *, interpolation str='linear') -> Tensor"); });

// rad2deg(self) -> Tensor
c.def("rad2deg", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("rad2deg with signature rad2deg(self) -> Tensor"); });

// rad2deg_(self) -> Tensor
c.def("rad2deg_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("rad2deg_ with signature rad2deg_(self) -> Tensor"); });

// @overload random_(self, *, generator Optional[Generator]=None) -> Tensor
c.def("random_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("random_ with signature @overload random_(self, *, generator Optional[Generator]=None) -> Tensor"); });

// @overload random_(self, from_ _int, to Optional[_int], *, generator Optional[Generator]=None) -> Tensor
c.def("random_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("random_ with signature @overload random_(self, from_ _int, to Optional[_int], *, generator Optional[Generator]=None) -> Tensor"); });

// @overload random_(self, to _int, *, generator Optional[Generator]=None) -> Tensor
c.def("random_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("random_ with signature @overload random_(self, to _int, *, generator Optional[Generator]=None) -> Tensor"); });

// ravel(self) -> Tensor
c.def("ravel", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("ravel with signature ravel(self) -> Tensor"); });

// reciprocal(self) -> Tensor
// aten::reciprocal : (Tensor) -> (Tensor)
c.def("reciprocal", py::overload_cast<const PyAnyTorchTensorValue&>(&reciprocal));

// reciprocal_(self) -> Tensor
// aten::reciprocal_ : (Tensor) -> (Tensor)
c.def("reciprocal_", py::overload_cast<const PyAnyTorchTensorValue&>(&reciprocal_));

// record_stream(self, s Stream) -> None
c.def("record_stream", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("record_stream with signature record_stream(self, s Stream) -> None"); });

// refine_names(self, names Sequence[Union[str, ellipsis, None]]) -> Tensor
c.def("refine_names", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("refine_names with signature refine_names(self, names Sequence[Union[str, ellipsis, None]]) -> Tensor"); });

// relu(self) -> Tensor
// aten::relu : (Tensor) -> (Tensor)
c.def("relu", py::overload_cast<const PyAnyTorchTensorValue&>(&relu));

// relu_(self) -> Tensor
// aten::relu_ : (Tensor) -> (Tensor)
c.def("relu_", py::overload_cast<const PyAnyTorchTensorValue&>(&relu_));

// @overload remainder(self, other Tensor) -> Tensor
// aten::remainder.Scalar : (Tensor, Scalar) -> (Tensor)
c.def("remainder", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&remainder));

// @overload remainder_(self, other Tensor) -> Tensor
c.def("remainder_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("remainder_ with signature @overload remainder_(self, other Tensor) -> Tensor"); });

// @overload remainder_(self, other Number) -> Tensor
c.def("remainder_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("remainder_ with signature @overload remainder_(self, other Number) -> Tensor"); });

// rename(self, names Optional[Sequence[Union[str, ellipsis, None]]]) -> Tensor
c.def("rename", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("rename with signature rename(self, names Optional[Sequence[Union[str, ellipsis, None]]]) -> Tensor"); });

// rename_(self, names Optional[Sequence[Union[str, ellipsis, None]]]) -> Tensor
c.def("rename_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("rename_ with signature rename_(self, names Optional[Sequence[Union[str, ellipsis, None]]]) -> Tensor"); });

// renorm(self, p Number, dim _int, maxnorm Number) -> Tensor
c.def("renorm", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("renorm with signature renorm(self, p Number, dim _int, maxnorm Number) -> Tensor"); });

// renorm_(self, p Number, dim _int, maxnorm Number) -> Tensor
c.def("renorm_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("renorm_ with signature renorm_(self, p Number, dim _int, maxnorm Number) -> Tensor"); });

// @overload repeat(self, repeats Sequence[Union[_int, SymInt]]) -> Tensor
// aten::repeat : (Tensor, int[]) -> (Tensor)
c.def("repeat", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&>(&repeat));

// @overload repeat_interleave(self, repeats Tensor, dim Optional[_int]=None, *, output_size Optional[_int]=None) -> Tensor
c.def("repeat_interleave", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("repeat_interleave with signature @overload repeat_interleave(self, repeats Tensor, dim Optional[_int]=None, *, output_size Optional[_int]=None) -> Tensor"); });

// @overload repeat_interleave(self, repeats Union[_int, SymInt], dim Optional[_int]=None, *, output_size Optional[_int]=None) -> Tensor
c.def("repeat_interleave", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("repeat_interleave with signature @overload repeat_interleave(self, repeats Union[_int, SymInt], dim Optional[_int]=None, *, output_size Optional[_int]=None) -> Tensor"); });

// requires_grad_(self, mode _bool=True) -> Tensor
c.def("requires_grad_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("requires_grad_ with signature requires_grad_(self, mode _bool=True) -> Tensor"); });

// @overload reshape(self, shape Sequence[Union[_int, SymInt]]) -> Tensor
// aten::reshape : (Tensor, int[]) -> (Tensor)
c.def("reshape", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&>(&reshape));

// reshape_as(self, other Tensor) -> Tensor
c.def("reshape_as", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("reshape_as with signature reshape_as(self, other Tensor) -> Tensor"); });

// @overload resize_(self, size Sequence[Union[_int, SymInt]], *, memory_format Optional[memory_format]=None) -> Tensor
// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
c.def("resize_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&, const PyAnyTorchOptionalIntValue&>(&resize_));

// resize_as_(self, the_template Tensor, *, memory_format Optional[memory_format]=None) -> Tensor
c.def("resize_as_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("resize_as_ with signature resize_as_(self, the_template Tensor, *, memory_format Optional[memory_format]=None) -> Tensor"); });

// resize_as_sparse_(self, the_template Tensor) -> Tensor
c.def("resize_as_sparse_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("resize_as_sparse_ with signature resize_as_sparse_(self, the_template Tensor) -> Tensor"); });

// resolve_conj(self) -> Tensor
c.def("resolve_conj", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("resolve_conj with signature resolve_conj(self) -> Tensor"); });

// resolve_neg(self) -> Tensor
c.def("resolve_neg", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("resolve_neg with signature resolve_neg(self) -> Tensor"); });

// retain_grad(self) -> None
c.def("retain_grad", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("retain_grad with signature retain_grad(self) -> None"); });

// roll(self, shifts Sequence[Union[_int, SymInt]], dims Union[_int, _size]=()) -> Tensor
// aten::roll : (Tensor, int[], int[]) -> (Tensor)
c.def("roll", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&, const PyAnyTorchListOfTorchIntValue&>(&roll));

// rot90(self, k _int=1, dims _size=(0, 1)) -> Tensor
c.def("rot90", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("rot90 with signature rot90(self, k _int=1, dims _size=(0, 1)) -> Tensor"); });

// @overload round(self) -> Tensor
// aten::round : (Tensor) -> (Tensor)
c.def("round", py::overload_cast<const PyAnyTorchTensorValue&>(&round));

// @overload round_(self) -> Tensor
// aten::round_ : (Tensor) -> (Tensor)
c.def("round_", py::overload_cast<const PyAnyTorchTensorValue&>(&round_));

// row_indices(self) -> Tensor
c.def("row_indices", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("row_indices with signature row_indices(self) -> Tensor"); });

// rsqrt(self) -> Tensor
// aten::rsqrt : (Tensor) -> (Tensor)
c.def("rsqrt", py::overload_cast<const PyAnyTorchTensorValue&>(&rsqrt));

// rsqrt_(self) -> Tensor
// aten::rsqrt_ : (Tensor) -> (Tensor)
c.def("rsqrt_", py::overload_cast<const PyAnyTorchTensorValue&>(&rsqrt_));

// @overload scatter(self, dim _int, index Tensor, src Tensor) -> Tensor
c.def("scatter", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter with signature @overload scatter(self, dim _int, index Tensor, src Tensor) -> Tensor"); });

// @overload scatter(self, dim _int, index Tensor, src Tensor, *, reduce str) -> Tensor
c.def("scatter", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter with signature @overload scatter(self, dim _int, index Tensor, src Tensor, *, reduce str) -> Tensor"); });

// @overload scatter(self, dim _int, index Tensor, value Number, *, reduce str) -> Tensor
c.def("scatter", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter with signature @overload scatter(self, dim _int, index Tensor, value Number, *, reduce str) -> Tensor"); });

// @overload scatter(self, dim Union[str, ellipsis, None], index Tensor, src Tensor) -> Tensor
c.def("scatter", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter with signature @overload scatter(self, dim Union[str, ellipsis, None], index Tensor, src Tensor) -> Tensor"); });

// @overload scatter(self, dim _int, index Tensor, value Number) -> Tensor
c.def("scatter", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter with signature @overload scatter(self, dim _int, index Tensor, value Number) -> Tensor"); });

// @overload scatter(self, dim Union[str, ellipsis, None], index Tensor, value Number) -> Tensor
c.def("scatter", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter with signature @overload scatter(self, dim Union[str, ellipsis, None], index Tensor, value Number) -> Tensor"); });

// @overload scatter_(self, dim _int, index Tensor, src Tensor) -> Tensor
c.def("scatter_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter_ with signature @overload scatter_(self, dim _int, index Tensor, src Tensor) -> Tensor"); });

// @overload scatter_(self, dim _int, index Tensor, src Tensor, *, reduce str) -> Tensor
c.def("scatter_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter_ with signature @overload scatter_(self, dim _int, index Tensor, src Tensor, *, reduce str) -> Tensor"); });

// @overload scatter_(self, dim _int, index Tensor, value Number, *, reduce str) -> Tensor
c.def("scatter_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter_ with signature @overload scatter_(self, dim _int, index Tensor, value Number, *, reduce str) -> Tensor"); });

// @overload scatter_(self, dim _int, index Tensor, value Number) -> Tensor
c.def("scatter_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("scatter_ with signature @overload scatter_(self, dim _int, index Tensor, value Number) -> Tensor"); });

// @overload scatter_add(self, dim _int, index Tensor, src Tensor) -> Tensor
// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
c.def("scatter_add", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&scatter_add));

// scatter_add_(self, dim _int, index Tensor, src Tensor) -> Tensor
// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
c.def("scatter_add_", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&scatter_add_));

// scatter_reduce(self, dim _int, index Tensor, src Tensor, reduce str, *, include_self _bool=True) -> Tensor
// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
c.def("scatter_reduce", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyTorch_StringValue&, const PyTorch_BoolValue&>(&scatter_reduce));

// scatter_reduce_(self, dim _int, index Tensor, src Tensor, reduce str, *, include_self _bool=True) -> Tensor
// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
c.def("scatter_reduce_", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyTorch_StringValue&, const PyTorch_BoolValue&>(&scatter_reduce_));

// @overload select(self, dim _int, index Union[_int, SymInt]) -> Tensor
// aten::select.int : (Tensor, int, int) -> (Tensor)
c.def("select", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyTorch_IntValue&>(&select));

// select_scatter(self, src Tensor, dim _int, index Union[_int, SymInt]) -> Tensor
// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
c.def("select_scatter", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyTorch_IntValue&>(&select_scatter));

// @overload set_(self, storage Union[Storage, TypedStorage, UntypedStorage], offset _int, size _size, stride _size) -> Tensor
c.def("set_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("set_ with signature @overload set_(self, storage Union[Storage, TypedStorage, UntypedStorage], offset _int, size _size, stride _size) -> Tensor"); });

// @overload set_(self, storage Union[Storage, TypedStorage, UntypedStorage]) -> Tensor
c.def("set_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("set_ with signature @overload set_(self, storage Union[Storage, TypedStorage, UntypedStorage]) -> Tensor"); });

// sgn(self) -> Tensor
c.def("sgn", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sgn with signature sgn(self) -> Tensor"); });

// sgn_(self) -> Tensor
c.def("sgn_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sgn_ with signature sgn_(self) -> Tensor"); });

// short(self) -> Tensor
c.def("short", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("short with signature short(self) -> Tensor"); });

// sigmoid(self) -> Tensor
// aten::sigmoid : (Tensor) -> (Tensor)
c.def("sigmoid", py::overload_cast<const PyAnyTorchTensorValue&>(&sigmoid));

// sigmoid_(self) -> Tensor
// aten::sigmoid_ : (Tensor) -> (Tensor)
c.def("sigmoid_", py::overload_cast<const PyAnyTorchTensorValue&>(&sigmoid_));

// sign(self) -> Tensor
c.def("sign", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sign with signature sign(self) -> Tensor"); });

// sign_(self) -> Tensor
c.def("sign_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sign_ with signature sign_(self) -> Tensor"); });

// signbit(self) -> Tensor
c.def("signbit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("signbit with signature signbit(self) -> Tensor"); });

// sin(self) -> Tensor
// aten::sin : (Tensor) -> (Tensor)
c.def("sin", py::overload_cast<const PyAnyTorchTensorValue&>(&sin));

// sin_(self) -> Tensor
// aten::sin_ : (Tensor) -> (Tensor)
c.def("sin_", py::overload_cast<const PyAnyTorchTensorValue&>(&sin_));

// sinc(self) -> Tensor
c.def("sinc", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sinc with signature sinc(self) -> Tensor"); });

// sinc_(self) -> Tensor
c.def("sinc_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sinc_ with signature sinc_(self) -> Tensor"); });

// sinh(self) -> Tensor
c.def("sinh", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sinh with signature sinh(self) -> Tensor"); });

// sinh_(self) -> Tensor
c.def("sinh_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sinh_ with signature sinh_(self) -> Tensor"); });

// @overload size(self) -> Size
// aten::size : (Tensor) -> (int[])
c.def("size", py::overload_cast<const PyAnyTorchTensorValue&>(&size));

// @overload size(self, dim _int) -> _int
// aten::size.int : (Tensor, int) -> (int)
c.def("size", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&>(&size));

// slice_scatter(self, src Tensor, dim _int=0, start Optional[Union[_int, SymInt]]=None, end Optional[Union[_int, SymInt]]=None, step Union[_int, SymInt]=1) -> Tensor
// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
c.def("slice_scatter", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchOptionalIntValue&, const PyAnyTorchOptionalIntValue&, const PyTorch_IntValue&>(&slice_scatter));

// slogdet(self) -> torch.return_types.slogdet
c.def("slogdet", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("slogdet with signature slogdet(self) -> torch.return_types.slogdet"); });

// smm(self, mat2 Tensor) -> Tensor
c.def("smm", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("smm with signature smm(self, mat2 Tensor) -> Tensor"); });

// @overload softmax(self, dim _int, dtype Optional[_dtype]=None) -> Tensor
// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
c.def("softmax", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyAnyTorchOptionalIntValue&>(&softmax));

// sparse_dim(self) -> _int
c.def("sparse_dim", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sparse_dim with signature sparse_dim(self) -> _int"); });

// sparse_mask(self, mask Tensor) -> Tensor
c.def("sparse_mask", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sparse_mask with signature sparse_mask(self, mask Tensor) -> Tensor"); });

// sparse_resize_(self, size _size, sparse_dim _int, dense_dim _int) -> Tensor
c.def("sparse_resize_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sparse_resize_ with signature sparse_resize_(self, size _size, sparse_dim _int, dense_dim _int) -> Tensor"); });

// sparse_resize_and_clear_(self, size _size, sparse_dim _int, dense_dim _int) -> Tensor
c.def("sparse_resize_and_clear_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sparse_resize_and_clear_ with signature sparse_resize_and_clear_(self, size _size, sparse_dim _int, dense_dim _int) -> Tensor"); });

// @overload split(self, split_size _int, dim _int=0) -> Sequence[Tensor]
c.def("split", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("split with signature @overload split(self, split_size _int, dim _int=0) -> Sequence[Tensor]"); });

// @overload split(self, split_size Tuple[_int, ], dim _int=0) -> Sequence[Tensor]
c.def("split", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("split with signature @overload split(self, split_size Tuple[_int, ], dim _int=0) -> Sequence[Tensor]"); });

// split_with_sizes(self, split_sizes Sequence[Union[_int, SymInt]], dim _int=0) -> List[Tensor]
c.def("split_with_sizes", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("split_with_sizes with signature split_with_sizes(self, split_sizes Sequence[Union[_int, SymInt]], dim _int=0) -> List[Tensor]"); });

// sqrt(self) -> Tensor
// aten::sqrt : (Tensor) -> (Tensor)
c.def("sqrt", py::overload_cast<const PyAnyTorchTensorValue&>(&sqrt));

// sqrt_(self) -> Tensor
// aten::sqrt_ : (Tensor) -> (Tensor)
c.def("sqrt_", py::overload_cast<const PyAnyTorchTensorValue&>(&sqrt_));

// square(self) -> Tensor
// aten::square : (Tensor) -> (Tensor)
c.def("square", py::overload_cast<const PyAnyTorchTensorValue&>(&square));

// square_(self) -> Tensor
// aten::square_ : (Tensor) -> (Tensor)
c.def("square_", py::overload_cast<const PyAnyTorchTensorValue&>(&square_));

// @overload squeeze(self) -> Tensor
// aten::squeeze : (Tensor) -> (Tensor)
c.def("squeeze", py::overload_cast<const PyAnyTorchTensorValue&>(&squeeze));

// @overload squeeze(self, dim _int) -> Tensor
// aten::squeeze.dim : (Tensor, int) -> (Tensor)
c.def("squeeze", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&>(&squeeze));

// @overload squeeze_(self) -> Tensor
c.def("squeeze_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("squeeze_ with signature @overload squeeze_(self) -> Tensor"); });

// @overload squeeze_(self, dim _int) -> Tensor
c.def("squeeze_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("squeeze_ with signature @overload squeeze_(self, dim _int) -> Tensor"); });

// @overload squeeze_(self, dim _size) -> Tensor
c.def("squeeze_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("squeeze_ with signature @overload squeeze_(self, dim _size) -> Tensor"); });

// @overload squeeze_(self, *dim _int) -> Tensor
c.def("squeeze_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("squeeze_ with signature @overload squeeze_(self, *dim _int) -> Tensor"); });

// @overload squeeze_(self, dim Union[str, ellipsis, None]) -> Tensor
c.def("squeeze_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("squeeze_ with signature @overload squeeze_(self, dim Union[str, ellipsis, None]) -> Tensor"); });

// sspaddmm(self, mat1 Tensor, mat2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor
c.def("sspaddmm", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sspaddmm with signature sspaddmm(self, mat1 Tensor, mat2 Tensor, *, beta Number=1, alpha Number=1) -> Tensor"); });

// @overload std(self, unbiased _bool=True) -> Tensor
// aten::std : (Tensor, bool) -> (Tensor)
c.def("std", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_BoolValue&>(&std));

// untyped_storage(self) -> UntypedStorage
c.def("untyped_storage", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("untyped_storage with signature untyped_storage(self) -> UntypedStorage"); });

// storage_offset(self) -> _int
c.def("storage_offset", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("storage_offset with signature storage_offset(self) -> _int"); });

// storage_type(self) -> Storage
c.def("storage_type", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("storage_type with signature storage_type(self) -> Storage"); });

// @overload stride(self) -> Tuple[_int, ]
c.def("stride", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("stride with signature @overload stride(self) -> Tuple[_int, ]"); });

// @overload stride(self, _int) -> _int
c.def("stride", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("stride with signature @overload stride(self, _int) -> _int"); });

// sub_(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat], *, alpha Optional[Number]=1) -> Tensor
// aten::sub_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
c.def("sub_", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchScalarValue&>(&sub_));

// @overload subtract(self, other Tensor, *, alpha Number=1) -> Tensor
c.def("subtract", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("subtract with signature @overload subtract(self, other Tensor, *, alpha Number=1) -> Tensor"); });

// @overload subtract(self, other Number, alpha Number=1) -> Tensor
c.def("subtract", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("subtract with signature @overload subtract(self, other Number, alpha Number=1) -> Tensor"); });

// @overload subtract_(self, other Tensor, *, alpha Number=1) -> Tensor
c.def("subtract_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("subtract_ with signature @overload subtract_(self, other Tensor, *, alpha Number=1) -> Tensor"); });

// @overload subtract_(self, other Number, alpha Number=1) -> Tensor
c.def("subtract_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("subtract_ with signature @overload subtract_(self, other Number, alpha Number=1) -> Tensor"); });

// @overload sum(self, *, dtype Optional[_dtype]=None) -> Tensor
// aten::sum : (Tensor, int?) -> (Tensor)
c.def("sum", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchOptionalIntValue&>(&sum));

// @overload sum_to_size(self, size Sequence[Union[_int, SymInt]]) -> Tensor
c.def("sum_to_size", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sum_to_size with signature @overload sum_to_size(self, size Sequence[Union[_int, SymInt]]) -> Tensor"); });

// @overload sum_to_size(self, *size _int) -> Tensor
c.def("sum_to_size", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("sum_to_size with signature @overload sum_to_size(self, *size _int) -> Tensor"); });

// svd(self, some _bool=True, compute_uv _bool=True) -> torch.return_types.svd
c.def("svd", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("svd with signature svd(self, some _bool=True, compute_uv _bool=True) -> torch.return_types.svd"); });

// swapaxes(self, axis0 _int, axis1 _int) -> Tensor
c.def("swapaxes", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("swapaxes with signature swapaxes(self, axis0 _int, axis1 _int) -> Tensor"); });

// swapaxes_(self, axis0 _int, axis1 _int) -> Tensor
c.def("swapaxes_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("swapaxes_ with signature swapaxes_(self, axis0 _int, axis1 _int) -> Tensor"); });

// swapdims(self, dim0 _int, dim1 _int) -> Tensor
c.def("swapdims", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("swapdims with signature swapdims(self, dim0 _int, dim1 _int) -> Tensor"); });

// swapdims_(self, dim0 _int, dim1 _int) -> Tensor
c.def("swapdims_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("swapdims_ with signature swapdims_(self, dim0 _int, dim1 _int) -> Tensor"); });

// t(self) -> Tensor
// aten::t : (Tensor) -> (Tensor)
c.def("t", py::overload_cast<const PyAnyTorchTensorValue&>(&t));

// t_(self) -> Tensor
c.def("t_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("t_ with signature t_(self) -> Tensor"); });

// take(self, index Tensor) -> Tensor
c.def("take", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("take with signature take(self, index Tensor) -> Tensor"); });

// take_along_dim(self, indices Tensor, dim Optional[_int]=None) -> Tensor
c.def("take_along_dim", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("take_along_dim with signature take_along_dim(self, indices Tensor, dim Optional[_int]=None) -> Tensor"); });

// tan(self) -> Tensor
c.def("tan", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tan with signature tan(self) -> Tensor"); });

// tan_(self) -> Tensor
c.def("tan_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tan_ with signature tan_(self) -> Tensor"); });

// tanh(self) -> Tensor
// aten::tanh : (Tensor) -> (Tensor)
c.def("tanh", py::overload_cast<const PyAnyTorchTensorValue&>(&tanh));

// tanh_(self) -> Tensor
// aten::tanh_ : (Tensor) -> (Tensor)
c.def("tanh_", py::overload_cast<const PyAnyTorchTensorValue&>(&tanh_));

// @overload tensor_split(self, indices Sequence[Union[_int, SymInt]], dim _int=0) -> List[Tensor]
c.def("tensor_split", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tensor_split with signature @overload tensor_split(self, indices Sequence[Union[_int, SymInt]], dim _int=0) -> List[Tensor]"); });

// @overload tensor_split(self, tensor_indices_or_sections Tensor, dim _int=0) -> List[Tensor]
c.def("tensor_split", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tensor_split with signature @overload tensor_split(self, tensor_indices_or_sections Tensor, dim _int=0) -> List[Tensor]"); });

// @overload tensor_split(self, sections Union[_int, SymInt], dim _int=0) -> List[Tensor]
c.def("tensor_split", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tensor_split with signature @overload tensor_split(self, sections Union[_int, SymInt], dim _int=0) -> List[Tensor]"); });

// @overload tile(self, dims _size) -> Tensor
c.def("tile", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tile with signature @overload tile(self, dims _size) -> Tensor"); });

// @overload tile(self, *dims _int) -> Tensor
c.def("tile", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tile with signature @overload tile(self, *dims _int) -> Tensor"); });

// @overload to(self, device Optional[Union[_device, str]]=None, dtype Optional[_dtype]=None, non_blocking _bool=False, copy _bool=False) -> Tensor
// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
c.def("to", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchOptionalDeviceValue&, const PyAnyTorchOptionalIntValue&, const PyTorch_BoolValue&, const PyTorch_BoolValue&>(&to));

// to_dense(self, dtype Optional[_dtype]=None, *, masked_grad Optional[_bool]=None) -> Tensor
c.def("to_dense", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_dense with signature to_dense(self, dtype Optional[_dtype]=None, *, masked_grad Optional[_bool]=None) -> Tensor"); });

// to_mkldnn(self, dtype Optional[_dtype]=None) -> Tensor
c.def("to_mkldnn", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_mkldnn with signature to_mkldnn(self, dtype Optional[_dtype]=None) -> Tensor"); });

// to_padded_tensor(self, padding _float, output_size Optional[Sequence[Union[_int, SymInt]]]=None) -> Tensor
c.def("to_padded_tensor", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_padded_tensor with signature to_padded_tensor(self, padding _float, output_size Optional[Sequence[Union[_int, SymInt]]]=None) -> Tensor"); });

// @overload to_sparse(self, *, layout Optional[_layout]=None, blocksize Optional[Union[_int, _size]]=None, dense_dim Optional[_int]=None) -> Tensor
c.def("to_sparse", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_sparse with signature @overload to_sparse(self, *, layout Optional[_layout]=None, blocksize Optional[Union[_int, _size]]=None, dense_dim Optional[_int]=None) -> Tensor"); });

// @overload to_sparse(self, sparse_dim _int) -> Tensor
c.def("to_sparse", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_sparse with signature @overload to_sparse(self, sparse_dim _int) -> Tensor"); });

// to_sparse_bsc(self, blocksize Union[_int, _size], dense_dim Optional[_int]=None) -> Tensor
c.def("to_sparse_bsc", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_sparse_bsc with signature to_sparse_bsc(self, blocksize Union[_int, _size], dense_dim Optional[_int]=None) -> Tensor"); });

// to_sparse_bsr(self, blocksize Union[_int, _size], dense_dim Optional[_int]=None) -> Tensor
c.def("to_sparse_bsr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_sparse_bsr with signature to_sparse_bsr(self, blocksize Union[_int, _size], dense_dim Optional[_int]=None) -> Tensor"); });

// to_sparse_csc(self, dense_dim Optional[_int]=None) -> Tensor
c.def("to_sparse_csc", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_sparse_csc with signature to_sparse_csc(self, dense_dim Optional[_int]=None) -> Tensor"); });

// to_sparse_csr(self, dense_dim Optional[_int]=None) -> Tensor
c.def("to_sparse_csr", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("to_sparse_csr with signature to_sparse_csr(self, dense_dim Optional[_int]=None) -> Tensor"); });

// tolist(self) -> List
c.def("tolist", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tolist with signature tolist(self) -> List"); });

// topk(self, k Union[_int, SymInt], dim _int=-1, largest _bool=True, sorted _bool=True) -> torch.return_types.topk
c.def("topk", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("topk with signature topk(self, k Union[_int, SymInt], dim _int=-1, largest _bool=True, sorted _bool=True) -> torch.return_types.topk"); });

// trace(self) -> Tensor
c.def("trace", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("trace with signature trace(self) -> Tensor"); });

// @overload transpose(self, dim0 _int, dim1 _int) -> Tensor
// aten::transpose.int : (Tensor, int, int) -> (Tensor)
c.def("transpose", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&, const PyTorch_IntValue&>(&transpose));

// transpose_(self, dim0 _int, dim1 _int) -> Tensor
c.def("transpose_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("transpose_ with signature transpose_(self, dim0 _int, dim1 _int) -> Tensor"); });

// triangular_solve(self, A Tensor, upper _bool=True, transpose _bool=False, unitriangular _bool=False) -> torch.return_types.triangular_solve
c.def("triangular_solve", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("triangular_solve with signature triangular_solve(self, A Tensor, upper _bool=True, transpose _bool=False, unitriangular _bool=False) -> torch.return_types.triangular_solve"); });

// tril(self, diagonal _int=0) -> Tensor
c.def("tril", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tril with signature tril(self, diagonal _int=0) -> Tensor"); });

// tril_(self, diagonal _int=0) -> Tensor
c.def("tril_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("tril_ with signature tril_(self, diagonal _int=0) -> Tensor"); });

// triu(self, diagonal _int=0) -> Tensor
// aten::triu : (Tensor, int) -> (Tensor)
c.def("triu", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&>(&triu));

// triu_(self, diagonal _int=0) -> Tensor
// aten::triu_ : (Tensor, int) -> (Tensor)
c.def("triu_", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&>(&triu_));

// true_divide(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat], *, out Optional[Tensor]=None) -> Tensor
c.def("true_divide", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("true_divide with signature true_divide(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat], *, out Optional[Tensor]=None) -> Tensor"); });

// true_divide_(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat]) -> Tensor
c.def("true_divide_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("true_divide_ with signature true_divide_(self, other Union[Tensor, Number, torch.SymInt, torch.SymFloat]) -> Tensor"); });

// trunc(self) -> Tensor
c.def("trunc", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("trunc with signature trunc(self) -> Tensor"); });

// trunc_(self) -> Tensor
c.def("trunc_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("trunc_ with signature trunc_(self) -> Tensor"); });

// type_as(self, other Tensor) -> Tensor
// aten::type_as : (Tensor, Tensor) -> (Tensor)
c.def("type_as", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&type_as));

// @overload unbind(self, dim _int=0) -> List[Tensor]
c.def("unbind", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("unbind with signature @overload unbind(self, dim _int=0) -> List[Tensor]"); });

// @overload unbind(self, dim Union[str, ellipsis, None]) -> List[Tensor]
c.def("unbind", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("unbind with signature @overload unbind(self, dim Union[str, ellipsis, None]) -> List[Tensor]"); });

// @overload unflatten(self, dim Union[str, ellipsis, None], sizes Sequence[Union[_int, SymInt]], names Sequence[Union[str, ellipsis, None]]) -> Tensor
c.def("unflatten", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("unflatten with signature @overload unflatten(self, dim Union[str, ellipsis, None], sizes Sequence[Union[_int, SymInt]], names Sequence[Union[str, ellipsis, None]]) -> Tensor"); });

// @overload unflatten(self, dim _int, sizes Sequence[Union[_int, SymInt]]) -> Tensor
c.def("unflatten", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("unflatten with signature @overload unflatten(self, dim _int, sizes Sequence[Union[_int, SymInt]]) -> Tensor"); });

// unfold(self, dimension _int, size _int, step _int) -> Tensor
c.def("unfold", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("unfold with signature unfold(self, dimension _int, size _int, step _int) -> Tensor"); });

// unsafe_chunk(self, chunks _int, dim _int=0) -> List[Tensor]
c.def("unsafe_chunk", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("unsafe_chunk with signature unsafe_chunk(self, chunks _int, dim _int=0) -> List[Tensor]"); });

// unsafe_split(self, split_size Union[_int, SymInt], dim _int=0) -> List[Tensor]
c.def("unsafe_split", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("unsafe_split with signature unsafe_split(self, split_size Union[_int, SymInt], dim _int=0) -> List[Tensor]"); });

// unsafe_split_with_sizes(self, split_sizes Sequence[Union[_int, SymInt]], dim _int=0) -> List[Tensor]
c.def("unsafe_split_with_sizes", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("unsafe_split_with_sizes with signature unsafe_split_with_sizes(self, split_sizes Sequence[Union[_int, SymInt]], dim _int=0) -> List[Tensor]"); });

// unsqueeze(self, dim _int) -> Tensor
// aten::unsqueeze : (Tensor, int) -> (Tensor)
c.def("unsqueeze", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&>(&unsqueeze));

// unsqueeze_(self, dim _int) -> Tensor
// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
c.def("unsqueeze_", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_IntValue&>(&unsqueeze_));

// values(self) -> Tensor
c.def("values", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("values with signature values(self) -> Tensor"); });

// @overload var(self, unbiased _bool=True) -> Tensor
// aten::var : (Tensor, bool) -> (Tensor)
c.def("var", py::overload_cast<const PyAnyTorchTensorValue&, const PyTorch_BoolValue&>(&var));

// vdot(self, other Tensor) -> Tensor
c.def("vdot", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("vdot with signature vdot(self, other Tensor) -> Tensor"); });

// @overload view(self, size Sequence[Union[_int, SymInt]]) -> Tensor
// aten::view : (Tensor, int[]) -> (Tensor)
c.def("view", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchListOfTorchIntValue&>(&view));

// view_as(self, other Tensor) -> Tensor
c.def("view_as", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("view_as with signature view_as(self, other Tensor) -> Tensor"); });

// @overload vsplit(self, sections _int) -> List[Tensor]
c.def("vsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("vsplit with signature @overload vsplit(self, sections _int) -> List[Tensor]"); });

// @overload vsplit(self, indices _size) -> List[Tensor]
c.def("vsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("vsplit with signature @overload vsplit(self, indices _size) -> List[Tensor]"); });

// @overload vsplit(self, *indices _int) -> List[Tensor]
c.def("vsplit", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("vsplit with signature @overload vsplit(self, *indices _int) -> List[Tensor]"); });

// @overload where(self, condition Tensor, other Tensor) -> Tensor
// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
c.def("where", py::overload_cast<const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&, const PyAnyTorchTensorValue&>(&where));

// @overload xlogy(self, other Tensor) -> Tensor
c.def("xlogy", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("xlogy with signature @overload xlogy(self, other Tensor) -> Tensor"); });

// @overload xlogy(self, other Number) -> Tensor
c.def("xlogy", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("xlogy with signature @overload xlogy(self, other Number) -> Tensor"); });

// @overload xlogy_(self, other Tensor) -> Tensor
c.def("xlogy_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("xlogy_ with signature @overload xlogy_(self, other Tensor) -> Tensor"); });

// @overload xlogy_(self, other Number) -> Tensor
c.def("xlogy_", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) { throw NotImplementedError("xlogy_ with signature @overload xlogy_(self, other Number) -> Tensor"); });

// zero_(self) -> Tensor
// aten::zero_ : (Tensor) -> (Tensor)
c.def("zero_", py::overload_cast<const PyAnyTorchTensorValue&>(&zero_));
