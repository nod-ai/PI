// aten::abs : (Tensor) -> (Tensor)
PyAnyTorchTensorValue abs(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.abs";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::abs_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue abs_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.abs_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue adaptive_avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.adaptive_avg_pool2d";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, output_size}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::add.float_int : (float, int) -> (float)
PyTorch_FloatValue add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.add.float_int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::add.int : (int, int) -> (int)
PyTorch_IntValue add(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.add.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue add(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.add.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::add.str : (str, str) -> (str)
PyTorch_StringValue add(const PyTorch_StringValue &a, const PyTorch_StringValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.add.str";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::add.t : (t[], t[]) -> (t[])
PyAnyTorchListValue add(const PyAnyTorchListValue &a, const PyAnyTorchListValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.add.t";
  std::vector<PyType> _returnTypes = {PyAnyTorchListType(torchMlirTorchListTypeGetContainedType(mlirValueGetType(a)), DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue add(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.add.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::add_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue add_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.add_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::add_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue add_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.add_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcdiv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.addcdiv";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, tensor1, tensor2, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::addcdiv_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcdiv_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.addcdiv_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, tensor1, tensor2, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.addcmul";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, tensor1, tensor2, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::addcmul_ : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue addcmul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &tensor1, const PyAnyTorchTensorValue &tensor2, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.addcmul_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, tensor1, tensor2, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue addmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat1, const PyAnyTorchTensorValue &mat2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.addmm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, mat1, mat2, beta, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::alias_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue alias_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.alias_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::all.bool : (bool[]) -> (bool)
PyTorch_BoolValue all(const PyAnyTorchListOfTorchBoolValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.all.bool";
  auto _returnTypes = inferReturnTypes(operationName, {self}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::all : (Tensor) -> (Tensor)
PyAnyTorchTensorValue all(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.all";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::amax : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue amax(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.amax";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::any.bool : (bool[]) -> (bool)
PyTorch_BoolValue any(const PyAnyTorchListOfTorchBoolValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.any.bool";
  auto _returnTypes = inferReturnTypes(operationName, {self}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::any.dim : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue any(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.any.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::any : (Tensor) -> (Tensor)
PyAnyTorchTensorValue any(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.any";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.arange";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {end, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.arange.start";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {start, end, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::arange.start_out : (Scalar, Scalar, Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchTensorValue &out, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.arange.start_out";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {start, end, step, out}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue arange(const PyAnyTorchScalarValue &start, const PyAnyTorchScalarValue &end, const PyAnyTorchScalarValue &step, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.arange.start_step";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {start, end, step, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::argmax : (Tensor, int?, bool) -> (Tensor)
PyAnyTorchTensorValue argmax(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.argmax";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::as_strided_copy : (Tensor, int[], int[], int?) -> (Tensor)
PyAnyTorchTensorValue as_strided_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.as_strided_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, stride, storage_offset}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::as_strided_scatter : (Tensor, Tensor, int[], int[], int?) -> (Tensor)
PyAnyTorchTensorValue as_strided_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &storage_offset, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.as_strided_scatter";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, src, size, stride, storage_offset}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::atan2 : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue atan2(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.atan2";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::atan2_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue atan2_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.atan2_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::atan : (Tensor) -> (Tensor)
PyAnyTorchTensorValue atan(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.atan";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::atan_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue atan_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.atan_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::avg_pool2d : (Tensor, int[], int[], int[], bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue avg_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyTorch_BoolValue &ceil_mode, const PyTorch_BoolValue &count_include_pad, const PyAnyTorchOptionalIntValue &divisor_override, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.avg_pool2d";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::baddbmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue baddbmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.baddbmm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, batch1, batch2, beta, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::baddbmm_ : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue baddbmm_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &batch1, const PyAnyTorchTensorValue &batch2, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.baddbmm_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, batch1, batch2, beta, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)
PyAnyTorchTensorValue batch_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enabled, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.batch_norm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bernoulli : (Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bernoulli";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, generator}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bernoulli.p : (Tensor, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bernoulli.p";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, p, generator}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bernoulli.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bernoulli.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, p, generator}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bernoulli_.float : (Tensor, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bernoulli_.float";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, p, generator}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)
PyAnyTorchTensorValue bernoulli_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &p, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bernoulli_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, p, generator}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bincount : (Tensor, Tensor?, int) -> (Tensor)
PyAnyTorchTensorValue bincount(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &weights, const PyTorch_IntValue &minlength, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bincount";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, weights, minlength}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bitwise_and.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bitwise_and_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bitwise_and_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bitwise_not : (Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_not(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bitwise_not";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bitwise_not_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_not_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bitwise_not_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bitwise_or.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bitwise_or.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bitwise_or_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bitwise_or_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bitwise_xor.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bitwise_xor.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bitwise_xor_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bitwise_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bitwise_xor_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bmm : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue bmm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bmm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, mat2}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Bool.float : (float) -> (bool)
PyTorch_BoolValue Bool(const PyTorch_FloatValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Bool.float";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Bool.int : (int) -> (bool)
PyTorch_BoolValue Bool(const PyTorch_IntValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Bool.int";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Bool.Tensor : (Tensor) -> (bool)
PyTorch_BoolValue Bool(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Bool.Tensor";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::broadcast_to : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue broadcast_to(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.broadcast_to";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue bucketize(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &boundaries, const PyTorch_BoolValue &out_int32, const PyTorch_BoolValue &right, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.bucketize.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, boundaries, out_int32, right}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::cat : (Tensor[], int) -> (Tensor)
PyAnyTorchTensorValue cat(const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.cat";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {tensors, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ceil.float : (float) -> (int)
PyTorch_IntValue ceil(const PyTorch_FloatValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ceil.float";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ceil : (Tensor) -> (Tensor)
PyAnyTorchTensorValue ceil(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ceil";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ceil_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue ceil_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ceil_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::chunk : (Tensor, int, int) -> (Tensor[])
PyAnyTorchListOfTensorValue chunk(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &chunks, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.chunk";
  std::vector<PyType> _returnTypes = {PyAnyTorchListOfTensorType(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, chunks, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clamp_max : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_max(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clamp_max";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, max}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clamp_max_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_max_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &max, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clamp_max_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, max}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clamp_min : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_min(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clamp_min";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, min}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clamp_min_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue clamp_min_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clamp_min_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, min}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clamp : (Tensor, Scalar?, Scalar?) -> (Tensor)
PyAnyTorchTensorValue clamp(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clamp";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, min, max}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clamp.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
PyAnyTorchTensorValue clamp(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clamp.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, min, max}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clamp_ : (Tensor, Scalar?, Scalar?) -> (Tensor)
PyAnyTorchTensorValue clamp_(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &min, const PyAnyTorchOptionalScalarValue &max, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clamp_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, min, max}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clamp_.Tensor : (Tensor, Tensor?, Tensor?) -> (Tensor)
PyAnyTorchTensorValue clamp_(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalTensorValue &min, const PyAnyTorchOptionalTensorValue &max, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clamp_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, min, max}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::clone : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue clone(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.clone";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)
PyAnyTorchTensorValue constant_pad_nd(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.constant_pad_nd";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, pad__, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::contiguous : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue contiguous(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.contiguous";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)
PyAnyTorchTensorValue conv2d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_IntValue &groups, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.conv2d";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, stride, padding, dilation, groups}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::conv_transpose1d : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose1d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.conv_transpose1d";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, stride, padding, output_padding, groups, dilation}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::conv_transpose2d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose2d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.conv_transpose2d.input";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, stride, padding, output_padding, groups, dilation}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::conv_transpose3d.input : (Tensor, Tensor, Tensor?, int[], int[], int[], int, int[]) -> (Tensor)
PyAnyTorchTensorValue conv_transpose3d(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchIntValue &dilation, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.conv_transpose3d.input";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, stride, padding, output_padding, groups, dilation}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::convolution_backward : (Tensor, Tensor, Tensor, int[]?, int[], int[], int[], bool, int[], int, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> convolution_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalListOfTorchIntValue &bias_sizes, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyAnyTorchListOfTorchBoolValue &output_mask, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.convolution_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)});
}
// aten::convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)
PyAnyTorchTensorValue convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.convolution";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, stride, padding, dilation, transposed, output_padding, groups}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::copy : (Tensor, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue copy(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, src, non_blocking}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue copy_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_BoolValue &non_blocking, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.copy_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, src, non_blocking}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::cos : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cos(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.cos";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::cos_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cos_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.cos_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::cpu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cpu(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.cpu";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::cross_entropy_loss : (Tensor, Tensor, Tensor?, int, int, float) -> (Tensor)
PyAnyTorchTensorValue cross_entropy_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyTorch_FloatValue &label_smoothing, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.cross_entropy_loss";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, target, weight, reduction, ignore_index, label_smoothing}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::cuda : (Tensor) -> (Tensor)
PyAnyTorchTensorValue cuda(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.cuda";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::cumsum : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue cumsum(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.cumsum";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Delete.Dict_str : (Dict(str, t), str) -> ()
void Delete(const PyTorch_DictValue &self, const PyTorch_StringValue &key, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Delete.Dict_str";
  std::vector<PyType> _returnTypes = {}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, key}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  // no result
}
// aten::detach_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue detach_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.detach_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::detach : (Tensor) -> (Tensor)
PyAnyTorchTensorValue detach(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.detach";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::diagonal_copy : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue diagonal_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.diagonal_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, offset, dim1, dim2}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::diagonal_scatter : (Tensor, Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue diagonal_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &offset, const PyTorch_IntValue &dim1, const PyTorch_IntValue &dim2, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.diagonal_scatter";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, src, offset, dim1, dim2}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::dim : (Tensor) -> (int)
PyTorch_IntValue dim(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.dim";
  auto _returnTypes = inferReturnTypes(operationName, {self}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div.float : (float, float) -> (float)
PyTorch_FloatValue div(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div.float";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div.int : (int, int) -> (float)
PyTorch_FloatValue div(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div : (Scalar, Scalar) -> (float)
PyTorch_FloatValue div(const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div.Tensor_mode";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, rounding_mode}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue div(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div_.Tensor_mode : (Tensor, Tensor, str?) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchOptionalStringValue &rounding_mode, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div_.Tensor_mode";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, rounding_mode}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::div_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue div_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.div_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::dropout : (Tensor, float, bool) -> (Tensor)
PyAnyTorchTensorValue dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.dropout";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, p, train}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::dropout_ : (Tensor, float, bool) -> (Tensor)
PyAnyTorchTensorValue dropout_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &p, const PyTorch_BoolValue &train, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.dropout_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, p, train}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::embedding_bag.padding_idx : (Tensor, Tensor, Tensor, bool, int, bool, Tensor?, bool, int?) -> (Tensor, Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> embedding_bag(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyAnyTorchTensorValue &offsets, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_IntValue &mode, const PyTorch_BoolValue &sparse, const PyAnyTorchOptionalTensorValue &per_sample_weights, const PyTorch_BoolValue &include_last_offset, const PyAnyTorchOptionalIntValue &padding_idx, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.embedding_bag.padding_idx";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)}, {opRef, mlirOperationGetResult(operation, 3)});
}
// aten::embedding_dense_backward : (Tensor, Tensor, int, int, bool) -> (Tensor)
PyAnyTorchTensorValue embedding_dense_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &num_weights, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.embedding_dense_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, indices, num_weights, padding_idx, scale_grad_by_freq}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)
PyAnyTorchTensorValue embedding(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyTorch_IntValue &padding_idx, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_BoolValue &sparse, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.embedding";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {weight, indices, padding_idx, scale_grad_by_freq, sparse}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue empty_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.empty_like";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype, layout, device, pin_memory, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue empty(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.empty.memory_format";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {size, dtype, layout, device, pin_memory, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq.device : (Device, Device) -> (bool)
PyTorch_BoolValue eq(const PyTorch_DeviceValue &a, const PyTorch_DeviceValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq.device";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq.float : (float, float) -> (bool)
PyTorch_BoolValue eq(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq.float";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq.int_list : (int[], int[]) -> (bool)
PyTorch_BoolValue eq(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq.int_list";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq.int : (int, int) -> (bool)
PyTorch_BoolValue eq(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue eq(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq.str : (str, str) -> (bool)
PyTorch_BoolValue eq(const PyTorch_StringValue &a, const PyTorch_StringValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq.str";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue eq(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::eq_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue eq_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.eq_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::erf : (Tensor) -> (Tensor)
PyAnyTorchTensorValue erf(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.erf";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::erf_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue erf_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.erf_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::exp : (Tensor) -> (Tensor)
PyAnyTorchTensorValue exp(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.exp";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::exp_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue exp_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.exp_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::expand_as : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue expand_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.expand_as";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::expand_copy : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue expand_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.expand_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, implicit}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::expand : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue expand(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyTorch_BoolValue &implicit, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.expand";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, implicit}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::expm1 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue expm1(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.expm1";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::expm1_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue expm1_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.expm1_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::fft_fft : (Tensor, int?, int, str?) -> (Tensor)
PyAnyTorchTensorValue fft_fft(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &n, const PyTorch_IntValue &dim, const PyAnyTorchOptionalStringValue &norm, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.fft_fft";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, n, dim, norm}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::fill.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fill(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.fill.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::fill.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.fill.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.fill_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::fill_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.fill_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue flatten(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &start_dim, const PyTorch_IntValue &end_dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.flatten.using_ints";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, start_dim, end_dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::flip : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue flip(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.flip";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dims}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::FloatImplicit : (Tensor) -> (float)
PyTorch_FloatValue FloatImplicit(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.FloatImplicit";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Float.Scalar : (Scalar) -> (float)
PyTorch_FloatValue Float(const PyAnyTorchScalarValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Float.Scalar";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Float.str : (str) -> (float)
PyTorch_FloatValue Float(const PyTorch_StringValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Float.str";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Float.Tensor : (Tensor) -> (float)
PyTorch_FloatValue Float(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Float.Tensor";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::floor_divide : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.floor_divide";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue floor_divide(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.floor_divide.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::floor : (Tensor) -> (Tensor)
PyAnyTorchTensorValue floor(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.floor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::floor_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue floor_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.floor_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::floordiv.int : (int, int) -> (int)
PyTorch_IntValue floordiv(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.floordiv.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fmod(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.fmod.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::fmod_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue fmod_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.fmod_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::frobenius_norm.dim : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue frobenius_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.frobenius_norm.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::full_like : (Tensor, Scalar, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue full_like(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.full_like";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, fill_value, dtype, layout, device, pin_memory, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::full : (int[], Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue full(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchScalarValue &fill_value, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.full";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {size, fill_value, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue gather(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyTorch_BoolValue &sparse_grad, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gather";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, sparse_grad}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ge.float_int : (float, int) -> (bool)
PyTorch_BoolValue ge(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ge.float_int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ge.float : (float, float) -> (bool)
PyTorch_BoolValue ge(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ge.float";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ge.int : (int, int) -> (bool)
PyTorch_BoolValue ge(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ge.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ge(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ge.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ge.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ge(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ge.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ge_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ge_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ge_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ge_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ge_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gelu_backward : (Tensor, Tensor, str) -> (Tensor)
PyAnyTorchTensorValue gelu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gelu_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, self, approximate}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gelu : (Tensor, str) -> (Tensor)
PyAnyTorchTensorValue gelu(const PyAnyTorchTensorValue &self, const PyTorch_StringValue &approximate, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gelu";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, approximate}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gt.float_int : (float, int) -> (bool)
PyTorch_BoolValue gt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gt.float_int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gt.float : (float, float) -> (bool)
PyTorch_BoolValue gt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gt.float";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gt.int : (int, int) -> (bool)
PyTorch_BoolValue gt(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gt.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue gt(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gt.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue gt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gt.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gt_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gt_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::gt_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue gt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.gt_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::hardsigmoid : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardsigmoid(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.hardsigmoid";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::hardsigmoid_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardsigmoid_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.hardsigmoid_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::hardswish : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardswish(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.hardswish";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::hardswish_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue hardswish_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.hardswish_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::hardtanh_backward : (Tensor, Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.hardtanh_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, self, min_val, max_val}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::hardtanh : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.hardtanh";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, min_val, max_val}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::hardtanh_ : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue hardtanh_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &min_val, const PyAnyTorchScalarValue &max_val, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.hardtanh_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, min_val, max_val}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::imag : (Tensor) -> (Tensor)
PyAnyTorchTensorValue imag(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.imag";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::index_put.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.index_put.hacked_twin";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, indices, values, accumulate}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::index_put : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.index_put";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, indices, values, accumulate}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::index_put_.hacked_twin : (Tensor, Tensor[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.index_put_.hacked_twin";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, indices, values, accumulate}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::index_put_ : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue index_put_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.index_put_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, indices, values, accumulate}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::index_select : (Tensor, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue index_select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.index_select";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::index.Tensor_hacked_twin : (Tensor, Tensor[]) -> (Tensor)
PyAnyTorchTensorValue index(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTensorValue &indices, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.index.Tensor_hacked_twin";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, indices}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::index.Tensor : (Tensor, Tensor?[]) -> (Tensor)
PyAnyTorchTensorValue index(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.index.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, indices}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Int.bool : (bool) -> (int)
PyTorch_IntValue Int(const PyTorch_BoolValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Int.bool";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Int.float : (float) -> (int)
PyTorch_IntValue Int(const PyTorch_FloatValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Int.float";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::IntImplicit : (Tensor) -> (int)
PyTorch_IntValue IntImplicit(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.IntImplicit";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Int.Scalar : (Scalar) -> (int)
PyTorch_IntValue Int(const PyAnyTorchScalarValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Int.Scalar";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::Int.Tensor : (Tensor) -> (int)
PyTorch_IntValue Int(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.Int.Tensor";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::is_floating_point : (Tensor) -> (bool)
PyTorch_BoolValue is_floating_point(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.is_floating_point";
  auto _returnTypes = inferReturnTypes(operationName, {self}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::isnan : (Tensor) -> (Tensor)
PyAnyTorchTensorValue isnan(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.isnan";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::join : (str, str[]) -> (str)
PyTorch_StringValue join(const PyTorch_StringValue &self, const PyAnyTorchListOfTorchStringValue &values, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.join";
  auto _returnTypes = inferReturnTypes(operationName, {self, values}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, values}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::keys.str : (Dict(str, t)) -> (str[])
PyAnyTorchListOfTorchStringValue keys(const PyTorch_DictValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.keys.str";
  std::vector<PyType> _returnTypes = {PyAnyTorchListOfTorchStringType(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)
PyAnyTorchTensorValue layer_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, const PyTorch_BoolValue &cudnn_enable, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.layer_norm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, normalized_shape, weight, bias, eps, cudnn_enable}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::le.int : (int, int) -> (bool)
PyTorch_BoolValue le(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.le.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::le.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue le(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.le.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::le.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue le(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.le.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::le_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue le_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.le_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::le_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue le_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.le_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::leaky_relu_backward : (Tensor, Tensor, Scalar, bool) -> (Tensor)
PyAnyTorchTensorValue leaky_relu_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, const PyTorch_BoolValue &self_is_result, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.leaky_relu_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, self, negative_slope, self_is_result}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::leaky_relu : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue leaky_relu(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.leaky_relu";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, negative_slope}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::leaky_relu_ : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue leaky_relu_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &negative_slope, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.leaky_relu_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, negative_slope}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::len.str : (str) -> (int)
PyTorch_IntValue len(const PyTorch_StringValue &s, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.len.str";
  auto _returnTypes = inferReturnTypes(operationName, {s}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {s}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::len.t : (t[]) -> (int)
PyTorch_IntValue len(const PyAnyTorchListValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.len.t";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::len.Tensor : (Tensor) -> (int)
PyTorch_IntValue len(const PyAnyTorchTensorValue &t, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.len.Tensor";
  auto _returnTypes = inferReturnTypes(operationName, {t}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {t}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lerp(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lerp.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, end, weight}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lerp_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lerp_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &end, const PyAnyTorchTensorValue &weight, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lerp_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, end, weight}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lift_fresh_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue lift_fresh_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lift_fresh_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::linalg_vector_norm : (Tensor, Scalar, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue linalg_vector_norm(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &ord, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.linalg_vector_norm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, ord, dim, keepdim, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.linear";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::list.t : (t[]) -> (t[])
PyAnyTorchListValue list(const PyAnyTorchListValue &l, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.list.t";
  std::vector<PyType> _returnTypes = {PyAnyTorchListType(torchMlirTorchListTypeGetContainedType(mlirValueGetType(l)), DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {l}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::log1p : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log1p(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.log1p";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::log1p_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log1p_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.log1p_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::log2 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log2(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.log2";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::log2_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log2_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.log2_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::log.int : (int) -> (float)
PyTorch_FloatValue log(const PyTorch_IntValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.log.int";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::log : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.log";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.log_softmax.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::log_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue log_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.log_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logical_and : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_and(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logical_and";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logical_and_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_and_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logical_and_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logical_not : (Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_not(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logical_not";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logical_not_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_not_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logical_not_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logical_or : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_or(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logical_or";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logical_or_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_or_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logical_or_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logical_xor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_xor(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logical_xor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logical_xor_ : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue logical_xor_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logical_xor_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::logsumexp : (Tensor, int[], bool) -> (Tensor)
PyAnyTorchTensorValue logsumexp(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.logsumexp";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lt.float_int : (float, int) -> (bool)
PyTorch_BoolValue lt(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lt.float_int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lt.float : (float, float) -> (bool)
PyTorch_BoolValue lt(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lt.float";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lt.int : (int, int) -> (bool)
PyTorch_BoolValue lt(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lt.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue lt(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lt.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lt(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lt.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lt_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lt_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::lt_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue lt_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.lt_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.masked_fill.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, mask, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::masked_fill.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_fill(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.masked_fill.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, mask, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::masked_fill_.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.masked_fill_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, mask, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::masked_fill_.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_fill_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, const PyAnyTorchTensorValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.masked_fill_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, mask, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::masked_select : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue masked_select(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mask, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.masked_select";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, mask}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::matmul : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue matmul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.matmul";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::max.dim : (Tensor, int, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> max(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.max.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::max : (Tensor) -> (Tensor)
PyAnyTorchTensorValue max(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.max";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
PyAnyTorchTensorValue max_pool2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.max_pool2d";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, kernel_size, stride, padding, dilation, ceil_mode}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::max_pool2d_with_indices_backward : (Tensor, Tensor, int[], int[], int[], int[], bool, Tensor) -> (Tensor)
PyAnyTorchTensorValue max_pool2d_with_indices_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, const PyAnyTorchTensorValue &indices, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.max_pool2d_with_indices_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::max_pool2d_with_indices : (Tensor, int[], int[], int[], int[], bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> max_pool2d_with_indices(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &kernel_size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &ceil_mode, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.max_pool2d_with_indices";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, kernel_size, stride, padding, dilation, ceil_mode}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::maximum : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue maximum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.maximum";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mean.dim : (Tensor, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mean.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, keepdim, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mean : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mean";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::minimum : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue minimum(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.minimum";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mish : (Tensor) -> (Tensor)
PyAnyTorchTensorValue mish(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mish";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mm : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mm(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &mat2, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, mat2}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::movedim.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue movedim(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &source, const PyTorch_IntValue &destination, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.movedim.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, source, destination}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mse_loss_backward : (Tensor, Tensor, Tensor, int) -> (Tensor)
PyAnyTorchTensorValue mse_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mse_loss_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, self, target, reduction}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mse_loss : (Tensor, Tensor, int) -> (Tensor)
PyAnyTorchTensorValue mse_loss(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyTorch_IntValue &reduction, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mse_loss";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, target, reduction}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mul.float : (float, float) -> (float)
PyTorch_FloatValue mul(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mul.float";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mul.int : (int, int) -> (int)
PyTorch_IntValue mul(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mul.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue mul(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mul.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mul(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mul.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mul_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mul_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mul_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mul_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mul_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::mv : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue mv(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &vec, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.mv";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, vec}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::narrow : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue narrow(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &start, const PyTorch_IntValue &length, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.narrow";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, start, length}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::native_batch_norm_backward : (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_batch_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyAnyTorchOptionalTensorValue &save_mean, const PyAnyTorchOptionalTensorValue &save_invstd, const PyTorch_BoolValue &train, const PyTorch_FloatValue &eps, const PyAnyTorchListOfTorchBoolValue &output_mask, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.native_batch_norm_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)});
}
// aten::native_batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_batch_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchOptionalTensorValue &running_mean, const PyAnyTorchOptionalTensorValue &running_var, const PyTorch_BoolValue &training, const PyTorch_FloatValue &momentum, const PyTorch_FloatValue &eps, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.native_batch_norm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, running_mean, running_var, training, momentum, eps}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)});
}
// aten::native_dropout_backward : (Tensor, Tensor, float) -> (Tensor)
PyAnyTorchTensorValue native_dropout_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &mask, const PyTorch_FloatValue &scale, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.native_dropout_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, mask, scale}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::native_dropout : (Tensor, float, bool?) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_dropout(const PyAnyTorchTensorValue &input, const PyTorch_FloatValue &p, const PyAnyTorchOptionalBoolValue &train, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.native_dropout";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, p, train}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::native_group_norm_backward : (Tensor, Tensor, Tensor, Tensor, Tensor?, int, int, int, int, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_group_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &mean, const PyAnyTorchTensorValue &rstd, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &N, const PyTorch_IntValue &C, const PyTorch_IntValue &HxW, const PyTorch_IntValue &group, const PyAnyTorchListOfTorchBoolValue &output_mask, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.native_group_norm_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)});
}
// aten::native_group_norm : (Tensor, Tensor?, Tensor?, int, int, int, int, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_group_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_IntValue &N, const PyTorch_IntValue &C, const PyTorch_IntValue &HxW, const PyTorch_IntValue &group, const PyTorch_FloatValue &eps, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.native_group_norm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, N, C, HxW, group, eps}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)});
}
// aten::native_layer_norm_backward : (Tensor, Tensor, int[], Tensor, Tensor, Tensor?, Tensor?, bool[]) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_layer_norm_backward(const PyAnyTorchTensorValue &grad_out, const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchTensorValue &mean, const PyAnyTorchTensorValue &rstd, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchBoolValue &output_mask, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.native_layer_norm_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)});
}
// aten::native_layer_norm : (Tensor, int[], Tensor?, Tensor?, float) -> (Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> native_layer_norm(const PyAnyTorchTensorValue &input, const PyAnyTorchListOfTorchIntValue &normalized_shape, const PyAnyTorchOptionalTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyTorch_FloatValue &eps, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.native_layer_norm";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, normalized_shape, weight, bias, eps}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)});
}
// aten::ne.bool : (bool, bool) -> (bool)
PyTorch_BoolValue ne(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ne.bool";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ne.float_int : (float, int) -> (bool)
PyTorch_BoolValue ne(const PyTorch_FloatValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ne.float_int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ne.int_list : (int[], int[]) -> (bool)
PyTorch_BoolValue ne(const PyAnyTorchListOfTorchIntValue &a, const PyAnyTorchListOfTorchIntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ne.int_list";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ne.int : (int, int) -> (bool)
PyTorch_BoolValue ne(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ne.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ne(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ne.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ne(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ne.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ne_.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ne_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ne_.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue ne_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ne_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::neg.float : (float) -> (float)
PyTorch_FloatValue neg(const PyTorch_FloatValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.neg.float";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::neg.int : (int) -> (int)
PyTorch_IntValue neg(const PyTorch_IntValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.neg.int";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::neg : (Tensor) -> (Tensor)
PyAnyTorchTensorValue neg(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.neg";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::neg_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue neg_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.neg_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::new_empty : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_empty(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.new_empty";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::new_empty_strided : (Tensor, int[], int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_empty_strided(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.new_empty_strided";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, stride, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::new_ones : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_ones(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.new_ones";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::new_zeros : (Tensor, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue new_zeros(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.new_zeros";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::nll_loss2d_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue nll_loss2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.nll_loss2d_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, self, target, weight, reduction, ignore_index, total_weight}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::nll_loss2d_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> nll_loss2d_forward(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.nll_loss2d_forward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, target, weight, reduction, ignore_index}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::nll_loss_backward : (Tensor, Tensor, Tensor, Tensor?, int, int, Tensor) -> (Tensor)
PyAnyTorchTensorValue nll_loss_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, const PyAnyTorchTensorValue &total_weight, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.nll_loss_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, self, target, weight, reduction, ignore_index, total_weight}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::nll_loss_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> nll_loss_forward(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &target, const PyAnyTorchOptionalTensorValue &weight, const PyTorch_IntValue &reduction, const PyTorch_IntValue &ignore_index, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.nll_loss_forward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, target, weight, reduction, ignore_index}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::norm.ScalarOpt_dim : (Tensor, Scalar?, int[], bool) -> (Tensor)
PyAnyTorchTensorValue norm(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalScalarValue &p, const PyAnyTorchListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.norm.ScalarOpt_dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, p, dim, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::numel : (Tensor) -> (int)
PyTorch_IntValue numel(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.numel";
  auto _returnTypes = inferReturnTypes(operationName, {self}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::numpy_T : (Tensor) -> (Tensor)
PyAnyTorchTensorValue numpy_T(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.numpy_T";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::one_hot : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue one_hot(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &num_classes, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.one_hot";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, num_classes}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue ones_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ones_like";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype, layout, device, pin_memory, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue ones(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.ones";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {size, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::pad : (Tensor, int[], str, float?) -> (Tensor)
PyAnyTorchTensorValue pad(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &pad__, const PyTorch_StringValue &mode, const PyAnyTorchOptionalFloatValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.pad";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, pad__, mode, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::permute_copy : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue permute_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.permute_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dims}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::permute : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue permute(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &dims, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.permute";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dims}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::pow.int_float : (int, float) -> (float)
PyTorch_FloatValue pow(const PyTorch_IntValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.pow.int_float";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::pow.Scalar : (Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &exponent, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.pow.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, exponent}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &exponent, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.pow.Tensor_Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, exponent}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::pow.Tensor_Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue pow(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &exponent, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.pow.Tensor_Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, exponent}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::prelu : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue prelu(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &weight, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.prelu";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, weight}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::rand_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue rand_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.rand_like";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype, layout, device, pin_memory, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::randint.low : (int, int, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randint(const PyTorch_IntValue &low, const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.randint.low";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {low, high, size, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::randint : (int, int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randint(const PyTorch_IntValue &high, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.randint";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {high, size, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::randn.generator : (int[], Generator?, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalGeneratorValue &generator, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.randn.generator";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {size, generator, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::randn_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue randn_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.randn_like";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype, layout, device, pin_memory, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::randn : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue randn(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.randn";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {size, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::real : (Tensor) -> (Tensor)
PyAnyTorchTensorValue real(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.real";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::reciprocal : (Tensor) -> (Tensor)
PyAnyTorchTensorValue reciprocal(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.reciprocal";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::reciprocal_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue reciprocal_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.reciprocal_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::relu6 : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu6(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.relu6";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::relu6_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu6_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.relu6_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::relu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.relu";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::relu_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue relu_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.relu_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::remainder.int : (int, int) -> (int)
PyTorch_IntValue remainder(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.remainder.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::remainder.Scalar : (Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue remainder(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.remainder.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::repeat : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue repeat(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &repeats, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.repeat";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, repeats}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::reshape : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue reshape(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shape, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.reshape";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, shape}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::resize_ : (Tensor, int[], int?) -> (Tensor)
PyAnyTorchTensorValue resize_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.resize_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::roll : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue roll(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &shifts, const PyAnyTorchListOfTorchIntValue &dims, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.roll";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, shifts, dims}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::round : (Tensor) -> (Tensor)
PyAnyTorchTensorValue round(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.round";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::round_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue round_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.round_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::rsqrt : (Tensor) -> (Tensor)
PyAnyTorchTensorValue rsqrt(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.rsqrt";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::rsqrt_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue rsqrt_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.rsqrt_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue rsub(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.rsub.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scalar_tensor : (Scalar, int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue scalar_tensor(const PyAnyTorchScalarValue &s, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scalar_tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {s, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scaled_dot_product_attention : (Tensor, Tensor, Tensor, Tensor?, float, bool, float?) -> (Tensor)
PyAnyTorchTensorValue scaled_dot_product_attention(const PyAnyTorchTensorValue &query, const PyAnyTorchTensorValue &key, const PyAnyTorchTensorValue &value, const PyAnyTorchOptionalTensorValue &attn_mask, const PyTorch_FloatValue &dropout_p, const PyTorch_BoolValue &is_causal, const PyAnyTorchOptionalFloatValue &scale, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scaled_dot_product_attention";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {query, key, value, attn_mask, dropout_p, is_causal, scale}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scatter_add : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_add(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scatter_add";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, src}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scatter_add_ : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_add_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scatter_add_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, src}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scatter_reduce.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
PyAnyTorchTensorValue scatter_reduce(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scatter_reduce.two";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, src, reduce, include_self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scatter_reduce_.two : (Tensor, int, Tensor, Tensor, str, bool) -> (Tensor)
PyAnyTorchTensorValue scatter_reduce_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, const PyTorch_StringValue &reduce, const PyTorch_BoolValue &include_self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scatter_reduce_.two";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, src, reduce, include_self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scatter.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scatter.src";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, src}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scatter.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue scatter(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scatter.value";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scatter_.src : (Tensor, int, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue scatter_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchTensorValue &src, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scatter_.src";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, src}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::scatter_.value : (Tensor, int, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue scatter_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchTensorValue &index, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.scatter_.value";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::select_copy.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.select_copy.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::select.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.select.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, index}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::select_scatter : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue select_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyTorch_IntValue &index, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.select_scatter";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, src, dim, index}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sigmoid : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sigmoid(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sigmoid";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sigmoid_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sigmoid_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sigmoid_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sign : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sign(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sign";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sign_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sign_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sign_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::silu : (Tensor) -> (Tensor)
PyAnyTorchTensorValue silu(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.silu";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::silu_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue silu_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.silu_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sin : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sin(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sin";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sin_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sin_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sin_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::size.int : (Tensor, int) -> (int)
PyTorch_IntValue size(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.size.int";
  auto _returnTypes = inferReturnTypes(operationName, {self, dim}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::size : (Tensor) -> (int[])
PyAnyTorchListOfTorchIntValue size(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.size";
  std::vector<PyType> _returnTypes = {PyAnyTorchListOfTorchIntType(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::slice_copy.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.slice_copy.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, start, end, step}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::slice_scatter : (Tensor, Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice_scatter(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &src, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.slice_scatter";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, src, dim, start, end, step}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::slice.t : (t[], int?, int?, int) -> (t[])
PyAnyTorchListValue slice(const PyAnyTorchListValue &l, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.slice.t";
  std::vector<PyType> _returnTypes = {PyAnyTorchListType(torchMlirTorchListTypeGetContainedType(mlirValueGetType(l)), DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {l, start, end, step}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)
PyAnyTorchTensorValue slice(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &start, const PyAnyTorchOptionalIntValue &end, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.slice.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, start, end, step}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::softmax.int : (Tensor, int, int?) -> (Tensor)
PyAnyTorchTensorValue softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.softmax.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::softplus : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue softplus(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &beta, const PyAnyTorchScalarValue &threshold__, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.softplus";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, beta, threshold__}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sort.int : (int[], bool) -> ()
void sort(const PyAnyTorchListOfTorchIntValue &self, const PyTorch_BoolValue &reverse, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sort.int";
  std::vector<PyType> _returnTypes = {}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, reverse}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  // no result
}
// aten::sort : (Tensor, int, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> sort(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &descending, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sort";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, descending}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::split.Tensor : (Tensor, int, int) -> (Tensor[])
PyAnyTorchListOfTensorValue split(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &split_size, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.split.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchListOfTensorType(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, split_size, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sqrt.int : (int) -> (float)
PyTorch_FloatValue sqrt(const PyTorch_IntValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sqrt.int";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sqrt : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sqrt(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sqrt";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sqrt_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue sqrt_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sqrt_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::square : (Tensor) -> (Tensor)
PyAnyTorchTensorValue square(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.square";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::square_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue square_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.square_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::squeeze_copy.dim : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue squeeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.squeeze_copy.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::squeeze_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue squeeze_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.squeeze_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::squeeze.dim : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.squeeze.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::squeeze : (Tensor) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.squeeze";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::stack : (Tensor[], int) -> (Tensor)
PyAnyTorchTensorValue stack(const PyAnyTorchListOfTensorValue &tensors, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.stack";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {tensors, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::std.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.std.correction";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, correction, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::std.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.std.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, unbiased, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::std : (Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue std(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.std";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, unbiased}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sub.float : (float, float) -> (float)
PyTorch_FloatValue sub(const PyTorch_FloatValue &a, const PyTorch_FloatValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sub.float";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sub.int : (int, int) -> (int)
PyTorch_IntValue sub(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sub.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sub.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sub.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sub_.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sub_.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sub_.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue sub_(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyAnyTorchScalarValue &alpha, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sub_.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, alpha}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sum.dim_IntList : (Tensor, int[]?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &keepdim, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sum.dim_IntList";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, keepdim, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::sum : (Tensor, int?) -> (Tensor)
PyAnyTorchTensorValue sum(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.sum";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::t_copy : (Tensor) -> (Tensor)
PyAnyTorchTensorValue t_copy(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.t_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::t : (Tensor) -> (Tensor)
PyAnyTorchTensorValue t(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.t";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tanh_backward : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tanh_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, output}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tanh : (Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tanh";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tanh_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue tanh_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tanh_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_BoolValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tensor.bool";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {t, dtype, device, requires_grad}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_FloatValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tensor.float";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {t, dtype, device, requires_grad}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyTorch_IntValue &t, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tensor.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {t, dtype, device, requires_grad}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tensor : (t[], int?, Device?, bool) -> (Tensor)
PyAnyTorchTensorValue tensor(const PyAnyTorchListValue &data, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalDeviceValue &device, const PyTorch_BoolValue &requires_grad, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {data, dtype, device, requires_grad}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.threshold_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, self, threshold__}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.threshold";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, threshold__, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::threshold_ : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue threshold_(const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &threshold__, const PyAnyTorchScalarValue &value, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.threshold_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, threshold__, value}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::to.device : (Tensor, Device, int, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyTorch_DeviceValue &device, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.to.device";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, device, dtype, non_blocking, copy, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::to.dtype_layout : (Tensor, int?, int?, Device?, bool?, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.to.dtype_layout";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.to.dtype";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype, non_blocking, copy, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.to.other";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other, non_blocking, copy, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue to(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalIntValue &dtype, const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.to.prim_Device";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, device, dtype, non_blocking, copy}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::topk : (Tensor, int, int, bool, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> topk(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &k, const PyTorch_IntValue &dim, const PyTorch_BoolValue &largest, const PyTorch_BoolValue &sorted, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.topk";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, k, dim, largest, sorted}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::transpose_copy.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue transpose_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.transpose_copy.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim0, dim1}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::transpose.int : (Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue transpose(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim0, const PyTorch_IntValue &dim1, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.transpose.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim0, dim1}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tril : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue tril(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tril";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, diagonal}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::tril_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue tril_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.tril_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, diagonal}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::triu : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue triu(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.triu";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, diagonal}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::triu_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue triu_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &diagonal, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.triu_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, diagonal}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::type_as : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue type_as(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.type_as";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::unbind.int : (Tensor, int) -> (Tensor[])
PyAnyTorchListOfTensorValue unbind(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.unbind.int";
  std::vector<PyType> _returnTypes = {PyAnyTorchListOfTensorType(PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::unfold_copy : (Tensor, int, int, int) -> (Tensor)
PyAnyTorchTensorValue unfold_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dimension, const PyTorch_IntValue &size, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.unfold_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dimension, size, step}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::uniform : (Tensor, float, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue uniform(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.uniform";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, from, to, generator}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::uniform_ : (Tensor, float, float, Generator?) -> (Tensor)
PyAnyTorchTensorValue uniform_(const PyAnyTorchTensorValue &self, const PyTorch_FloatValue &from, const PyTorch_FloatValue &to, const PyAnyTorchOptionalGeneratorValue &generator, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.uniform_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, from, to, generator}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::unsqueeze_copy : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.unsqueeze_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::unsqueeze : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.unsqueeze";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::unsqueeze_ : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue unsqueeze_(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.unsqueeze_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::upsample_nearest2d_backward : (Tensor, int[], int[], float?, float?) -> (Tensor)
PyAnyTorchTensorValue upsample_nearest2d_backward(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchListOfTorchIntValue &input_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.upsample_nearest2d_backward";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, output_size, input_size, scales_h, scales_w}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::upsample_nearest2d : (Tensor, int[], float?, float?) -> (Tensor)
PyAnyTorchTensorValue upsample_nearest2d(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &output_size, const PyAnyTorchOptionalFloatValue &scales_h, const PyAnyTorchOptionalFloatValue &scales_w, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.upsample_nearest2d";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, output_size, scales_h, scales_w}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::var.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.var.correction";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, correction, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::var.dim : (Tensor, int[]?, bool, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.var.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, unbiased, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::var_mean.correction : (Tensor, int[]?, Scalar?, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyAnyTorchOptionalScalarValue &correction, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.var_mean.correction";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, correction, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::var_mean.dim : (Tensor, int[]?, bool, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalListOfTorchIntValue &dim, const PyTorch_BoolValue &unbiased, const PyTorch_BoolValue &keepdim, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.var_mean.dim";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, unbiased, keepdim}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::var_mean : (Tensor, bool) -> (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> var_mean(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.var_mean";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, unbiased}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)});
}
// aten::var : (Tensor, bool) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &self, const PyTorch_BoolValue &unbiased, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.var";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, unbiased}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::view_as_complex : (Tensor) -> (Tensor)
PyAnyTorchTensorValue view_as_complex(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.view_as_complex";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::view_copy.dtype : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue view_copy(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.view_copy.dtype";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::view_copy : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue view_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.view_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::view : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.view";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::where.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.where.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {condition, self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::where.ScalarOther : (Tensor, Tensor, Scalar) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.where.ScalarOther";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {condition, self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::where.ScalarSelf : (Tensor, Scalar, Tensor) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchScalarValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.where.ScalarSelf";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {condition, self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue where(const PyAnyTorchTensorValue &condition, const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.where.self";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {condition, self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::zero : (Tensor) -> (Tensor)
PyAnyTorchTensorValue zero(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.zero";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::zero_ : (Tensor) -> (Tensor)
PyAnyTorchTensorValue zero_(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.zero_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)
PyAnyTorchTensorValue zeros_like(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.zeros_like";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype, layout, device, pin_memory, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)
PyAnyTorchTensorValue zeros(const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.zeros";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {size, dtype, layout, device, pin_memory}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_convolution.deprecated : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._convolution.deprecated";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_convolution : (Tensor, Tensor, Tensor?, int[], int[], int[], bool, int[], int, bool, bool, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _convolution(const PyAnyTorchTensorValue &input, const PyAnyTorchTensorValue &weight, const PyAnyTorchOptionalTensorValue &bias, const PyAnyTorchListOfTorchIntValue &stride, const PyAnyTorchListOfTorchIntValue &padding, const PyAnyTorchListOfTorchIntValue &dilation, const PyTorch_BoolValue &transposed, const PyAnyTorchListOfTorchIntValue &output_padding, const PyTorch_IntValue &groups, const PyTorch_BoolValue &benchmark, const PyTorch_BoolValue &deterministic, const PyTorch_BoolValue &cudnn_enabled, const PyTorch_BoolValue &allow_tf32, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._convolution";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_embedding_bag : (Tensor, Tensor, Tensor, bool, int, bool, Tensor?, bool, int) -> (Tensor, Tensor, Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue> _embedding_bag(const PyAnyTorchTensorValue &weight, const PyAnyTorchTensorValue &indices, const PyAnyTorchTensorValue &offsets, const PyTorch_BoolValue &scale_grad_by_freq, const PyTorch_IntValue &mode, const PyTorch_BoolValue &sparse, const PyAnyTorchOptionalTensorValue &per_sample_weights, const PyTorch_BoolValue &include_last_offset, const PyTorch_IntValue &padding_idx, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._embedding_bag";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve()), PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue, PyAnyTorchTensorValue>({opRef, mlirOperationGetResult(operation, 0)}, {opRef, mlirOperationGetResult(operation, 1)}, {opRef, mlirOperationGetResult(operation, 2)}, {opRef, mlirOperationGetResult(operation, 3)});
}
// aten::_index_put_impl : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _index_put_impl(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, const PyTorch_BoolValue &unsafe, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._index_put_impl";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, indices, values, accumulate, unsafe}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_index_put_impl_ : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)
PyAnyTorchTensorValue _index_put_impl_(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfOptionalTensorValue &indices, const PyAnyTorchTensorValue &values, const PyTorch_BoolValue &accumulate, const PyTorch_BoolValue &unsafe, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._index_put_impl_";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, indices, values, accumulate, unsafe}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue _log_softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._log_softmax_backward_data";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, output, dim, input_dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_log_softmax : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue _log_softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._log_softmax";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, half_to_float}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_reshape_alias_copy : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue _reshape_alias_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._reshape_alias_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, stride}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_reshape_alias : (Tensor, int[], int[]) -> (Tensor)
PyAnyTorchTensorValue _reshape_alias(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, const PyAnyTorchListOfTorchIntValue &stride, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._reshape_alias";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size, stride}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_shape_as_tensor : (Tensor) -> (Tensor)
PyAnyTorchTensorValue _shape_as_tensor(const PyAnyTorchTensorValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._shape_as_tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)
PyAnyTorchTensorValue _softmax_backward_data(const PyAnyTorchTensorValue &grad_output, const PyAnyTorchTensorValue &output, const PyTorch_IntValue &dim, const PyTorch_IntValue &input_dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._softmax_backward_data";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {grad_output, output, dim, input_dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_softmax : (Tensor, int, bool) -> (Tensor)
PyAnyTorchTensorValue _softmax(const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim, const PyTorch_BoolValue &half_to_float, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._softmax";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dim, half_to_float}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_to_copy : (Tensor, int?, int?, Device?, bool?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue _to_copy(const PyAnyTorchTensorValue &self, const PyAnyTorchOptionalIntValue &dtype, const PyAnyTorchOptionalIntValue &layout, const PyAnyTorchOptionalDeviceValue &device, const PyAnyTorchOptionalBoolValue &pin_memory, const PyTorch_BoolValue &non_blocking, const PyAnyTorchOptionalIntValue &memory_format, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._to_copy";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, dtype, layout, device, pin_memory, non_blocking, memory_format}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::_unsafe_view : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue _unsafe_view(const PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten._unsafe_view";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, size}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::__and__.bool : (bool, bool) -> (bool)
PyTorch_BoolValue __and__(const PyTorch_BoolValue &a, const PyTorch_BoolValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.__and__.bool";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)
PyAnyTorchTensorValue __and__(const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.__and__.Tensor";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self, other}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::__contains__.int_list : (int[], int) -> (bool)
PyTorch_BoolValue __contains__(const PyAnyTorchListOfTorchIntValue &l, const PyTorch_IntValue &item, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.__contains__.int_list";
  auto _returnTypes = inferReturnTypes(operationName, {l, item}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {l, item}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::__contains__.str : (Dict(str, t), str) -> (bool)
PyTorch_BoolValue __contains__(const PyTorch_DictValue &dict, const PyTorch_StringValue &key, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.__contains__.str";
  auto _returnTypes = inferReturnTypes(operationName, {dict, key}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {dict, key}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::__derive_index : (int, int, int) -> (int)
PyTorch_IntValue __derive_index(const PyTorch_IntValue &index, const PyTorch_IntValue &start, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.__derive_index";
  auto _returnTypes = inferReturnTypes(operationName, {index, start, step}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {index, start, step}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::__not__ : (bool) -> (bool)
PyTorch_BoolValue __not__(const PyTorch_BoolValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.__not__";
  auto _returnTypes = inferReturnTypes(operationName, {self}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// aten::__range_length : (int, int, int) -> (int)
PyTorch_IntValue __range_length(const PyTorch_IntValue &lo, const PyTorch_IntValue &hi, const PyTorch_IntValue &step, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.__range_length";
  auto _returnTypes = inferReturnTypes(operationName, {lo, hi, step}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {lo, hi, step}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::device : (Tensor) -> (Device)
PyTorch_DeviceValue device(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.device";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::dtype : (Tensor) -> (int)
PyTorch_IntValue dtype(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.dtype";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::layout : (Tensor) -> (int)
PyTorch_IntValue layout(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.layout";
  auto _returnTypes = inferReturnTypes(operationName, {a}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::max.int : (int, int) -> (int)
PyTorch_IntValue max(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.max.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::max.self_int : (int[]) -> (int)
PyTorch_IntValue max(const PyAnyTorchListOfTorchIntValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.max.self_int";
  auto _returnTypes = inferReturnTypes(operationName, {self}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::min.int : (int, int) -> (int)
PyTorch_IntValue min(const PyTorch_IntValue &a, const PyTorch_IntValue &b, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.min.int";
  auto _returnTypes = inferReturnTypes(operationName, {a, b}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, b}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::min.self_int : (int[]) -> (int)
PyTorch_IntValue min(const PyAnyTorchListOfTorchIntValue &self, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.min.self_int";
  auto _returnTypes = inferReturnTypes(operationName, {self}, loc->getContext().get(), loc); 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {self}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::NumToTensor.Scalar : (Scalar) -> (Tensor)
PyAnyTorchTensorValue NumToTensor(const PyAnyTorchScalarValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.NumToTensor.Scalar";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prim::RaiseException : (str, str?) -> ()
void RaiseException(const PyTorch_StringValue &msg, const PyAnyTorchOptionalStringValue &cls, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prim.RaiseException";
  std::vector<PyType> _returnTypes = {}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {msg, cls}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  // no result
}
// prims::convert_element_type : (Tensor, int) -> (Tensor)
PyAnyTorchTensorValue convert_element_type(const PyAnyTorchTensorValue &a, const PyTorch_IntValue &dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prims.convert_element_type";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prims::squeeze : (Tensor, int[]) -> (Tensor)
PyAnyTorchTensorValue squeeze(const PyAnyTorchTensorValue &a, const PyAnyTorchListOfTorchIntValue &dimensions, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prims.squeeze";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a, dimensions}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prims::var : (Tensor, int[]?, float, int?) -> (Tensor)
PyAnyTorchTensorValue var(const PyAnyTorchTensorValue &inp, const PyAnyTorchOptionalListOfTorchIntValue &dims, const PyTorch_FloatValue &correction, const PyAnyTorchOptionalIntValue &output_dtype, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prims.var";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {inp, dims, correction, output_dtype}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// prims::view_of : (Tensor) -> (Tensor)
PyAnyTorchTensorValue view_of(const PyAnyTorchTensorValue &a, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.prims.view_of";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {a}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
// quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &X, const PyTorch_LinearParamsValue &W_prepack, const PyTorch_FloatValue &Y_scale_i, const PyTorch_IntValue &Y_zero_point_i, PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.quantized.linear";
  std::vector<PyType> _returnTypes = {PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())}; 
  std::vector<std::reference_wrapper<const PyType>> returnTypes; 
  for (const auto& returnType : _returnTypes) 
    returnTypes.emplace_back(returnType);
  PyOperationRef opRef = createOperation(operationName,
            returnTypes,
            {X, W_prepack, Y_scale_i, Y_zero_point_i}, 
            /*attributes=*/{}, 
            loc, 
            ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}
