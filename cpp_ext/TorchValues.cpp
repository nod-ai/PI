//
// Created by maksim on 5/13/23.
//

#include "TorchValues.h"
#include "TorchTypes.h"

#include <pybind11/pybind11.h>

namespace mlir::torch {

bool isAAnyTorchDictKeyValue(MlirValue value) {
  return isAAnyTorchDictKeyType(mlirValueGetType(value));
}
bool isAAnyTorchListOfOptionalIntValue(MlirValue value) {
  return isAAnyTorchListOfOptionalIntType(mlirValueGetType(value));
}
bool isAAnyTorchListOfOptionalTensorValue(MlirValue value) {
  return isAAnyTorchListOfOptionalTensorType(mlirValueGetType(value));
}
bool isAAnyTorchListOfTensorValue(MlirValue value) {
  return isAAnyTorchListOfTensorType(mlirValueGetType(value));
}
bool isAAnyTorchListOfTorchBoolValue(MlirValue value) {
  return isAAnyTorchListOfTorchBoolType(mlirValueGetType(value));
}
bool isAAnyTorchListOfTorchIntValue(MlirValue value) {
  return isAAnyTorchListOfTorchIntType(mlirValueGetType(value));
}
bool isAAnyTorchListOfTorchStringValue(MlirValue value) {
  return isAAnyTorchListOfTorchStringType(mlirValueGetType(value));
}
bool isAAnyTorchOptionalFloatValue(MlirValue value) {
  return isAAnyTorchOptionalFloatType(mlirValueGetType(value));
}
bool isAAnyTorchListValue(MlirValue value) {
  return isAAnyTorchListType(mlirValueGetType(value));
}
bool isAAnyTorchOptionalBoolValue(MlirValue value) {
  return isAAnyTorchOptionalBoolType(mlirValueGetType(value));
}
bool isAAnyTorchOptionalDeviceValue(MlirValue value) {
  return isAAnyTorchOptionalDeviceType(mlirValueGetType(value));
}
bool isAAnyTorchOptionalGeneratorValue(MlirValue value) {
  return isAAnyTorchOptionalGeneratorType(mlirValueGetType(value));
}
bool isAAnyTorchOptionalIntValue(MlirValue value) {
  return isAAnyTorchOptionalIntType(mlirValueGetType(value));
}
bool isAAnyTorchOptionalListOfTorchIntValue(MlirValue value) {
  return isAAnyTorchOptionalListOfTorchIntType(mlirValueGetType(value));
}
bool isAAnyTorchOptionalStringValue(MlirValue value) {
  return isAAnyTorchOptionalStringType(mlirValueGetType(value));
}
bool isAAnyTorchOptionalTensorValue(MlirValue value) {
  return isAAnyTorchOptionalTensorType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalValue(MlirValue value) {
  return isAAnyTorchOptionalType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalScalarValue(MlirValue value) {
  return isAAnyTorchOptionalScalarType(mlirValueGetType(value));
}
bool isAAnyTorchScalarValue(MlirValue value) {
  return isAAnyTorchScalarType(mlirValueGetType(value));
}

bool isAAnyTorchTensorValue(MlirValue value) {
  return isAAnyTorchTensorType(mlirValueGetType(value));
}

bool isAAnyTorchValue(MlirValue value) {
  return isAAnyTorchType(mlirValueGetType(value));
}

#define DECLARE_ISA_UNDERSCORE_VALUE(UNDERSCOREVALUE)                          \
  bool isATorch_##UNDERSCOREVALUE##Value(MlirValue value) {                    \
    return isATorch_##UNDERSCOREVALUE##Type(mlirValueGetType(value));          \
  }
FORALL_UNDERSCORE_TYPES(DECLARE_ISA_UNDERSCORE_VALUE)
#undef DECLARE_ISA_UNDERSCORE_VALUE

// these are here (even though they're empty) as a reminder (because it might
// be hard to miss in the macros...
void PyAnyTorchListValue::bindDerived(ClassTy &c) {}

#define DEFINE_LIST_BASE_CONCRETE_VALUE(CONCRETEVALUE)                         \
  void PyAnyTorchListOf##CONCRETEVALUE##Value::bindDerived(ClassTy &c) {}
FORALL_LIST_BASE_CONCRETE_TYPES(DEFINE_LIST_BASE_CONCRETE_VALUE)
DEFINE_LIST_BASE_CONCRETE_VALUE(Tensor)
#undef DEFINE_LIST_BASE_CONCRETE_VALUE

#define DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(CONCRETEVALUE)                     \
  void PyAnyTorchOptional##CONCRETEVALUE##Value::bindDerived(ClassTy &c) {}
FORALL_OPTIONAL_BASE_CONCRETE_TYPES(DEFINE_OPTIONAL_BASE_CONCRETE_VALUE)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE()
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Tensor)
#undef DEFINE_OPTIONAL_BASE_CONCRETE_VALUE

#define DEFINE_SCALAR_VALUE(SCALARVALUE)                                       \
  void PyTorch_##SCALARVALUE##Value::bindDerived(ClassTy &c) {}
FORALL_SCALAR_TYPES(DEFINE_SCALAR_VALUE)
#undef DEFINE_SCALAR_VALUE

void PyTorch_DictValue::bindDerived(ClassTy &c) {}
void PyTorch_TupleValue::bindDerived(ClassTy &c) {}
void PyTorch_NnModuleValue::bindDerived(ClassTy &c) {}
void PyTorch_NonValueTensorValue::bindDerived(ClassTy &c) {}
void PyTorch_ValueTensorValue::bindDerived(ClassTy &c) {}

////////////////////////////////////////////////////////////////////////////////

void populateTorchMLIRValues(py::module &m) {
  PyAnyTorchListValue::bind(m);
  PyAnyTorchOptionalValue::bind(m);

#define BIND_VALUE(VALUE) PyAnyTorchListOf##VALUE##Value::bind(m);
  FORALL_LIST_BASE_CONCRETE_TYPES(BIND_VALUE)
  BIND_VALUE(Tensor)
#undef BIND_VALUE
#define BIND_VALUE(VALUE) PyAnyTorchOptional##VALUE##Value::bind(m);
  FORALL_OPTIONAL_BASE_CONCRETE_TYPES(BIND_VALUE)
  BIND_VALUE(Tensor)
#undef BIND_VALUE
#define BIND_VALUE(VALUE) PyTorch_##VALUE##Value::bind(m);
  FORALL_SCALAR_TYPES(BIND_VALUE)
#undef BIND_VALUE

  PyTorch_DictValue::bind(m);
  PyTorch_TupleValue::bind(m);
  PyTorch_NnModuleValue::bind(m);
  PyTorch_NonValueTensorValue::bind(m);
  PyTorch_ValueTensorValue::bind(m);
  PyAnyTorchScalarValue::bind(m);
}

} // namespace mlir::torch