//
// Created by maksim on 5/13/23.
//

#include "TorchValues.h"
#include "TorchTypes.h"

using llvm::Twine;

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
  return torchMlirTypeIsATorchOptional(mlirValueGetType(value));
}

// bool isAAnyTorchOptionalScalarValue(MlirValue value) ;
// bool isAAnyTorchScalarValue(MlirValue value) ;

bool isAAnyTorchTensorValue(MlirValue value) {
  return isAAnyTorchTensorType(mlirValueGetType(value));
}

// bool isAAnyTorchValue(MlirValue value) ;
bool isATorch_BoolValue(MlirValue value) {
  return isATorch_BoolType(mlirValueGetType(value));
}
bool isATorch_DeviceValue(MlirValue value) {
  return isATorch_DeviceType(mlirValueGetType(value));
}
bool isATorch_DictValue(MlirValue value) {
  return isATorch_DictType(mlirValueGetType(value));
}
bool isATorch_FloatValue(MlirValue value) {
  return isATorch_FloatType(mlirValueGetType(value));
}
bool isATorch_IntValue(MlirValue value) {
  return isATorch_IntType(mlirValueGetType(value));
}
bool isATorch_LinearParamsValue(MlirValue value) {
  return isATorch_LinearParamsType(mlirValueGetType(value));
}
bool isATorch_NnModuleValue(MlirValue value) {
  return isATorch_NnModuleType(mlirValueGetType(value));
}
bool isATorch_NonValueTensorValue(MlirValue value) {
  return isATorch_NonValueTensorType(mlirValueGetType(value));
}
bool isATorch_NoneValue(MlirValue value) {
  return isATorch_NoneType(mlirValueGetType(value));
}
bool isATorch_NumberValue(MlirValue value) {
  return isATorch_NumberType(mlirValueGetType(value));
}
bool isATorch_StringValue(MlirValue value) {
  return isATorch_StringType(mlirValueGetType(value));
}
bool isATorch_TupleValue(MlirValue value) {
  return isATorch_TupleType(mlirValueGetType(value));
}
bool isATorch_ValueTensorValue(MlirValue value) {
  return isATorch_ValueTensorType(mlirValueGetType(value));
}

template <typename DerivedTy>
MlirValue PyConcreteValue<DerivedTy>::castFrom(PyValue &orig) {
  if (!DerivedTy::isaFunction(orig.get())) {
    auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
    throw SetPyError(PyExc_ValueError, Twine("Cannot cast value to ") +
                                           DerivedTy::pyClassName + " (from " +
                                           origRepr + ")");
  }
  return orig.get();
}

template <typename DerivedTy>
void PyConcreteValue<DerivedTy>::bind(py::module &m) {
  auto cls = ClassTy(m, DerivedTy::pyClassName);
  cls.def(py::init<PyValue &>(), py::keep_alive<0, 1>(), py::arg("value"));
  cls.def_static(
      "isinstance",
      [](PyValue &otherValue) -> bool {
        return DerivedTy::isaFunction(otherValue);
      },
      py::arg("other_value"));
  cls.def("__str__", [](const py::object &self) {
    auto Value =
        py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Value");
    return py::str(Value(self))
        .attr("replace")("Value", DerivedTy::pyClassName);
  });
  DerivedTy::bindDerived(cls);
}

// these are here (even though they're empty) as a reminder (because it might
// be hard to miss in the macros...
void PyAnyTorchListValue::bindDerived(ClassTy &c) {}

void PyAnyTorchOptionalValue::bindDerived(ClassTy &c) {}

#define DECLARE_LIST_BASE_CONCRETE_VALUE(CONCRETEVALUE)                        \
  void PyAnyTorchListOf##CONCRETEVALUE##Value::bindDerived(ClassTy &c) {}
FORALL_LIST_BASE_CONCRETE_TYPES(DECLARE_LIST_BASE_CONCRETE_VALUE)
#undef DECLARE_LIST_BASE_CONCRETE_VALUE

#define DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(CONCRETEVALUE)                    \
  void PyAnyTorchOptional##CONCRETEVALUE##Value::bindDerived(ClassTy &c) {}
FORALL_OPTIONAL_BASE_CONCRETE_TYPES(DECLARE_OPTIONAL_BASE_CONCRETE_VALUE)
#undef DECLARE_OPTIONAL_BASE_CONCRETE_VALUE

#define DECLARE_CONCRETE_VALUE(CONCRETEVALUE)                                  \
  void PyTorch##CONCRETEVALUE##Value::bindDerived(ClassTy &c) {}
FORALL_CONCRETE_TYPES(DECLARE_CONCRETE_VALUE)
#undef DECLARE_CONCRETE_VALUE

////////////////////////////////////////////////////////////////////////////////

void populateTorchMLIRValues(py::module &m) {
  PyAnyTorchListValue::bind(m);
  PyAnyTorchOptionalValue::bind(m);

#define BIND_VALUE(VALUE) PyAnyTorchListOf##VALUE##Value::bind(m);
  FORALL_LIST_BASE_CONCRETE_TYPES(BIND_VALUE)
#undef BIND_VALUE
#define BIND_VALUE(VALUE) PyAnyTorchOptional##VALUE##Value::bind(m);
  FORALL_OPTIONAL_BASE_CONCRETE_TYPES(BIND_VALUE)
#undef BIND_VALUE
#define BIND_VALUE(VALUE) PyTorch##VALUE##Value::bind(m);
  FORALL_CONCRETE_TYPES(BIND_VALUE)
#undef BIND_VALUE
}

} // namespace mlir::torch