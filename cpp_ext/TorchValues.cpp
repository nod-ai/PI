//
// Created by maksim on 5/13/23.
//

#include "TorchValues.h"
#include "TorchTypes.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace mlir::torch {

bool isAAnyTorchDictKeyValue(MlirValue value) {
  return isAAnyTorchDictKeyType(mlirValueGetType(value));
}

bool isAAnyTorchListOfOptionalIntValue(MlirValue value) {
  return isAAnyTorchListOfOptionalIntType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchBoolValue(MlirValue value) {
  return isAAnyTorchListOfTorchBoolType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchIntValue(MlirValue value) {
  return isAAnyTorchListOfTorchIntType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchFloatValue(MlirValue value) {
  return isAAnyTorchListOfTorchFloatType(mlirValueGetType(value));
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

bool isAAnyTorchOptionalValue(MlirValue value) {
  return isAAnyTorchOptionalType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalScalarValue(MlirValue value) {
  return isAAnyTorchOptionalScalarType(mlirValueGetType(value));
}

bool isAAnyTorchScalarValue(MlirValue value) {
  return isAAnyTorchScalarType(mlirValueGetType(value));
}

bool isAAnyTorchValue(MlirValue value) {
  return isAAnyTorchType(mlirValueGetType(value));
}

#define DECLARE_ISA_UNDERSCORE_VALUE(TORCHTYPE)                                \
  bool isATorch_##TORCHTYPE##Value(MlirValue value) {                          \
    return isATorch_##TORCHTYPE##Type(mlirValueGetType(value));                \
  }
FORALL_UNDERSCORE_TYPES(DECLARE_ISA_UNDERSCORE_VALUE)
#undef DECLARE_ISA_UNDERSCORE_VALUE

// these are here (even though they're empty) as a reminder (because it might
// be hard to miss in the macros...
void PyAnyTorchListValue::bindDerived(ClassTy &c) {}

#define DEFINE_LIST_BASE_CONCRETE_VALUE(TORCHTYPE)                             \
  void PyAnyTorchListOf##TORCHTYPE##Value::bindDerived(ClassTy &c) {           \
    c.def(py::init<py::list>(), py::arg("value"));                             \
    c.def(py::init<py::tuple>(), py::arg("value"));                            \
    py::implicitly_convertible<py::list,                                       \
                               PyAnyTorchListOf##TORCHTYPE##Value>();          \
    py::implicitly_convertible<py::tuple,                                      \
                               PyAnyTorchListOf##TORCHTYPE##Value>();          \
  }
FORALL_LIST_BASE_CONCRETE_TYPES(DEFINE_LIST_BASE_CONCRETE_VALUE)
#undef DEFINE_LIST_BASE_CONCRETE_VALUE

void PyAnyTorchOptionalGeneratorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalGeneratorValue>();
}

void PyAnyTorchOptionalValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalValue>();
}

#define DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(TORCHTYPE, CPPTYPE)                \
  void PyAnyTorchOptional##TORCHTYPE##Value::bindDerived(ClassTy &c) {         \
    c.def(py::init<py::none>(), py::arg("value"));                             \
    c.def(py::init<CPPTYPE>(), py::arg("value"));                              \
    py::implicitly_convertible<py::none,                                       \
                               PyAnyTorchOptional##TORCHTYPE##Value>();        \
    py::implicitly_convertible<CPPTYPE,                                        \
                               PyAnyTorchOptional##TORCHTYPE##Value>();        \
  }
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Bool, bool)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Device, int)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Int, int)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Float, float)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(String, std::string)
#undef DEFINE_OPTIONAL_BASE_CONCRETE_VALUE

void PyAnyTorchOptionalScalarValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  c.def(py::init<int>(), py::arg("value"));
  c.def(py::init<float>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalScalarValue>();
  py::implicitly_convertible<int, PyAnyTorchOptionalScalarValue>();
  py::implicitly_convertible<float, PyAnyTorchOptionalScalarValue>();
}

void PyAnyTorchOptionalListOfTorchIntValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  c.def(py::init<py::list>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalListOfTorchIntValue>();
  py::implicitly_convertible<py::list, PyAnyTorchOptionalListOfTorchIntValue>();
  py::implicitly_convertible<py::tuple,
                             PyAnyTorchOptionalListOfTorchIntValue>();
}

#define DEFINE_BIND_SCALAR_VALUE(TORCHTYPE)                                    \
  void PyTorch_##TORCHTYPE##Value::bindDerived(ClassTy &c) {}
DEFINE_BIND_SCALAR_VALUE(Any)
DEFINE_BIND_SCALAR_VALUE(LinearParams)
DEFINE_BIND_SCALAR_VALUE(None)
DEFINE_BIND_SCALAR_VALUE(Number)
#undef DEFINE_BIND_SCALAR_VALUE

#define DEFINE_BIND_SCALAR_VALUE(TORCHTYPE, CPPTYPE)                           \
  void PyTorch_##TORCHTYPE##Value::bindDerived(ClassTy &c) {                   \
    c.def(py::init<CPPTYPE>(), py::arg("value"));                              \
    py::implicitly_convertible<CPPTYPE, PyTorch_##TORCHTYPE##Value>();         \
  }
DEFINE_BIND_SCALAR_VALUE(Bool, bool)
DEFINE_BIND_SCALAR_VALUE(Device, int)
DEFINE_BIND_SCALAR_VALUE(Int, int)
DEFINE_BIND_SCALAR_VALUE(Float, float)
DEFINE_BIND_SCALAR_VALUE(String, std::string)
#undef DEFINE_BIND_SCALAR_VALUE

void PyTorch_DictValue::bindDerived(ClassTy &c) {}
void PyTorch_TupleValue::bindDerived(ClassTy &c) {}
void PyTorch_NnModuleValue::bindDerived(ClassTy &c) {}
void PyAnyTorchScalarValue::bindDerived(ClassTy &c) {
  c.def("__repr__", [](PyAnyTorchScalarValue &self) {
    auto origRepr =
        pybind11::repr(pybind11::cast(PyValue(self))).cast<std::string>();
    return std::regex_replace(origRepr, std::regex("Value"),
                              "AnyTorchScalarValue");
  });
  c.def(py::init<int>(), py::arg("value"));
  c.def(py::init<float>(), py::arg("value"));
  py::implicitly_convertible<int, PyAnyTorchScalarValue>();
  py::implicitly_convertible<float, PyAnyTorchScalarValue>();
};

////////////////////////////////////////////////////////////////////////////////

void populateTorchMLIRValues(py::module &m) {
  PyAnyTorchListValue::bind(m);
  PyAnyTorchOptionalValue::bind(m);

#define BIND_VALUE(VALUE) PyAnyTorchListOf##VALUE##Value::bind(m);
  FORALL_LIST_BASE_CONCRETE_TYPES(BIND_VALUE)
#undef BIND_VALUE

#define BIND_VALUE(VALUE) PyAnyTorchOptional##VALUE##Value::bind(m);
  FORALL_OPTIONAL_BASE_CONCRETE_TYPES(BIND_VALUE)
  BIND_VALUE(Scalar)
#undef BIND_VALUE

  PyAnyTorchOptionalListOfTorchIntValue::bind(m);

#define BIND_VALUE(VALUE) PyTorch_##VALUE##Value::bind(m);
  FORALL_SCALAR_TYPES(BIND_VALUE)
#undef BIND_VALUE

  PyTorch_DictValue::bind(m);
  PyTorch_TupleValue::bind(m);
  PyTorch_NnModuleValue::bind(m);
  PyAnyTorchScalarValue::bind(m);
}

} // namespace mlir::torch