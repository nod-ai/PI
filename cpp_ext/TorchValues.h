//
// Created by maksim on 5/13/23.
//

#ifndef PI_TORCHVALUES_H
#define PI_TORCHVALUES_H

#include "IRModule.h"
#include "TorchTypes.h"

namespace py = pybind11;
using namespace mlir::python;

namespace mlir::torch {

bool isAAnyTorchDictKeyValue(MlirValue value);

bool isAAnyTorchListOfOptionalIntValue(MlirValue value);
bool isAAnyTorchListOfOptionalTensorValue(MlirValue value);

bool isAAnyTorchListOfTensorValue(MlirValue value);
bool isAAnyTorchListOfTorchBoolValue(MlirValue value);
bool isAAnyTorchListOfTorchIntValue(MlirValue value);
bool isAAnyTorchListOfTorchStringValue(MlirValue value);
bool isAAnyTorchListValue(MlirValue value);

bool isAAnyTorchOptionalBoolValue(MlirValue value);
bool isAAnyTorchOptionalDeviceValue(MlirValue value);
bool isAAnyTorchOptionalFloatValue(MlirValue value);
bool isAAnyTorchOptionalGeneratorValue(MlirValue value);
bool isAAnyTorchOptionalIntValue(MlirValue value);
bool isAAnyTorchOptionalStringValue(MlirValue value);
bool isAAnyTorchOptionalTensorValue(MlirValue value);
bool isAAnyTorchOptionalValue(MlirValue value);

bool isAAnyTorchOptionalListOfTorchIntValue(MlirValue value);

bool isAAnyTorchOptionalScalarValue(MlirValue value);
bool isAAnyTorchScalarValue(MlirValue value);
bool isAAnyTorchTensorValue(MlirValue value);
bool isAAnyTorchValue(MlirValue value);

bool isATorch_BoolValue(MlirValue value);
bool isATorch_DeviceValue(MlirValue value);
bool isATorch_DictValue(MlirValue value);
bool isATorch_FloatValue(MlirValue value);
bool isATorch_IntValue(MlirValue value);
bool isATorch_LinearParamsValue(MlirValue value);
bool isATorch_NnModuleValue(MlirValue value);
bool isATorch_NonValueTensorValue(MlirValue value);
bool isATorch_NoneValue(MlirValue value);
bool isATorch_NumberValue(MlirValue value);
bool isATorch_StringValue(MlirValue value);
bool isATorch_TupleValue(MlirValue value);
bool isATorch_ValueTensorValue(MlirValue value);

template <typename DerivedTy> class PyConcreteValue : public PyValue {
public:
  using ClassTy = py::class_<DerivedTy, PyValue>;
  using IsAFunctionTy = bool (*)(MlirValue);

  PyConcreteValue() = default;
  PyConcreteValue(PyOperationRef operationRef, MlirValue value)
      : PyValue(operationRef, value) {}
  PyConcreteValue(PyValue &orig)
      : PyConcreteValue(orig.getParentOperation(), castFrom(orig)) {}

  /// Attempts to cast the original value to the derived type and throws on
  /// type mismatches.
  static MlirValue castFrom(PyValue &orig);

  /// Binds the Python module objects to functions of this class.
  static void bind(py::module &m);

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class PyAnyTorchListValue : public PyConcreteValue<PyAnyTorchListValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchListValue;
  static constexpr const char *pyClassName = "AnyTorchListValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyAnyTorchOptionalValue
    : public PyConcreteValue<PyAnyTorchOptionalValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalValue;
  static constexpr const char *pyClassName = "AnyTorchOptionalValue";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

#define DECLARE_LIST_BASE_CONCRETE_VALUE(CONCRETEVALUE)                        \
  class PyAnyTorchListOf##CONCRETEVALUE##Value                                 \
      : public PyConcreteValue<PyAnyTorchListOf##CONCRETEVALUE##Value> {       \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction =                               \
        isAAnyTorchListOf##CONCRETEVALUE##Value;                               \
    static constexpr const char *pyClassName =                                 \
        "AnyTorchListOf" #CONCRETEVALUE "Value";                               \
    using PyConcreteValue::PyConcreteValue;                                    \
    static void bindDerived(ClassTy &c);                                       \
  };
FORALL_LIST_BASE_CONCRETE_TYPES(DECLARE_LIST_BASE_CONCRETE_VALUE)
#undef DECLARE_LIST_BASE_CONCRETE_VALUE

#define DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(CONCRETEVALUE)                    \
  class PyAnyTorchOptional##CONCRETEVALUE##Value                               \
      : public PyConcreteValue<PyAnyTorchOptional##CONCRETEVALUE##Value> {     \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction =                               \
        isAAnyTorchOptional##CONCRETEVALUE##Value;                             \
    static constexpr const char *pyClassName =                                 \
        "AnyTorchOptional" #CONCRETEVALUE "Value";                             \
    using PyConcreteValue::PyConcreteValue;                                    \
    static void bindDerived(ClassTy &c);                                       \
  };
FORALL_OPTIONAL_BASE_CONCRETE_TYPES(DECLARE_OPTIONAL_BASE_CONCRETE_VALUE)
#undef DECLARE_OPTIONAL_BASE_CONCRETE_VALUE

#define DECLARE_SCALAR_VALUE(SCALARVALUE)                                      \
  class PyTorch_##SCALARVALUE##Value                                           \
      : public PyConcreteValue<PyTorch_##SCALARVALUE##Value> {                 \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction =                               \
        isATorch_##SCALARVALUE##Value;                                         \
    static constexpr const char *pyClassName = "Torch_" #SCALARVALUE "Value";  \
    using PyConcreteValue::PyConcreteValue;                                    \
    static void bindDerived(ClassTy &c);                                       \
  };
FORALL_SCALAR_TYPES(DECLARE_SCALAR_VALUE)
#undef DECLARE_SCALAR_VALUE

class PyTorch_DictValue : public PyConcreteValue<PyTorch_DictValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_DictValue;
  static constexpr const char *pyClassName = "Torch_DictValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyTorch_TupleValue : public PyConcreteValue<PyTorch_TupleValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_TupleValue;
  static constexpr const char *pyClassName = "Torch_TupleValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyTorch_NnModuleValue : public PyConcreteValue<PyTorch_NnModuleValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_NnModuleValue;
  static constexpr const char *pyClassName = "Torch_NnModuleValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyTorch_NonValueTensorValue
    : public PyConcreteValue<PyTorch_NonValueTensorValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_NonValueTensorValue;
  static constexpr const char *pyClassName = "Torch_NonValueTensorValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyTorch_ValueTensorValue
    : public PyConcreteValue<PyTorch_ValueTensorValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_ValueTensorValue;
  static constexpr const char *pyClassName = "Torch_ValueTensorValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyAnyTorchTensorValue : public PyConcreteValue<PyAnyTorchTensorValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchTensorValue;
  static constexpr const char *pyClassName = "AnyTorchTensorValue";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

void populateTorchMLIRValues(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHVALUES_H
