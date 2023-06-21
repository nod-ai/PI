//
// Created by mlevental on 5/15/23.
//

#ifndef PI_TORCHTENSOR_H
#define PI_TORCHTENSOR_H

#include "TorchTypes.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "torch-mlir-c/TorchTypes.h"

#include <algorithm>

#include "TorchTypes.h"
#include "TorchValues.h"

namespace mlir::torch {

// type constraints

bool isAAnyTorchListOfOptionalTensorType(MlirType type);
bool isAAnyTorchListOfTensorType(MlirType type);
bool isAAnyTorchOptionalTensorType(MlirType type);
bool isAAnyTorchTensorType(MlirType type);
bool isATorch_NonValueTensorType(MlirType type);
bool isATorch_ValueTensorType(MlirType type);

// value constraints

bool isATorch_ValueTensorValue(MlirValue value);
bool isATorch_NonValueTensorValue(MlirValue value);

class PyAnyTorchListOfTensorType
    : public PyConcreteType<PyAnyTorchListOfTensorType, PyAnyTorchListType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchListOfTensorType;
  static constexpr const char *pyClassName = "AnyTorchListOfTensorType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchListOfTensorType(MlirType containedType,
                             DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchListTypeGet(containedType)) {}
};

class PyAnyTorchListOfOptionalTensorType
    : public PyConcreteType<PyAnyTorchListOfOptionalTensorType,
                            PyAnyTorchListType> {
public:
  static constexpr IsAFunctionTy isaFunction =
      isAAnyTorchListOfOptionalTensorType;
  static constexpr const char *pyClassName = "AnyTorchListOfOptionalTensorType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchListOfOptionalTensorType(MlirType containedType,
                                     DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchListTypeGet(containedType)) {}
};

class PyAnyTorchOptionalTensorType
    : public PyConcreteType<PyAnyTorchOptionalTensorType,
                            PyAnyTorchOptionalType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalTensorType;
  static constexpr const char *pyClassName = "AnyTorchOptionalTensorType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchOptionalTensorType(MlirType containedType,
                               DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchOptionalTypeGet(containedType)) {}
};

class PyTorch_NonValueTensorType
    : public PyConcreteType<PyTorch_NonValueTensorType> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_NonValueTensorType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchNonValueTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "Torch_NonValueTensorType";
  using PyConcreteType::PyConcreteType;
  PyTorch_NonValueTensorType(std::vector<int64_t> sizes, MlirType dtype,
                             DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchNonValueTensorTypeGet(
                           context->get(), sizes.size(), sizes.data(), dtype)) {
  }
  static PyTorch_NonValueTensorType
  getWithLeastStaticInformation(DefaultingPyMlirContext context);

  static void bindDerived(ClassTy &c);
};

class PyTorch_ValueTensorType : public PyConcreteType<PyTorch_ValueTensorType> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_ValueTensorType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchValueTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "Torch_ValueTensorType";
  using PyConcreteType::PyConcreteType;
  PyTorch_ValueTensorType(std::vector<int64_t> sizes, MlirType dtype,
                          DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchValueTensorTypeGet(
                           context->get(), sizes.size(), sizes.data(), dtype)) {
  }
  static PyTorch_ValueTensorType
  getWithLeastStaticInformation(DefaultingPyMlirContext context);

  static void bindDerived(ClassTy &c);
};

class PyAnyTorchTensorType
    : public PyConcreteType<PyAnyTorchTensorType, PyTorch_NonValueTensorType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchTensorType;
  static constexpr const char *pyClassName = "AnyTorchTensorType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchTensorType(std::vector<int64_t> sizes, MlirType dtype,
                       DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchNonValueTensorTypeGet(
                           context->get(), sizes.size(), sizes.data(), dtype)) {
  }
  static PyAnyTorchTensorType
  getWithLeastStaticInformation(DefaultingPyMlirContext context);

  static void bindDerived(ClassTy &c);
};

bool isAAnyTorchListOfOptionalTensorValue(MlirValue value);
bool isAAnyTorchListOfTensorValue(MlirValue value);
bool isAAnyTorchOptionalTensorValue(MlirValue value);
bool isAAnyTorchTensorValue(MlirValue value);

class PyAnyTorchListOfTensorValue;
PyAnyTorchListOfTensorValue mapListToTorchListOfTensorValue(const py::list &l);

class PyAnyTorchListOfTensorValue
    : public PyConcreteValue<PyAnyTorchListOfTensorValue, PyAnyTorchListValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchListOfTensorValue;
  static constexpr const char *pyClassName = "AnyTorchListOfTensorValue";
  using PyConcreteValue::PyConcreteValue;
  PyAnyTorchListOfTensorValue(const py::list &l)
      : PyAnyTorchListOfTensorValue(mapListToTorchListOfTensorValue(l)){};
  PyAnyTorchListOfTensorValue(const py::tuple &l)
      : PyAnyTorchListOfTensorValue(mapListToTorchListOfTensorValue(l)){};
  static void bindDerived(ClassTy &c);
};

class PyAnyTorchOptionalTensorValue
    : public PyConcreteValue<PyAnyTorchOptionalTensorValue,
                             PyAnyTorchOptionalValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalTensorValue;
  static constexpr const char *pyClassName = "AnyTorchOptionalTensorValue";
  PyAnyTorchOptionalTensorValue(const py::none &n)
      : PyAnyTorchOptionalTensorValue(
            py::cast(PyTorch_NoneValue(n))
                .cast<PyAnyTorchOptionalTensorValue>()) {}
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
  static constexpr const char *pyClassName = "Tensor";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

class PyAnyTorchListOfOptionalTensorValue;
PyAnyTorchListOfOptionalTensorValue
mapListToTorchListOfOptionalTensorValue(const py::list &l);

class PyAnyTorchListOfOptionalTensorValue
    : public PyConcreteValue<PyAnyTorchListOfOptionalTensorValue,
                             PyAnyTorchListValue> {
public:
  static constexpr IsAFunctionTy isaFunction =
      isAAnyTorchListOfOptionalTensorValue;
  static constexpr const char *pyClassName =
      "AnyTorchListOfOptionalTensorValue";
  using PyConcreteValue::PyConcreteValue;
  PyAnyTorchListOfOptionalTensorValue(const py::list &l)
      : PyAnyTorchListOfOptionalTensorValue(
            mapListToTorchListOfOptionalTensorValue(l)){};
  PyAnyTorchListOfOptionalTensorValue(const py::tuple &l)
      : PyAnyTorchListOfOptionalTensorValue(
            mapListToTorchListOfOptionalTensorValue(l)){};
  static void bindDerived(ClassTy &c);
};

void populateTorchTensorOps(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHTENSOR_H
