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

class PyAnyTorchTensorValue : public PyConcreteValue<PyAnyTorchTensorValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchTensorValue;
  static constexpr const char *pyClassName = "Tensor";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

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
            (l.empty() || std::all_of(l.begin(), l.end(),
                                      [](auto o) { return o.is_none(); }))
                ? py::cast(
                      PyAnyTorchListValue(
                          py::cast(PyAnyTorchListOfOptionalTensorType(
                              torchMlirTorchNoneTypeGet(
                                  DefaultingPyMlirContext::resolve().get()),
                              DefaultingPyMlirContext::resolve())),
                          l, tag<PyTorch_NoneValue>{}))
                      .cast<PyAnyTorchListOfOptionalTensorValue>()
                : py::cast(PyAnyTorchListValue(
                               py::cast(PyAnyTorchListOfOptionalTensorType(
                                   mlirValueGetType(l[0].cast<PyValue>().get()),
                                   DefaultingPyMlirContext::resolve())),
                               l, tag<PyAnyTorchTensorValue>{}))
                      .cast<PyAnyTorchListOfOptionalTensorValue>()

        ){};
  static void bindDerived(ClassTy &c);
};

void populateTorchTensorOps(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHTENSOR_H
