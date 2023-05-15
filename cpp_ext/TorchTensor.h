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

#include "TorchValues.h"

namespace mlir::torch {

class PyAnyTorchTensorValue : public PyConcreteValue<PyAnyTorchTensorValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchTensorValue;
  static constexpr const char *pyClassName = "Tensor";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

void populateTorchTensorOps(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHTENSOR_H
