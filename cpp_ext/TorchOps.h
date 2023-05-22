//
// Created by maksim on 5/14/23.
//

#ifndef PI_TORCHOPS_H
#define PI_TORCHOPS_H

// hack
#include "IRModule.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "torch-mlir-c/TorchTypes.h"

#include "TorchTensor.h"
#include "TorchTypes.h"
#include "TorchValues.h"

namespace py = pybind11;
using namespace mlir::python;

struct NotImplementedError : public std::exception {
  NotImplementedError(std::string msg) : message(std::move(msg)) {}
  [[nodiscard]] const char *what() const noexcept override {
    return message.data();
  }

  std::string message;
};

namespace mlir::torch {

#include "TorchOps.inc.h"

// aten::ScalarImplicit : (Tensor) -> (Scalar)
// py::object ScalarImplicit(const PyAnyTorchTensorValue &a);

// prim::abs.Scalar : (Scalar) -> (Scalar)
py::object abs(const PyAnyTorchScalarValue &a);

// aten::add : (Scalar, Scalar) -> (Scalar)
py::object add(const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b);

// aten::ceil.Scalar : (Scalar) -> (Scalar)
py::object ceil(const PyAnyTorchScalarValue &a);

// aten::item : (Tensor) -> (Scalar)
py::object item(const PyAnyTorchTensorValue &self);

// aten::sub : (Scalar, Scalar) -> (Scalar)
// py::object sub(const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue
// &b);

void populateTorchMLIROps(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHOPS_H
