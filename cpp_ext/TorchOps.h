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
PyAnyTorchScalarValue abs(const PyAnyTorchScalarValue &a, PyLocation *loc,
                          PyInsertionPoint *ip);

// aten::add : (Scalar, Scalar) -> (Scalar)
PyAnyTorchScalarValue add(const PyAnyTorchScalarValue &a,
                          const PyAnyTorchScalarValue &b, PyLocation *loc,
                          PyInsertionPoint *ip);

// aten::ceil.Scalar : (Scalar) -> (Scalar)
PyAnyTorchScalarValue ceil(const PyAnyTorchScalarValue &a, PyLocation *loc,
                           PyInsertionPoint *ip);

// aten::item : (Tensor) -> (Scalar)
PyAnyTorchScalarValue item(const PyAnyTorchTensorValue &self, PyLocation *loc,
                           PyInsertionPoint *ip);

// aten::sub : (Scalar, Scalar) -> (Scalar)
PyAnyTorchScalarValue sub(const PyAnyTorchScalarValue &a,
                          const PyAnyTorchScalarValue &b, PyLocation *loc,
                          PyInsertionPoint *ip);

// aten::view : (Tensor, int[]) -> (Tensor)
// PyAnyTorchTensorValue view(const PyAnyTorchTensorValue &self,
//                           const PyAnyTorchListOfTorchIntValue &size,
//                           PyLocation *loc, PyInsertionPoint *ip);

// aten::chunk : (Tensor, int, int) -> (Tensor[])
PyAnyTorchListOfTensorValue chunk(const PyAnyTorchTensorValue &self,
                                  const PyTorch_IntValue &chunks,
                                  const PyTorch_IntValue &dim, PyLocation *loc,
                                  PyInsertionPoint *ip);

void populateTorchMLIROps(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHOPS_H
