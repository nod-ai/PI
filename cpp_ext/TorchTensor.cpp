//
// Created by mlevental on 5/15/23.
//

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <exception>
#include <utility>

#include "TorchOps.h"
#include "TorchTensor.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlir::python;

namespace {
#include "TorchTensor.pybinds_tramps.cpp"
}

namespace mlir::torch {

void PyAnyTorchTensorValue::bindDerived(ClassTy &c) {
#include "TorchTensor.pybinds.cpp"
}

void populateTorchTensorOps(py::module &m) { PyAnyTorchTensorValue::bind(m); }
} // namespace mlir::torch
