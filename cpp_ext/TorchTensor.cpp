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
using namespace mlir::torch;
#include "TorchTensor.pybinds_tramps.cpp"
} // namespace

namespace mlir::torch {

void PyAnyTorchTensorValue::bindDerived(ClassTy &c) {
#include "TorchTensor.pybinds.cpp"
}

void PyAnyTorchListOfOptionalTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::list>(), py::arg("value"));
  py::implicitly_convertible<py::list, PyAnyTorchListOfOptionalTensorValue>();
  py::implicitly_convertible<py::tuple, PyAnyTorchListOfOptionalTensorValue>();
}

void populateTorchTensorOps(py::module &m) {
  PyAnyTorchTensorValue::bind(m);
  PyAnyTorchListOfOptionalTensorValue::bind(m);
}
} // namespace mlir::torch
