//
// Created by maksim on 5/14/23.
//

#include "TorchOps.h"
#include "TorchTensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "Globals.h"

namespace py = pybind11;
using namespace mlir::python;

namespace mlir::torch {

#include "TorchOps.impls.cpp"

void populateTorchMLIROps(py::module &m) {
#include "TorchOps.pybinds.cpp"
}

} // namespace mlir::torch
