//
// Created by maksim on 5/14/23.
//

#include "TorchOps.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <vector>

#include "TorchTypes.h"
#include "TorchValues.h"

namespace py = pybind11;
using namespace mlir::python;

namespace mlir::torch {

#include "TorchOps.impls.cpp"

void populateTorchMLIROps(py::module &m) {
#include "TorchOps.pybinds.cpp"
}

} // namespace mlir::torch
