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

namespace mlir::torch {

#include "TorchOps.inc.h"

void populateTorchMLIROps(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHOPS_H
