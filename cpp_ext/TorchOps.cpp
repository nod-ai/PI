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

py::object add(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  auto torch =
      py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAddIntOp")(a, b);
}

py::object add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  auto torch =
      py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
  return torch.attr("AtenAddFloatIntOp")(a, b);
}

void populateTorchMLIROps(py::module &m) {
  m.def("add",
        py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(
            &add));
  m.def("add",
        py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(
            &add));
}

} // namespace mlir::torch
