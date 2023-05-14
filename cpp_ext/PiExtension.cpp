#include "mlir-c/Bindings/Python/Interop.h"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <vector>

#include "TorchTypes.h"
#include "TorchValues.h"

namespace py = pybind11;
using namespace mlir::python;
using namespace mlir::torch;

auto torch =
    py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");

py::object add(const PyTorch_IntValue &a, const PyTorch_IntValue &b) {
  return torch.attr("AtenAddIntOp")(a, b);
}

py::object add(const PyTorch_FloatValue &a, const PyTorch_IntValue &b) {
  return torch.attr("AtenAddFloatIntOp")(a, b);
}

PYBIND11_MODULE(_pi_mlir, m) {
  populateTorchMLIRTypes(m);
  populateTorchMLIRValues(m);

  auto nnModule = m.def_submodule("nn");
  nnModule.def(
      "add",
      py::overload_cast<const PyTorch_IntValue &, const PyTorch_IntValue &>(
          &add));
  nnModule.def(
      "add",
      py::overload_cast<const PyTorch_FloatValue &, const PyTorch_IntValue &>(
          &add));
}
