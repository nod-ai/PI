#include "IRModule.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "TorchTypes.h"
#include "TensorValue.h"

#include <iostream>
#include <utility>

namespace py = pybind11;
using namespace mlir::python;

namespace llvm {
int DisableABIBreakingChecks = 1;
int EnableABIBreakingChecks = 0;
}// namespace llvm



PYBIND11_MODULE(_mlir, m) {
  py::object value_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("Value");
  py::object op_result_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("OpResult");
  py::object type_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("Type");

  py::class_<PyTensor>(m, "_Tensor", value_)
      .def(py::init<>([](const py::capsule& capsule) {
        return PyTensor::createFromCapsule_(capsule);
      }));

  py::class_<PyTorchIntType>(m, "_TorchIntType", type_)
      .def(py::init<>([](py::capsule capsule) {
        return PyTorchIntType::createFromCapsule_(capsule);
      }));
}
