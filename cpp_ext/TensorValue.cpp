#include "IRModule.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include <iostream>

namespace py = pybind11;
using namespace mlir::python;

namespace llvm {
int DisableABIBreakingChecks = 1;
int EnableABIBreakingChecks = 0;
}// namespace llvm

struct PyTensor : PyValue {
  PyTensor(PyOperationRef operationRef, MlirValue value)
      : PyValue(operationRef, value) {}

  static PyTensor createFromCapsule_(py::capsule capsule) {
    MlirValue value = {capsule.get_pointer()};
    if (mlirValueIsNull(value))
      throw py::error_already_set();
    MlirOperation owner;
    if (mlirValueIsAOpResult(value))
      owner = mlirOpResultGetOwner(value);
    if (mlirValueIsABlockArgument(value))
      owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(value));
    if (mlirOperationIsNull(owner))
      throw py::error_already_set();

    MlirContext ctx = mlirOperationGetContext(owner);
    PyMlirContext *unownedContextWrapper = new PyMlirContext(ctx);
    py::object pyCtxRef = py::reinterpret_steal<py::object>(mlirPythonContextToCapsule(ctx));
    assert(pyCtxRef && "cast to py::object failed");
    auto ctxRef = PyMlirContextRef(unownedContextWrapper, std::move(pyCtxRef));

    py::object pyOpRef = py::reinterpret_steal<py::object>(mlirPythonOperationToCapsule(owner));
    PyOperation *unownedOperation =
        new PyOperation(std::move(ctxRef), owner);
    unownedOperation->handle = pyOpRef;
    auto ownerRef = PyOperationRef(unownedOperation, std::move(pyOpRef));

    return PyTensor(ownerRef, value);
  }
};

PYBIND11_MODULE(_tensor, m) {
  py::object value_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("Value");
  py::object op_result_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("OpResult");

  py::class_<PyTensor>(m, "_Tensor", value_)
      .def(py::init<>([](py::capsule capsule) {
        return PyTensor::createFromCapsule_(std::move(capsule));
      }));
}
