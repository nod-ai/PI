//
// Created by maksim on 5/14/23.
//

#include "TorchOps.h"
#include "IRModule.h"
#include "TorchTensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>

#include "Globals.h"
#include "TorchTypes.h"
#include "mlir-c/IR.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"

namespace py = pybind11;
using namespace mlir::python;

namespace mlir::torch {

#include "TorchOps.impls.cpp"

// aten::ScalarImplicit : (Tensor) -> (Scalar)
py::object ScalarImplicit(const PyAnyTorchTensorValue &a) {
  throw NotImplementedError("ScalarImplicit with signature "
                            "aten::ScalarImplicit : (Tensor) -> (Scalar)");
  //  return PyGlobals::get()
  //      .lookupOperationClass("torch.aten.ScalarImplicit")
  //      .value()(a);
}

// prim::abs.Scalar : (Scalar) -> (Scalar)
py::object abs(const PyAnyTorchScalarValue &a) {
  auto mlirType = mlirValueGetType(a);
  auto returnType =
      PyType(PyMlirContext::forContext(mlirTypeGetContext(mlirType)), mlirType);
  return PyGlobals::get()
      .lookupOperationClass("torch.prim.abs.Scalar")
      .value()(returnType, a);
}

// aten::add : (Scalar, Scalar) -> (Scalar)
py::object add(const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b) {
  auto mlirTypeA = mlirValueGetType(a);
  auto mlirTypeB = mlirValueGetType(b);
  if (!mlirTypeEqual(mlirTypeA, mlirTypeB))
    throw NotImplementedError(
        "Arithmetic ops on Scalar values with like types supported; type a: " +
        py::str(py::cast(a)).cast<std::string>() +
        ", type b: " + py::str(py::cast(b)).cast<std::string>());
  auto returnType = PyType(
      PyMlirContext::forContext(mlirTypeGetContext(mlirTypeA)), mlirTypeA);
  return PyGlobals::get()
      .lookupOperationClass("torch.aten.add")
      .value()(returnType, a, b);
}

// aten::ceil.Scalar : (Scalar) -> (Scalar)
py::object ceil(const PyAnyTorchScalarValue &a) {
  auto mlirType = mlirValueGetType(a);
  auto returnType =
      PyType(PyMlirContext::forContext(mlirTypeGetContext(mlirType)), mlirType);
  return PyGlobals::get()
      .lookupOperationClass("torch.aten.ceil.Scalar")
      .value()(returnType, a);
}

// aten::item : (Tensor) -> (Scalar)
py::object item(const PyAnyTorchTensorValue &self) {
  throw NotImplementedError("ScalarImplicit with signature "
                            "aten::ScalarImplicit : (Tensor) -> (Scalar)");
  //  return
  //  PyGlobals::get().lookupOperationClass("torch.aten.item").value()(self);
}

// aten::sub : (Scalar, Scalar) -> (Scalar)
py::object sub(const PyAnyTorchScalarValue &a, const PyAnyTorchScalarValue &b) {
  auto mlirTypeA = mlirValueGetType(a);
  auto mlirTypeB = mlirValueGetType(b);
  if (!mlirTypeEqual(mlirTypeA, mlirTypeB))
    throw NotImplementedError(
        "Arithmetic ops on Scalar values with like types supported; type a: " +
        py::str(py::cast(a)).cast<std::string>() +
        ", type b: " + py::str(py::cast(b)).cast<std::string>());
  auto returnType = PyType(
      PyMlirContext::forContext(mlirTypeGetContext(mlirTypeA)), mlirTypeA);
  return PyGlobals::get()
      .lookupOperationClass("torch.aten.sub")
      .value()(returnType, a, b);
}

void populateTorchMLIROps(py::module &m) {

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const NotImplementedError &e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    }
  });

#include "TorchOps.pybinds.cpp"

  // aten::ScalarImplicit : (Tensor) -> (Scalar)
  //  m.def("ScalarImplicit",
  //        py::overload_cast<const PyAnyTorchTensorValue &>(&ScalarImplicit));

  // prim::abs.Scalar : (Scalar) -> (Scalar)
  m.def("abs", py::overload_cast<const PyAnyTorchScalarValue &>(&abs));

  // aten::add : (Scalar, Scalar) -> (Scalar)
  m.def("add", py::overload_cast<const PyAnyTorchScalarValue &,
                                 const PyAnyTorchScalarValue &>(&add));

  // aten::ceil.Scalar : (Scalar) -> (Scalar)
  m.def("ceil", py::overload_cast<const PyAnyTorchScalarValue &>(&ceil));

  // aten::item : (Tensor) -> (Scalar)
  //  m.def("item", py::overload_cast<const PyAnyTorchTensorValue &>(&item));

  // aten::sub : (Scalar, Scalar) -> (Scalar)
  m.def("sub", py::overload_cast<const PyAnyTorchScalarValue &,
                                 const PyAnyTorchScalarValue &>(&sub));
}

} // namespace mlir::torch
