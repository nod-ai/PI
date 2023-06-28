//
// Created by maksim on 5/14/23.
//

#include "Globals.h"
#include "IRModule.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>

#include "TorchOps.h"
#include "TorchTensor.h"
#include "TorchTypes.h"
#include "TorchValues.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlir::python;

using llvm::StringRef;
using llvm::Twine;

namespace mlir::torch {

#include "TorchOps.impls.cpp"

// prim::abs.Scalar : (Scalar) -> (Scalar)
PyAnyTorchScalarValue abs(const PyAnyTorchScalarValue &a, PyLocation *loc,
                          PyInsertionPoint *ip) {
  auto resultType = py::cast(mlirValueGetType(a)).cast<PyType>();
  PyOperationRef opRef =
      createOperation("torch.prim.abs.Scalar", {resultType}, {a},
                      /*attributes=*/{}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

// aten::add : (Scalar, Scalar) -> (Scalar)
PyAnyTorchScalarValue add(const PyAnyTorchScalarValue &a,
                          const PyAnyTorchScalarValue &b, PyLocation *loc,
                          PyInsertionPoint *ip) {
  auto resultType = py::cast(mlirValueGetType(a)).cast<PyType>();
  PyOperationRef opRef = createOperation("torch.aten.add", {resultType}, {a, b},
                                         /*attributes=*/{}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

// aten::ceil.Scalar : (Scalar) -> (Scalar)
PyAnyTorchScalarValue ceil(const PyAnyTorchScalarValue &a, PyLocation *loc,
                           PyInsertionPoint *ip) {
  auto resultType = py::cast(mlirValueGetType(a)).cast<PyType>();
  PyOperationRef opRef =
      createOperation("torch.aten.ceil.Scalar", {resultType}, {a},
                      /*attributes=*/{}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

// aten::sub : (Scalar, Scalar) -> (Scalar)
PyAnyTorchScalarValue sub(const PyAnyTorchScalarValue &a,
                          const PyAnyTorchScalarValue &b, PyLocation *loc,
                          PyInsertionPoint *ip) {
  auto resultType = py::cast(mlirValueGetType(a)).cast<PyType>();
  PyOperationRef opRef = createOperation("torch.aten.sub", {resultType}, {a, b},
                                         /*attributes=*/{}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

// prim::device : (str) -> (Device)
PyTorch_DeviceValue device(const std::string &type, PyLocation *loc,
                           PyInsertionPoint *ip) {
  return makePyTorchDeviceValue(type, loc, ip);
}

// aten::mean.dim : (Tensor, int?, bool, int?) -> (Tensor)
PyAnyTorchTensorValue mean(const PyAnyTorchTensorValue &self,
                           const PyAnyTorchOptionalIntValue &dim,
                           const PyTorch_BoolValue &keepdim,
                           const PyAnyTorchOptionalIntValue &dtype,
                           PyLocation *loc, PyInsertionPoint *ip) {
  auto resultType = PyAnyTorchTensorType::getWithLeastStaticInformation(
      loc->getContext().get());
  PyOperationRef opRef = createOperation("torch.aten.mean.dim", {resultType},
                                         {
                                             self,
                                             dim,
                                             keepdim,
                                             dtype,
                                         },
                                         /*attributes=*/{}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

// aten::chunk : (Tensor, int, int) -> (Tensor[])
PyAnyTorchListOfTensorValue chunk(const PyAnyTorchTensorValue &self,
                                  const PyTorch_IntValue &chunks,
                                  const PyTorch_IntValue &dim, PyLocation *loc,
                                  PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.chunk";
  std::vector<PyType> _returnTypes = {PyAnyTorchListOfTensorType(
      PyAnyTorchTensorType::getWithLeastStaticInformation(
          loc->getContext().get()),
      loc->getContext().get())};
  std::vector<std::reference_wrapper<const PyType>> returnTypes;
  for (const auto &returnType : _returnTypes)
    returnTypes.push_back(returnType);
  PyOperationRef opRef =
      createOperation(operationName, returnTypes, {self, chunks, dim},
                      /*attributes=*/{}, loc, ip);
  MlirOperation operation = opRef->get();
  auto res = mlirOperationGetResult(operation, 0);
  auto list = py::cast(res).cast<PyAnyTorchListOfTensorValue>();
  auto owner = getOwner(chunks);
  if (unwrap(mlirIdentifierStr(mlirOperationGetName(owner)))
          .starts_with("torch.constant"))
    list.length = getAttributeValue(chunks);
  return list;
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

  // prim::abs.Scalar : (Scalar) -> (Scalar)
  m.def(
      "abs",
      [](const PyAnyTorchScalarValue &x, const DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) {
        return abs(x, loc.get(), ip.get());
      },
      "x"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

  // aten::add : (Scalar, Scalar) -> (Scalar)
  m.def(
      "add",
      [](const PyAnyTorchScalarValue &lhs, const PyAnyTorchScalarValue &rhs,
         DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) {
        return add(lhs, rhs, loc.get(), ip.get());
      },
      "lhs"_a, "rhs"_a, py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());

  // aten::ceil.Scalar : (Scalar) -> (Scalar)
  m.def(
      "ceil",
      [](const PyAnyTorchScalarValue &a, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) {
        return ceil(a, loc.get(), ip.get());
      },
      "a"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

  // aten::sub : (Scalar, Scalar) -> (Scalar)
  m.def(
      "sub",
      [](const PyAnyTorchScalarValue &lhs, const PyAnyTorchScalarValue &rhs,
         DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) {
        return sub(lhs, rhs, loc.get(), ip.get());
      },
      "lhs"_a, "rhs"_a, py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());
  m.def("avg_pool1d",
        [](PyAnyTorchTensorValue &self,
           PyAnyTorchListOfTorchIntType &kernel_size,
           PyAnyTorchListOfTorchIntType &stride,
           PyAnyTorchListOfTorchIntType &padding, PyTorch_BoolValue &ceil_mode,
           PyAnyTorchListOfTorchIntType &count_include_pad) {
          throw NotImplementedError("aten::avg_pool1d : (Tensor, int[], int[], "
                                    "int[], bool, bool) -> (Tensor)");
        });

  // aten::view : (Tensor, int[]) -> (Tensor)
  m.def(
      "view",
      [](const PyAnyTorchTensorValue &self,
         const PyAnyTorchListOfTorchIntValue &size, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        return view(self, size, loc.get(), ip.get());
      },
      "self"_a, "size"_a, py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());

  // prim::device : (str) -> (Device)
  m.def(
      "device",
      [](const std::string &type, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyTorch_DeviceValue {
        return device(type, loc.get(), ip.get());
      },
      "type"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

  // aten::mean.dim : (Tensor, int?, bool, int?) -> (Tensor)
  m.def(
      "mean",
      [](const PyAnyTorchTensorValue &self,
         const PyAnyTorchOptionalIntValue &dim,
         const PyTorch_BoolValue &keepdim,
         const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        auto dims = PyAnyTorchOptionalListOfTorchIntValue(py::make_tuple(dim));
        return mean(self, dims, keepdim, dtype, loc.get(), ip.get());
      },
      "self"_a, "dim"_a = py::none(), "keepdim"_a = false,
      "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());

  m.def(
      "vector_norm",
      [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &ord,
         const PyAnyTorchOptionalListOfTorchIntValue &dim,
         const PyTorch_BoolValue &keepdim,
         const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        return linalg_vector_norm(self, ord, dim, keepdim, dtype, loc.get(),
                                  ip.get());
      },
      "self"_a, "ord"_a = 2, "dim"_a = py::none(), "keepdim"_a = false,
      "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());

  // aten::chunk : (Tensor, int, int) -> (Tensor[])
  m.def(
      "chunk",
      [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &chunks,
         const PyTorch_IntValue &dim, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListOfTensorValue {
        return chunk(self, chunks, dim, loc.get(), ip.get());
      },
      "self"_a, "chunks"_a, "dim"_a = 0, py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());
}

} // namespace mlir::torch
