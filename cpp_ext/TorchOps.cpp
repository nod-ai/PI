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

// aten::softplus : (Tensor, Scalar, Scalar) -> (Tensor)
PyAnyTorchTensorValue softplus(const PyAnyTorchTensorValue &self,
                               const PyAnyTorchScalarValue &beta,
                               const PyAnyTorchScalarValue &threshold__,
                               PyLocation *loc, PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.softplus";
  std::vector<PyType> _returnTypes = {
      PyAnyTorchTensorType::getWithLeastStaticInformation(
          loc->getContext().get())};
  std::vector<std::reference_wrapper<const PyType>> returnTypes;
  for (const auto &returnType : _returnTypes)
    returnTypes.push_back(returnType);
  PyOperationRef opRef =
      createOperation(operationName, returnTypes, {self, beta, threshold__},
                      /*attributes=*/{}, loc, ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}

template <typename T>
const PyAnyTorchListOfTorchIntValue castTypeToListInt(const T arg) {
  if constexpr (std::is_same_v<T, PyAnyTorchListOfTorchIntValue>) {
    return arg;
  } else {
    return PyAnyTorchListOfTorchIntValue(py::make_tuple(arg, arg));
  }
}

// aten::max_pool2d_with_indices : (Tensor, int[], int[], int[], int[], bool) ->
// (Tensor, Tensor)
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>
max_pool2d_with_indices_(const PyAnyTorchTensorValue &self,
                         const PyAnyTorchListOfTorchIntValue &kernel_size,
                         const PyAnyTorchListOfTorchIntValue &stride,
                         const PyAnyTorchListOfTorchIntValue &padding,
                         const PyAnyTorchListOfTorchIntValue &dilation,
                         const PyTorch_BoolValue &ceil_mode, PyLocation *loc,
                         PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.max_pool2d_with_indices";
  std::vector<PyType> _returnTypes = {
      PyAnyTorchTensorType::getWithLeastStaticInformation(
          loc->getContext().get()),
      PyAnyTorchTensorType::getWithLeastStaticInformation(
          loc->getContext().get())};
  std::vector<std::reference_wrapper<const PyType>> returnTypes;
  for (const auto &returnType : _returnTypes)
    returnTypes.push_back(returnType);
  PyOperationRef opRef =
      createOperation(operationName, returnTypes,
                      {self, kernel_size, stride, padding, dilation, ceil_mode},
                      /*attributes=*/{}, loc, ip);
  MlirOperation operation = opRef->get();
  return std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>(
      {opRef, mlirOperationGetResult(operation, 0)},
      {opRef, mlirOperationGetResult(operation, 1)});
}

//  aten::max_pool2d_with_indices : (Tensor, Union[int[], int], Union[int[],
//  int], Union[int[], int], Union[int[], int], bool) -> (Tensor, Tensor)
template <typename T1, typename T2, typename T3, typename T4>
std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue>
max_pool2d_with_indices(PyAnyTorchTensorValue &self, T1 &kernel_size,
                        T2 &stride, T3 &padding, T4 &dilation,
                        PyTorch_BoolValue &ceil_mode, PyLocation *loc,
                        PyInsertionPoint *ip) {

  PyAnyTorchListOfTorchIntValue kernel_size_ = castTypeToListInt(kernel_size);
  PyAnyTorchListOfTorchIntValue stride_ = castTypeToListInt(stride);
  PyAnyTorchListOfTorchIntValue padding_ = castTypeToListInt(padding);
  PyAnyTorchListOfTorchIntValue dilation_ = castTypeToListInt(dilation);
  PyLocation *loc_ = &DefaultingPyLocation::resolve();
  PyInsertionPoint *ip_ = &DefaultingPyInsertionPoint::resolve();

  return max_pool2d_with_indices_(self, kernel_size_, stride_, padding_,
                                  dilation_, ceil_mode, loc_, ip_);
}

struct bind_max_pool2d_with_indices {
  template <typename T1, typename T2 = PyAnyTorchListOfTorchIntValue,
            typename T3 = PyAnyTorchListOfTorchIntValue,
            typename T4 = PyAnyTorchListOfTorchIntValue>
  static void bind(py::module &m) {
    m.def(
        "max_pool2d_with_indices",
        [](PyAnyTorchTensorValue &self, T1 &kernel_size, T2 &stride,
           T3 &padding, T4 &dilation, PyTorch_BoolValue &ceil_mode,
           PyLocation *loc, PyInsertionPoint *ip)
            -> std::tuple<PyAnyTorchTensorValue, PyAnyTorchTensorValue> {
          return max_pool2d_with_indices(self, kernel_size, stride, padding,
                                         dilation, ceil_mode, loc, ip);
        },
        "self"_a, "kernel_size"_a, "stride"_a = std::vector<int>{},
        "padding"_a = std::vector<int>{0, 0},
        "dilation"_a = std::vector<int>{1, 1}, "ceil_mode"_a = false,
        py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());
  }
};

// aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)
PyAnyTorchTensorValue
max_pool2d_(const PyAnyTorchTensorValue &self,
            const PyAnyTorchListOfTorchIntValue &kernel_size,
            const PyAnyTorchListOfTorchIntValue &stride,
            const PyAnyTorchListOfTorchIntValue &padding,
            const PyAnyTorchListOfTorchIntValue &dilation,
            const PyTorch_BoolValue &ceil_mode, PyLocation *loc,
            PyInsertionPoint *ip) {
  std::string operationName = "torch.aten.max_pool2d";
  std::vector<PyType> _returnTypes = {
      PyAnyTorchTensorType::getWithLeastStaticInformation(
          loc->getContext().get())};
  std::vector<std::reference_wrapper<const PyType>> returnTypes;
  for (const auto &returnType : _returnTypes)
    returnTypes.push_back(returnType);
  PyOperationRef opRef =
      createOperation(operationName, returnTypes,
                      {self, kernel_size, stride, padding, dilation, ceil_mode},
                      /*attributes=*/{}, loc, ip);
  MlirOperation operation = opRef->get();
  return {opRef, mlirOperationGetResult(operation, 0)};
}

// aten::max_pool2d : (Tensor, Union[int[], int], Union[int[], int],
// Union[int[], int], Union[int[], int], bool) -> (Tensor)
template <typename T1, typename T2, typename T3, typename T4>
PyAnyTorchTensorValue max_pool2d(PyAnyTorchTensorValue &self, T1 &kernel_size,
                                 T2 &stride, T3 &padding, T4 &dilation,
                                 PyTorch_BoolValue &ceil_mode, PyLocation *loc,
                                 PyInsertionPoint *ip) {

  PyAnyTorchListOfTorchIntValue kernel_size_ = castTypeToListInt(kernel_size);
  PyAnyTorchListOfTorchIntValue stride_ = castTypeToListInt(stride);
  PyAnyTorchListOfTorchIntValue padding_ = castTypeToListInt(padding);
  PyAnyTorchListOfTorchIntValue dilation_ = castTypeToListInt(dilation);
  PyLocation *loc_ = &DefaultingPyLocation::resolve();
  PyInsertionPoint *ip_ = &DefaultingPyInsertionPoint::resolve();

  return max_pool2d_(self, kernel_size_, stride_, padding_, dilation_,
                     ceil_mode, loc_, ip_);
}

struct bind_max_pool2d {
  template <typename T1, typename T2 = PyAnyTorchListOfTorchIntValue,
            typename T3 = PyAnyTorchListOfTorchIntValue,
            typename T4 = PyAnyTorchListOfTorchIntValue>
  static void bind(py::module &m) {
    m.def(
        "max_pool2d",
        [](PyAnyTorchTensorValue &self, T1 &kernel_size, T2 &stride,
           T3 &padding, T4 &dilation, PyTorch_BoolValue &ceil_mode,
           PyLocation *loc, PyInsertionPoint *ip) -> PyAnyTorchTensorValue {
          return max_pool2d(self, kernel_size, stride, padding, dilation,
                            ceil_mode, loc.get(), ip.get());
        },
        "self"_a, "kernel_size"_a, "stride"_a = std::vector<int>{},
        "padding"_a = std::vector<int>{0, 0},
        "dilation"_a = std::vector<int>{1, 1}, "ceil_mode"_a = false,
        py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());
  }
};

// Recursive function to generate bindings for all combinations of
// Torch_IntValue and AnyTorchListOfTorchIntValue in N slots
template <unsigned int N, class Callback, typename... Args>
struct generateListIntCompatibleBindings {
  static void generate(py::module &m) {
    generateListIntCompatibleBindings<N - 1, Callback, Args...,
                                      PyTorch_IntValue>::generate(m);
    generateListIntCompatibleBindings<
        N - 1, Callback, Args..., PyAnyTorchListOfTorchIntValue>::generate(m);
  }
};

template <class Callback, typename... Args>
struct generateListIntCompatibleBindings<0, Callback, Args...> {
  static void generate(py::module &m) { Callback::template bind<Args...>(m); }
};

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

  // aten::linalg_vector_norm : (Tensor, Scalar, int[]?, bool, int?) ->
  // (Tensor)
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

  // aten::linalg_vector_norm : (Tensor, Scalar, int, bool, int?) -> (Tensor)
  m.def(
      "vector_norm",
      [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &ord,
         const PyTorch_IntValue &dim, const PyTorch_BoolValue &keepdim,
         const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        auto dims = PyAnyTorchOptionalListOfTorchIntValue(py::make_tuple(dim));

        return linalg_vector_norm(self, ord, dims, keepdim, dtype, loc.get(),
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

  // aten::amax : (Tensor, int, bool) -> (Tensor)
  m.def(
      "amax",
      [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim,
         const PyTorch_BoolValue &keepdim, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        auto dims = PyAnyTorchListOfTorchIntValue(py::make_tuple(dim));
        return amax(self, dims, keepdim, loc.get(), ip.get());
      },
      "self"_a, "dim"_a, "keepdim"_a = false, py::kw_only(),
      "loc"_a = py::none(), "ip"_a = py::none());

  // aten::sum.dim_IntList : (Tensor, int[]?, bool, int?) -> (Tensor)
  m.def(
      "sum",
      [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dim,
         const PyTorch_BoolValue &keepdim,
         const PyAnyTorchOptionalIntValue &dtype, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        auto dims = PyAnyTorchListOfTorchIntValue(py::make_tuple(dim));
        return sum(self, dims, keepdim, dtype, loc.get(), ip.get());
      },
      "self"_a, "dim"_a = py::none(), "keepdim"_a = false,
      "dtype"_a = py::none(), py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());

  // aten::softplus : (Tensor, Scalar, Scalar) -> (Tensor)
  m.def(
      "softplus",
      [](const PyAnyTorchTensorValue &self, const PyAnyTorchScalarValue &beta,
         const PyAnyTorchScalarValue &threshold__, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        return softplus(self, beta, threshold__, loc.get(), ip.get());
      },
      "self"_a, "beta"_a = 1, "threshold__"_a = 20, py::kw_only(),
      "loc"_a = py::none(), "ip"_a = py::none());

  generateListIntCompatibleBindings<4, bind_max_pool2d>::generate(m);
  generateListIntCompatibleBindings<4, bind_max_pool2d_with_indices>::generate(
      m);
}

}; // namespace mlir::torch