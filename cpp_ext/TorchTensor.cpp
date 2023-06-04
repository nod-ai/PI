//
// Created by mlevental on 5/15/23.
//

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <exception>
#include <utility>

#include "TorchOps.h"
#include "TorchTensor.h"
#include "TorchValues.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlir::python;

namespace {
using namespace mlir::torch;
#include "TorchTensor.pybinds_tramps.cpp"
} // namespace

namespace mlir::torch {

// type constraints

bool isAAnyTorchListOfOptionalTensorType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           (((torchMlirTypeIsATorchBaseTensor(
                torchMlirTorchListTypeGetContainedType(type)))) ||
            ((torchMlirTypeIsATorchOptional(
                torchMlirTorchListTypeGetContainedType(type)))) ||
            ((torchMlirTypeIsATorchNone(
                torchMlirTorchListTypeGetContainedType(type)))))));
}

bool isAAnyTorchListOfTensorType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           ((torchMlirTypeIsATorchBaseTensor(
               torchMlirTorchListTypeGetContainedType(type))))));
}

bool isAAnyTorchOptionalTensorType(MlirType type) {
  return ((((torchMlirTypeIsATorchBaseTensor(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchTensorType(MlirType type) {
  return (((torchMlirTypeIsATorchBaseTensor(type))));
}

bool isATorch_NonValueTensorType(MlirType type) {
  return torchMlirTypeIsATorchNonValueTensor(type);
}

bool isATorch_ValueTensorType(MlirType type) {
  return torchMlirTypeIsATorchValueTensor(type);
}

// value constraints

bool isAAnyTorchListOfOptionalTensorValue(MlirValue value) {
  return isAAnyTorchListOfOptionalTensorType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTensorValue(MlirValue value) {
  return isAAnyTorchListOfTensorType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalTensorValue(MlirValue value) {
  return isAAnyTorchOptionalTensorType(mlirValueGetType(value));
}

bool isAAnyTorchTensorValue(MlirValue value) {
  return isAAnyTorchTensorType(mlirValueGetType(value));
}

bool isATorch_ValueTensorValue(MlirValue value) {
  return isATorch_ValueTensorType(mlirValueGetType(value));
}

bool isATorch_NonValueTensorValue(MlirValue value) {
  return isATorch_NonValueTensorType(mlirValueGetType(value));
}

// helpers

PyTorch_NonValueTensorType
PyTorch_NonValueTensorType::getWithLeastStaticInformation(
    DefaultingPyMlirContext context) {
  MlirType valueTensorType =
      torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
          context->get());
  return {context->getRef(), valueTensorType};
}

PyTorch_ValueTensorType PyTorch_ValueTensorType::getWithLeastStaticInformation(
    DefaultingPyMlirContext context) {
  MlirType valueTensorType =
      torchMlirTorchValueTensorTypeGetWithLeastStaticInformation(
          context->get());
  return {context->getRef(), valueTensorType};
}

// type binders

void PyTorch_NonValueTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> sizes, MlirType dtype,
         DefaultingPyMlirContext context) {
        return PyTorch_NonValueTensorType(std::move(sizes), dtype, context);
      },
      py::arg("sizes"), py::arg("dtype"), py::arg("context") = py::none());
  c.def_static(
      "get_with_least_static_information",
      [](DefaultingPyMlirContext context) {
        return PyTorch_NonValueTensorType::getWithLeastStaticInformation(
            context);
      },
      py::arg("context") = py::none());
  c.def("sizes", [](MlirType self) {
    std::vector<int64_t> sizes(torchMlirTorchNonValueTensorTypeGetRank(self));
    if (torchMlirTorchNonValueTensorTypeGetSizes(self, sizes.data()))
      throw py::value_error("no sizes");
    return py::tuple(py::cast(sizes));
  });
  c.def("dtype", [](MlirType self) {
    return torchMlirTorchNonValueTensorTypeGetDtype(self);
  });
}

void PyTorch_ValueTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> sizes, MlirType dtype,
         DefaultingPyMlirContext context) {
        return PyTorch_ValueTensorType(sizes, dtype, context);
      },
      py::arg("sizes"), py::arg("dtype"), py::arg("context") = py::none());
  c.def_static(
      "get_with_least_static_information",
      [](DefaultingPyMlirContext context) {
        return PyTorch_ValueTensorType::getWithLeastStaticInformation(context);
      },
      py::arg("context") = py::none());
  c.def("sizes", [](MlirType self) {
    std::vector<int64_t> sizes(torchMlirTorchValueTensorTypeGetRank(self));
    if (torchMlirTorchValueTensorTypeGetSizes(self, sizes.data()))
      throw py::value_error("no sizes");
    return py::tuple(py::cast(sizes));
  });
  c.def("dtype", [](MlirType self) {
    return torchMlirTorchValueTensorTypeGetDtype(self);
  });
}

PyAnyTorchTensorType PyAnyTorchTensorType::getWithLeastStaticInformation(
    DefaultingPyMlirContext context) {
  MlirType nonValueTensorType =
      torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
          context->get());
  return {context->getRef(), nonValueTensorType};
}

void PyAnyTorchTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> sizes, MlirType dtype,
         DefaultingPyMlirContext context) {
        return PyAnyTorchTensorType(sizes, dtype, context);
      },
      py::arg("sizes"), py::arg("dtype"), py::arg("context") = py::none());
  c.def_static(
      "get_with_least_static_information",
      [](DefaultingPyMlirContext context) {
        return PyAnyTorchTensorType::getWithLeastStaticInformation(context);
      },
      py::arg("context") = py::none());
}

// value binders

void PyAnyTorchOptionalTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalTensorValue>();
}

void PyTorch_NonValueTensorValue::bindDerived(ClassTy &c) {}

void PyTorch_ValueTensorValue::bindDerived(ClassTy &c) {}

void PyAnyTorchTensorValue::bindDerived(ClassTy &c) {

  // @overload __add__(self, other Tensor) -> Tensor
  // aten::__add__.Tensor : (Tensor, Tensor) -> (Tensor)
  c.def(
      "__add__",
      [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other)
          -> PyAnyTorchTensorValue { return add(self, other, 1.0f); },
      "other"_a);

  c.def(
      "__iadd__",
      [](const PyAnyTorchTensorValue &self, const PyAnyTorchTensorValue &other)
          -> PyAnyTorchTensorValue { return add(self, other, 1.0f); },
      "other"_a);

#include "TorchTensor.pybinds.cpp"
}

PyAnyTorchListOfTensorValue mapListToTorchListOfTensorValue(const py::list &l) {
  return py::cast(PyAnyTorchListValue(
                      py::cast(PyAnyTorchListOfTensorType(
                          mlirValueGetType(l[0].cast<PyValue>().get()),
                          DefaultingPyMlirContext::resolve())),
                      l, tag<PyAnyTorchTensorValue>{}))
      .cast<PyAnyTorchListOfTensorValue>();
}

void PyAnyTorchListOfTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::list>(), py::arg("value"));
  c.def(py::init<py::tuple>(), py::arg("value"));
  py::implicitly_convertible<py::list, PyAnyTorchListOfTensorValue>();
  py::implicitly_convertible<py::tuple, PyAnyTorchListOfTensorValue>();
}

PyAnyTorchListOfOptionalTensorValue
mapListToTorchListOfOptionalTensorValue(const py::list &l) {
  return (l.empty() ||
          std::all_of(l.begin(), l.end(), [](auto o) { return o.is_none(); }))
             ? py::cast(PyAnyTorchListValue(l))
                   .cast<PyAnyTorchListOfOptionalTensorValue>()
             : py::cast(PyAnyTorchListValue(
                            py::cast(PyAnyTorchListOfOptionalTensorType(
                                mlirValueGetType(l[0].cast<PyValue>().get()),
                                DefaultingPyMlirContext::resolve())),
                            l, tag<PyAnyTorchTensorValue>{}))
                   .cast<PyAnyTorchListOfOptionalTensorValue>();
}

void PyAnyTorchListOfOptionalTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::list>(), py::arg("value"));
  c.def(py::init<py::tuple>(), py::arg("value"));
  py::implicitly_convertible<py::list, PyAnyTorchListOfOptionalTensorValue>();
  py::implicitly_convertible<py::tuple, PyAnyTorchListOfOptionalTensorValue>();
}

void populateTorchTensorOps(py::module &m) {
  // bind types
  PyTorch_NonValueTensorType::bind(m);
  PyTorch_ValueTensorType::bind(m);
  PyAnyTorchTensorType::bind(m);
  PyAnyTorchOptionalTensorType::bind(m);
  PyAnyTorchListOfOptionalTensorType::bind(m);
  PyAnyTorchListOfTensorType::bind(m);

  // bind values
  PyAnyTorchTensorValue::bind(m);
  PyAnyTorchListOfOptionalTensorValue::bind(m);
  PyAnyTorchListOfTensorValue::bind(m);
  PyAnyTorchOptionalTensorValue::bind(m);
  PyTorch_NonValueTensorValue::bind(m);
  PyTorch_ValueTensorValue::bind(m);
}
} // namespace mlir::torch
