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

void PyAnyTorchListOfTensorValue::bindDerived(ClassTy &c) {}

void PyAnyTorchOptionalTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalTensorValue>();
}

void PyTorch_NonValueTensorValue::bindDerived(ClassTy &c) {}

void PyTorch_ValueTensorValue::bindDerived(ClassTy &c) {}

void PyAnyTorchTensorValue::bindDerived(ClassTy &c) {
#include "TorchTensor.pybinds.cpp"
}

void PyAnyTorchListOfOptionalTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::list>(), py::arg("value"));
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
