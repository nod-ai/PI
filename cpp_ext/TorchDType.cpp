//
// Created by maksim on 6/13/23.
//

#include "IRModule.h"
#include "TorchTypes.h"
#include "mlir-c/BuiltinTypes.h"

#include "TorchDType.h"

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mlir::python;
using namespace py::literals;

namespace mlir::torch {

DType fromNpDType(const py::dtype &npDType) {
  switch (npDType.num()) {
  case 0: // numpy.bool_
    return DType::bool_;
  case 2: // numpy.uint8
    return DType::uint8;
  case 1: // numpy.int8
    return DType::int8;
  case 3: // numpy.int16
    return DType::int16;
  case 5: // numpy.int32
    return DType::int32;
  case 7: // numpy.int64
    return DType::int64;
  case 23: // numpy.float16
    return DType::float16;
  case 11: // numpy.float32
    return DType::float32;
  case 12: // numpy.float64
    return DType::float64;
  case 14: // numpy.complex64
    return DType::complex32;
  case 15: // numpy.complex128
    return DType::complex64;
  default:
    throw py::value_error(py::repr(npDType).operator std::string() +
                          " unsupported");
  }
}

void populateTorchDType(py::module &m) {
  py::enum_<DType>(m, "dtype", py::arithmetic())
      .value("uint8", DType::uint8)
      .value("int8", DType::int8)
      .value("int16", DType::int16)
      .value("int32", DType::int32)
      .value("int64", DType::int64)
      .value("float16", DType::float16)
      .value("float32", DType::float32)
      .value("float64", DType::float64)
      .value("complex32", DType::complex32)
      .value("complex64", DType::complex64)
      .value("bool", DType::bool_)
      .value("qint8", DType::qint8)
      .value("quint8", DType::quint8)
      .value("bfloat16", DType::bfloat16)
      .def(
          "to_mlir_type",
          [](DType &self, DefaultingPyMlirContext &context) {
            switch (self) {
            case bool_:
              // default is signless
              return mlirIntegerTypeGet(context->get(), 1);
            case uint8:
              return mlirIntegerTypeUnsignedGet(context->get(), 8);
            case int8:
              return mlirIntegerTypeSignedGet(context->get(), 8);
            case int16:
              return mlirIntegerTypeSignedGet(context->get(), 16);
            case int32:
              return mlirIntegerTypeSignedGet(context->get(), 32);
            case int64:
              return mlirIntegerTypeSignedGet(context->get(), 64);
            case float16:
              return mlirF16TypeGet(context->get());
            case float32:
              return mlirF32TypeGet(context->get());
            case float64:
              return mlirF64TypeGet(context->get());
            case complex32:
              return mlirComplexTypeGet(mlirF32TypeGet(context->get()));
            case complex64:
              return mlirComplexTypeGet(mlirF64TypeGet(context->get()));
            case bfloat16:
              return mlirBF16TypeGet(context->get());
            case qint8:
            case quint8:
              throw py::value_error("qint8, quint8 unsupported");
            }
          },
          py::kw_only(), "context"_a = py::none())
      .def(
          "to_torch_value_type",
          [](DType &self, DefaultingPyMlirContext &context) -> py::object {
            switch (self) {
            case bool_:
              return py::cast(PyTorch_BoolType(context.get()));
            case uint8:
            case int8:
            case int16:
            case int32:
            case int64:
              return py::cast(PyTorch_IntType(context.get()));
            case float16:
            case float32:
            case float64:
              return py::cast(PyTorch_FloatType(context.get()));
            case complex32:
            case complex64:
            case qint8:
            case quint8:
            case bfloat16:
              throw py::value_error(
                  py::repr(py::cast(self)).operator std::string() +
                  " unsupported");
            }
          },
          py::kw_only(), "context"_a = py::none())
      .def_static(
          "from_np_type",
          [](const py::type &npType) { return fromNpDType(py::dtype(npType)); })
      .def_static("from_np_type", fromNpDType)
      .def("to_np_type",
           [](DType &self) {
             auto np = py::module::import("numpy");
             switch (self) {
             case bool_:
               return np.attr("bool_");
             case uint8:
               return np.attr("uint8");
             case int8:
               return np.attr("int8");
             case int16:
               return np.attr("int16");
             case int32:
               return np.attr("int32");
             case int64:
               return np.attr("int64");
             case float16:
               return np.attr("float16");
             case float32:
               return np.attr("float32");
             case float64:
               return np.attr("float64");
             case complex32:
               return np.attr("complex64");
             case complex64:
               return np.attr("complex128");
             case bfloat16:
             case qint8:
             case quint8:
               throw py::value_error("bfloat16, qint8, quint8 unsupported");
             }
           })
      .def("is_signless", [](DType &self) { return self == bool_; });
}

} // namespace mlir::torch