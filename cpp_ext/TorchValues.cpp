//
// Created by maksim on 5/13/23.
//

#include "TorchValues.h"
#include "TorchDType.h"
#include "TorchTypes.h"
#include "TorchOps.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

using namespace py::literals;

namespace mlir::torch {

bool isAAnyTorchDictKeyValue(MlirValue value) {
  return isAAnyTorchDictKeyType(mlirValueGetType(value));
}

bool isAAnyTorchListOfOptionalIntValue(MlirValue value) {
  return isAAnyTorchListOfOptionalIntType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchBoolValue(MlirValue value) {
  return isAAnyTorchListOfTorchBoolType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchIntValue(MlirValue value) {
  return isAAnyTorchListOfTorchIntType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchFloatValue(MlirValue value) {
  return isAAnyTorchListOfTorchFloatType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchStringValue(MlirValue value) {
  return isAAnyTorchListOfTorchStringType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalFloatValue(MlirValue value) {
  return isAAnyTorchOptionalFloatType(mlirValueGetType(value));
}

bool isAAnyTorchListValue(MlirValue value) {
  return isAAnyTorchListType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalBoolValue(MlirValue value) {
  return isAAnyTorchOptionalBoolType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalDeviceValue(MlirValue value) {
  return isAAnyTorchOptionalDeviceType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalGeneratorValue(MlirValue value) {
  return isAAnyTorchOptionalGeneratorType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalIntValue(MlirValue value) {
  return isAAnyTorchOptionalIntType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalListOfTorchIntValue(MlirValue value) {
  return isAAnyTorchOptionalListOfTorchIntType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalStringValue(MlirValue value) {
  return isAAnyTorchOptionalStringType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalValue(MlirValue value) {
  return isAAnyTorchOptionalType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalScalarValue(MlirValue value) {
  return isAAnyTorchOptionalScalarType(mlirValueGetType(value));
}

bool isAAnyTorchScalarValue(MlirValue value) {
  return isAAnyTorchScalarType(mlirValueGetType(value));
}

bool isAAnyTorchValue(MlirValue value) {
  return isAAnyTorchType(mlirValueGetType(value));
}

#define DECLARE_ISA_UNDERSCORE_VALUE(TORCHTYPE)                                \
  bool isATorch_##TORCHTYPE##Value(MlirValue value) {                          \
    return isATorch_##TORCHTYPE##Type(mlirValueGetType(value));                \
  }
FORALL_UNDERSCORE_TYPES(DECLARE_ISA_UNDERSCORE_VALUE)
#undef DECLARE_ISA_UNDERSCORE_VALUE

py::object mapListToPyTorchListValue(const py::list &list) {
  if (list.empty())
    throw std::runtime_error("Can't cast empty list");

  MlirType containedType;
  auto &ctx = DefaultingPyMlirContext::resolve();

  if (py::isinstance<py::int_>(list[0])) {
    containedType = torchMlirTorchIntTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_IntValue>(list);
  } else if (py::isinstance<py::float_>(list[0])) {
    containedType = torchMlirTorchFloatTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_FloatValue>(list);
  } else if (py::isinstance<py::bool_>(list[0])) {
    containedType = torchMlirTorchBoolTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_BoolValue>(list);
  } else if (py::isinstance<py::str>(list[0])) {
    containedType = torchMlirTorchStringTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_StringValue>(list);
  } else if (list[0].is_none()) {
    containedType = torchMlirTorchNoneTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_NoneValue>(list);
  } else {
    if (!py::isinstance<PyValue>(list[0]))
      throw std::runtime_error("Can't infer list element type.");
    containedType = mlirValueGetType(list[0].cast<PyValue>().get());
  }

  auto pyType =
      py::cast(PyType(ctx.getRef(), torchMlirTorchListTypeGet(containedType)));
  return pyType;
}

void PyAnyTorchListValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::list>(), py::arg("value"));
  c.def(py::init<py::tuple>(), py::arg("value"));
  py::implicitly_convertible<py::list, PyAnyTorchListValue>();
  py::implicitly_convertible<py::tuple, PyAnyTorchListValue>();
}

#define DEFINE_LIST_BASE_CONCRETE_VALUE(TORCHTYPE, SCALARTYPE)                 \
  void PyAnyTorchListOf##TORCHTYPE##Value::bindDerived(ClassTy &c) {           \
    c.def(py::init<py::list>(), py::arg("value"));                             \
    c.def(py::init<py::tuple>(), py::arg("value"));                            \
    c.def(                                                                     \
        "__getitem__",                                                         \
        [](PyAnyTorchListOf##TORCHTYPE##Value & self,                          \
           const PyTorch_IntValue &idx) -> PyTorch_##SCALARTYPE##Value {       \
          MlirType containedType = torchMlirTorchListTypeGetContainedType(     \
              mlirValueGetType(self.get()));                                   \
          return PyGlobals::get()                                              \
              .lookupOperationClass("torch.aten.__getitem__.t")                \
              .value()(py::cast(containedType), self, idx)                     \
              .cast<PyTorch_##SCALARTYPE##Value>();                            \
        });                                                                    \
    py::implicitly_convertible<py::list,                                       \
                               PyAnyTorchListOf##TORCHTYPE##Value>();          \
    py::implicitly_convertible<py::tuple,                                      \
                               PyAnyTorchListOf##TORCHTYPE##Value>();          \
  }
FORALL_LIST_BASE_CONCRETE_TYPES_WITH_TYPE(DEFINE_LIST_BASE_CONCRETE_VALUE)
#undef DEFINE_LIST_BASE_CONCRETE_VALUE

void PyAnyTorchOptionalGeneratorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalGeneratorValue>();
}

void PyAnyTorchOptionalValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalValue>();
}

#define DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(TORCHTYPE, CPPTYPE)                \
  void PyAnyTorchOptional##TORCHTYPE##Value::bindDerived(ClassTy &c) {         \
    c.def(py::init<py::none>(), py::arg("value"));                             \
    c.def(py::init<CPPTYPE>(), py::arg("value"));                              \
    py::implicitly_convertible<py::none,                                       \
                               PyAnyTorchOptional##TORCHTYPE##Value>();        \
    py::implicitly_convertible<CPPTYPE,                                        \
                               PyAnyTorchOptional##TORCHTYPE##Value>();        \
  }
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Bool, bool)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Device, std::string)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Float, float)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(String, std::string)
#undef DEFINE_OPTIONAL_BASE_CONCRETE_VALUE

void PyAnyTorchOptionalIntValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  c.def(py::init<int>(), py::arg("value"));
  c.def(py::init<DType>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalIntValue>();
  py::implicitly_convertible<int, PyAnyTorchOptionalIntValue>();
  py::implicitly_convertible<DType, PyAnyTorchOptionalIntValue>();
}

void PyAnyTorchOptionalScalarValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  c.def(py::init<int>(), py::arg("value"));
  c.def(py::init<float>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalScalarValue>();
  py::implicitly_convertible<int, PyAnyTorchOptionalScalarValue>();
  py::implicitly_convertible<float, PyAnyTorchOptionalScalarValue>();
}

void PyAnyTorchOptionalListOfTorchIntValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), py::arg("value"));
  c.def(py::init<py::list>(), py::arg("value"));
  c.def(py::init<py::tuple>(), py::arg("value"));
  py::implicitly_convertible<py::none, PyAnyTorchOptionalListOfTorchIntValue>();
  py::implicitly_convertible<py::list, PyAnyTorchOptionalListOfTorchIntValue>();
  py::implicitly_convertible<py::tuple,
                             PyAnyTorchOptionalListOfTorchIntValue>();
}

#define DEFINE_BIND_SCALAR_VALUE(TORCHTYPE)                                    \
  void PyTorch_##TORCHTYPE##Value::bindDerived(ClassTy &c) {}
DEFINE_BIND_SCALAR_VALUE(Any)
DEFINE_BIND_SCALAR_VALUE(LinearParams)
DEFINE_BIND_SCALAR_VALUE(None)
DEFINE_BIND_SCALAR_VALUE(Number)
#undef DEFINE_BIND_SCALAR_VALUE

#define DEFINE_BIND_SCALAR_VALUE(TORCHTYPE, CPPTYPE, DUNDER, ATTR)             \
  void PyTorch_##TORCHTYPE##Value::bindDerived(ClassTy &c) {                   \
    c.def(py::init<CPPTYPE>(), py::arg("value"));                              \
    c.def("__" #DUNDER "__", [](py::object &self) {                            \
      return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))               \
          .attr(#ATTR "Attr")(self.attr("owner")                               \
                                  .attr("attributes")                          \
                                  .attr("__getitem__")("value"))               \
          .attr("value")                                                       \
          .cast<CPPTYPE>();                                                    \
    });                                                                        \
                                                                               \
    py::implicitly_convertible<CPPTYPE, PyTorch_##TORCHTYPE##Value>();         \
  }
DEFINE_BIND_SCALAR_VALUE(Bool, bool, bool, Bool)
DEFINE_BIND_SCALAR_VALUE(Device, std::string, str, String)
DEFINE_BIND_SCALAR_VALUE(String, std::string, str, String)
#undef DEFINE_BIND_SCALAR_VALUE

void PyTorch_IntValue::bindDerived(ClassTy &c) {
  c.def(py::init<int>(), py::arg("value"));
  c.def(py::init<DType>(), py::arg("value"));
  c.def("__int__", [](py::object &self) {
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("IntegerAttr")(
            self.attr("owner").attr("attributes").attr("__getitem__")("value"))
        .attr("value")
        .cast<int>();
  });
  c.def(
      "__add__",
      [](const PyTorch_IntValue &self, const PyTorch_IntValue &other)
          -> PyTorch_IntValue { return add(self, other); },
      "other"_a);
  c.def(
      "__sub__",
      [](const PyTorch_IntValue &self, const PyTorch_IntValue &other)
          -> PyTorch_IntValue { return sub(self, other); },
      "other"_a);
  c.def(
      "__mul__",
      [](const PyTorch_IntValue &self, const PyTorch_IntValue &other)
          -> PyTorch_IntValue { return mul(self, other); },
      "other"_a);
  c.def(
      "__truediv__",
      [](const PyTorch_IntValue &self, const PyTorch_IntValue &other)
          -> PyTorch_FloatValue { return div(self, other); },
      "other"_a);
  c.def(
      "__floordiv__",
      [](const PyTorch_IntValue &self, const PyTorch_IntValue &other)
          -> PyTorch_IntValue { return floordiv(self, other); },
      "other"_a);
  py::implicitly_convertible<int, PyTorch_IntValue>();
  py::implicitly_convertible<DType, PyTorch_IntValue>();
}

void PyTorch_FloatValue::bindDerived(ClassTy &c) {
  c.def(py::init<float>(), py::arg("value"));
  c.def("__"
        "float"
        "__",
        [](py::object &self) {
          return py::module::import("pi.mlir."
                                    "ir")
              .attr("Float"
                    "Attr")(self.attr("owner")
                                .attr("attributes")
                                .attr("__getitem__")("value"))
              .attr("value")
              .cast<float>();
        });
  c.def(
      "__sub__",
      [](const PyTorch_FloatValue &self, const PyTorch_FloatValue &other)
          -> PyTorch_FloatValue { return sub(self, other); },
      "other"_a);
  c.def(
      "__mul__",
      [](const PyTorch_FloatValue &self, const PyTorch_FloatValue &other)
          -> PyTorch_FloatValue { return mul(self, other); },
      "other"_a);
  c.def(
      "__truediv__",
      [](const PyTorch_FloatValue &self, const PyTorch_FloatValue &other)
          -> PyTorch_FloatValue { return div(self, other); },
      "other"_a);
  py::implicitly_convertible<float, PyTorch_FloatValue>();
}

void PyTorch_DictValue::bindDerived(ClassTy &c) {}
void PyTorch_TupleValue::bindDerived(ClassTy &c) {}
void PyTorch_NnModuleValue::bindDerived(ClassTy &c) {}
void PyAnyTorchScalarValue::bindDerived(ClassTy &c) {
  c.def("__repr__", [](PyAnyTorchScalarValue &self) {
    auto origRepr =
        pybind11::repr(pybind11::cast(PyValue(self))).cast<std::string>();
    return std::regex_replace(origRepr, std::regex("Value"),
                              "AnyTorchScalarValue");
  });
  c.def(py::init<int>(), py::arg("value"));
  c.def(py::init<float>(), py::arg("value"));
  py::implicitly_convertible<int, PyAnyTorchScalarValue>();
  py::implicitly_convertible<float, PyAnyTorchScalarValue>();
};

////////////////////////////////////////////////////////////////////////////////

void populateTorchMLIRValues(py::module &m) {
  PyAnyTorchListValue::bind(m);
  PyAnyTorchOptionalValue::bind(m);

#define BIND_VALUE(VALUE) PyAnyTorchListOf##VALUE##Value::bind(m);
  FORALL_LIST_BASE_CONCRETE_TYPES(BIND_VALUE)
#undef BIND_VALUE

#define BIND_VALUE(VALUE) PyAnyTorchOptional##VALUE##Value::bind(m);
  FORALL_OPTIONAL_BASE_CONCRETE_TYPES(BIND_VALUE)
  BIND_VALUE(Scalar)
#undef BIND_VALUE

  PyAnyTorchOptionalListOfTorchIntValue::bind(m);

#define BIND_VALUE(VALUE) PyTorch_##VALUE##Value::bind(m);
  FORALL_SCALAR_TYPES(BIND_VALUE)
#undef BIND_VALUE

  PyTorch_DictValue::bind(m);
  PyTorch_TupleValue::bind(m);
  PyTorch_NnModuleValue::bind(m);
  PyAnyTorchScalarValue::bind(m);
}

} // namespace mlir::torch