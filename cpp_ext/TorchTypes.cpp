//
// Created by maksim on 5/13/23.
//

#include "TorchTypes.h"

#include <utility>

#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace pybind11::detail {
/// Casts string -> MlirStringRef.
template <> struct type_caster<MlirStringRef> {
  PYBIND11_TYPE_CASTER(MlirStringRef, _("MlirStringRef"));
  bool load(handle src, bool) {
    auto s = py::reinterpret_borrow<py::object>(src).cast<std::string_view>();
    value = mlirStringRefCreate(s.data(), s.length());
    return true;
  }
};
} // namespace pybind11::detail

namespace mlir::torch {

bool torchMlirTypeIsATorchBaseTensor(MlirType type) {
  return torchMlirTypeIsATorchValueTensor(type) ||
         torchMlirTypeIsATorchNonValueTensor(type);
}

bool isAAnyTorchDictKeyType(MlirType type) {
  return ((((torchMlirTypeIsATorchAny(type))) ||
           ((torchMlirTypeIsATorchInt(type))) ||
           ((torchMlirTypeIsATorchBool(type))) ||
           ((torchMlirTypeIsATorchFloat(type))) ||
           ((torchMlirTypeIsATorchString(type))) ||
           ((torchMlirTypeIsATorchBaseTensor(type)))));
}

bool isAAnyTorchListOfOptionalIntType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           (((torchMlirTypeIsATorchInt(
                torchMlirTorchListTypeGetContainedType(type)))) ||
            ((torchMlirTypeIsATorchOptional(
                torchMlirTorchListTypeGetContainedType(type)))) ||
            ((torchMlirTypeIsATorchNone(
                torchMlirTorchListTypeGetContainedType(type)))))));
}

bool isAAnyTorchListOfTorchBoolType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           ((torchMlirTypeIsATorchBool(
               torchMlirTorchListTypeGetContainedType(type))))));
}

bool isAAnyTorchListOfTorchIntType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           ((torchMlirTypeIsATorchInt(
               torchMlirTorchListTypeGetContainedType(type))))));
}

bool isAAnyTorchListOfTorchFloatType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           ((torchMlirTypeIsATorchFloat(
               torchMlirTorchListTypeGetContainedType(type))))));
}

bool isAAnyTorchListOfTorchStringType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           ((torchMlirTypeIsATorchString(
               torchMlirTorchListTypeGetContainedType(type))))));
}

bool isAAnyTorchOptionalFloatType(MlirType type) {
  return ((((torchMlirTypeIsATorchFloat(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchListType(MlirType type) {
  return ((torchMlirTypeIsATorchList(type)));
}

bool isAAnyTorchOptionalBoolType(MlirType type) {
  return ((((torchMlirTypeIsATorchBool(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchOptionalDeviceType(MlirType type) {
  return ((((torchMlirTypeIsATorchDevice(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchOptionalGeneratorType(MlirType type) {
  return ((((torchMlirTypeIsATorchGenerator(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchOptionalIntType(MlirType type) {
  return ((((torchMlirTypeIsATorchInt(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchOptionalListOfTorchIntType(MlirType type) {
  return (((((torchMlirTypeIsATorchList(type))) &&
            ((torchMlirTypeIsATorchInt(
                torchMlirTorchListTypeGetContainedType(type))))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchOptionalStringType(MlirType type) {
  return ((((torchMlirTypeIsATorchString(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchOptionalType(MlirType type) {
  return torchMlirTypeIsATorchOptional(type);
}

bool isAAnyTorchOptionalScalarType(MlirType type) {
  auto isValidSubtype = torchMlirTypeIsValidSubtype(
      type, torchMlirTorchNumberTypeGet(mlirTypeGetContext(type)));
  return ((((isValidSubtype)) || ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchScalarType(MlirType type) {
  return torchMlirTypeIsValidSubtype(
      type, torchMlirTorchNumberTypeGet(mlirTypeGetContext(type)));
}

bool isAAnyTorchType(MlirType type) {

  auto isValidSubtype = torchMlirTypeIsValidSubtype(
      type, torchMlirTorchNumberTypeGet(mlirTypeGetContext(type)));
  return ((((isValidSubtype)) || ((torchMlirTypeIsATorchBaseTensor(type))) ||
           ((torchMlirTypeIsATorchAny(type))) ||
           ((torchMlirTypeIsATorchBool(type))) ||
           ((torchMlirTypeIsATorchDict(type))) ||
           ((torchMlirTypeIsATorchDevice(type))) ||
           ((torchMlirTypeIsATorchGenerator(type))) ||
           ((torchMlirTypeIsATorchList(type))) ||
           ((torchMlirTypeIsATorchLinearParams(type))) ||
           ((torchMlirTypeIsATorchNumber(type))) ||
           ((torchMlirTypeIsATorchNnModule(type))) ||
           ((torchMlirTypeIsATorchNone(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchString(type))) ||
           ((torchMlirTypeIsATorchTuple(type))) ||
           ((torchMlirTypeIsATorchUnion(type)))));
}

#define DECLARE_ISA_UNDERSCORE_TYPE(UNDERSCORETYPE)                            \
  bool isATorch_##UNDERSCORETYPE##Type(MlirType type) {                        \
    return torchMlirTypeIsATorch##UNDERSCORETYPE(type);                        \
  }
FORALL_UNDERSCORE_TYPES(DECLARE_ISA_UNDERSCORE_TYPE)
#undef DECLARE_ISA_UNDERSCORE_TYPE

void PyAnyTorchListType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](MlirType containedType, DefaultingPyMlirContext &context) {
        return PyAnyTorchListType(containedType, context.get());
      },
      py::arg("contained_type"), py::kw_only(), py::arg("context") = py::none(),
      "Create a list type.");
  c.def(
      "contained_type",
      [](PyAnyTorchListType &self) {
        return torchMlirTorchListTypeGetContainedType(self.get());
      },
      "Get list element type.");
}

void PyAnyTorchOptionalType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](MlirType containedType, DefaultingPyMlirContext &context) {
        return PyAnyTorchOptionalType(containedType, context.get());
      },
      py::arg("contained_type"), py::kw_only(), py::arg("context") = py::none(),
      "Create a optional type.");
  c.def(
      "get_contained_type",
      [](PyAnyTorchOptionalType &self) {
        return torchMlirTorchOptionalTypeGetContained(self.get());
      },
      "Get optional contained type.");
}

#define DEFINE_LIST_BASE_CONCRETE_TYPE(CONCRETETYPE)                           \
  void PyAnyTorchListOf##CONCRETETYPE##Type::bindDerived(ClassTy &c) {         \
    c.def_static(                                                              \
        "get",                                                                 \
        [](DefaultingPyMlirContext &context) {                                 \
          return PyAnyTorchListOf##CONCRETETYPE##Type(context.get());          \
        },                                                                     \
        py::kw_only(), py::arg("context") = py::none(),                        \
        "Create a " #CONCRETETYPE " type.");                                   \
  }

FORALL_LIST_BASE_CONCRETE_TYPES(DEFINE_LIST_BASE_CONCRETE_TYPE)
#undef DEFINE_LIST_BASE_CONCRETE_TYPE

#define DEFINE_OPTIONAL_BASE_CONCRETE_TYPE(CONCRETETYPE)                       \
  void PyAnyTorchOptional##CONCRETETYPE##Type::bindDerived(ClassTy &c) {       \
    c.def_static(                                                              \
        "get",                                                                 \
        [](DefaultingPyMlirContext &context) {                                 \
          return PyAnyTorchOptional##CONCRETETYPE##Type(context.get());        \
        },                                                                     \
        py::kw_only(), py::arg("context") = py::none(),                        \
        "Create a " #CONCRETETYPE " type.");                                   \
  }

FORALL_OPTIONAL_BASE_CONCRETE_TYPES(DEFINE_OPTIONAL_BASE_CONCRETE_TYPE)
#undef DEFINE_OPTIONAL_BASE_CONCRETE_TYPE

#define DEFINE_SCALAR_TYPE(SCALARTYPE)                                         \
  void PyTorch_##SCALARTYPE##Type::bindDerived(ClassTy &c) {                   \
    c.def_static(                                                              \
        "get",                                                                 \
        [](DefaultingPyMlirContext &context) {                                 \
          return PyTorch_##SCALARTYPE##Type(context.get());                    \
        },                                                                     \
        py::kw_only(), py::arg("context") = py::none(),                        \
        "Create a " #SCALARTYPE " type.");                                     \
  }
FORALL_SCALAR_TYPES(DEFINE_SCALAR_TYPE)
#undef DEFINE_SCALAR_TYPE

void PyTorch_DictType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](MlirType keyType, MlirType valueType,
         DefaultingPyMlirContext &context) {
        return PyTorch_DictType(keyType, valueType, context.get());
      },
      py::arg("key_type"), py::arg("value_type"), py::kw_only(),
      py::arg("context") = py::none(), "Create a dict type.");
  c.def("get_key_type", torchMlirTorchDictTypeGetKeyType);
  c.def("get_value_type", torchMlirTorchDictTypeGetValueType);
}

void PyTorch_TupleType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](const py::tuple &elementTypes, DefaultingPyMlirContext &context) {
        auto types = elementTypes.cast<std::vector<MlirType>>();
        return PyTorch_TupleType(types, context.get());
      },
      py::arg("element_types"), py::kw_only(), py::arg("context") = py::none(),
      "Create a tuple type.");
  c.def("__len__",
        [](MlirType self) { return torchMlirTorchTupleTypeGetNumTypes(self); });
  c.def("__getitem__", [](MlirType self, intptr_t pos) {
    return torchMlirTorchTupleTypeGetType(self, pos);
  });
}

void PyTorch_NnModuleType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](MlirStringRef name, DefaultingPyMlirContext &context) {
        return PyTorch_NnModuleType(name, context.get());
      },
      py::arg("element_types"), py::kw_only(), py::arg("context") = py::none(),
      "Create a tuple type.");
}

void populateTorchMLIRTypes(py::module &m) {
  // order here matters - these have to be found before the types that depends
  // on them...
  PyAnyTorchListType::bind(m);
  PyAnyTorchOptionalType::bind(m);

#define BIND_TYPE(TYPE) PyAnyTorchListOf##TYPE##Type::bind(m);
  FORALL_LIST_BASE_CONCRETE_TYPES(BIND_TYPE)
#undef BIND_TYPE

#define BIND_TYPE(TYPE) PyAnyTorchOptional##TYPE##Type::bind(m);
  FORALL_OPTIONAL_BASE_CONCRETE_TYPES(BIND_TYPE)
#undef BIND_TYPE
#define BIND_TYPE(TYPE) PyTorch_##TYPE##Type::bind(m);
  FORALL_SCALAR_TYPES(BIND_TYPE)
#undef BIND_TYPE

  PyTorch_DictType::bind(m);
  PyTorch_TupleType::bind(m);
  PyTorch_NnModuleType::bind(m);
  PyAnyTorchScalarType::bind(m);
}

} // namespace mlir::torch
