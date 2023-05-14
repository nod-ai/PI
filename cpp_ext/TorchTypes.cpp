//
// Created by maksim on 5/13/23.
//

#include "TorchTypes.h"

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

bool isAAnyTorchOptionalTensorType(MlirType type) {
  return ((((torchMlirTypeIsATorchBaseTensor(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchOptionalType(MlirType type) {
  return torchMlirTypeIsATorchOptional(type);
}

// bool isAAnyTorchOptionalScalarType(MlirType type) {
//   return ((((isValidSubtype(type, ::mlir::torch::Torch::NumberType::get(
//                                       type.getContext())))) ||
//            ((torchMlirTypeIsATorchOptional(type))) ||
//            ((torchMlirTypeIsATorchNone(type)))));
// }

// bool isAAnyTorchScalarType(MlirType type) {
//   return (((isValidSubtype(
//       type, ::mlir::torch::Torch::NumberType::get(type.getContext())))));
// }

bool isAAnyTorchTensorType(MlirType type) {
  return (((torchMlirTypeIsATorchBaseTensor(type))));
}

// bool isAAnyTorchType(MlirType type) {
//   return ((((isValidSubtype(type, ::mlir::torch::Torch::NumberType::get(
//                                       type.getContext())))) ||
//            ((torchMlirTypeIsATorchBaseTensor(type))) ||
//            ((torchMlirTypeIsATorchAny(type))) ||
//            ((torchMlirTypeIsATorchBool(type))) ||
//            ((torchMlirTypeIsATorchDict(type))) ||
//            ((torchMlirTypeIsATorchDevice(type))) ||
//            ((torchMlirTypeIsATorchGenerator(type))) ||
//            ((torchMlirTypeIsATorchList(type))) ||
//            ((torchMlirTypeIsATorchLinearParams(type))) ||
//            ((torchMlirTypeIsATorchNumber(type))) ||
//            ((torchMlirTypeIsATorchNnModule(type))) ||
//            ((torchMlirTypeIsATorchNone(type))) ||
//            ((torchMlirTypeIsATorchOptional(type))) ||
//            ((torchMlirTypeIsATorchString(type))) ||
//            ((torchMlirTypeIsATorchTuple(type))) ||
//            ((torchMlirTypeIsATorchUnion(type)))));
// }

bool isATorch_BoolType(MlirType type) {
  return (((torchMlirTypeIsATorchBool(type))));
}

bool isATorch_DeviceType(MlirType type) {
  return (((torchMlirTypeIsATorchDevice(type))));
}

bool isATorch_DictType(MlirType type) {
  return (((torchMlirTypeIsATorchDict(type))));
}

bool isATorch_FloatType(MlirType type) {
  return (((torchMlirTypeIsATorchFloat(type))));
}

bool isATorch_IntType(MlirType type) {
  return (((torchMlirTypeIsATorchInt(type))));
}

bool isATorch_LinearParamsType(MlirType type) {
  return (((torchMlirTypeIsATorchLinearParams(type))));
}

bool isATorch_NnModuleType(MlirType type) {
  return (((torchMlirTypeIsATorchNnModule(type))));
}

bool isATorch_NonValueTensorType(MlirType type) {
  return (((torchMlirTypeIsATorchNonValueTensor(type))));
}

bool isATorch_NoneType(MlirType type) {
  return (((torchMlirTypeIsATorchNone(type))));
}

bool isATorch_NumberType(MlirType type) {
  return (((torchMlirTypeIsATorchNumber(type))));
}

bool isATorch_StringType(MlirType type) {
  return (((torchMlirTypeIsATorchString(type))));
}

bool isATorch_TupleType(MlirType type) {
  return (((torchMlirTypeIsATorchTuple(type))));
}

bool isATorch_ValueTensorType(MlirType type) {
  return (((torchMlirTypeIsATorchValueTensor(type))));
}

void PyAnyTorchListType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](MlirType element_type, DefaultingPyMlirContext context) {
        MlirType listType = torchMlirTorchListTypeGet(element_type);
        return PyAnyTorchListType(context->getRef(), listType);
      },
      py::arg("element_type"), py::arg("context") = py::none(),
      "Create a list type.");
  c.def(
      "get_contained_type",
      [](PyAnyTorchListType &self) {
        return torchMlirTorchListTypeGetContainedType(self.get());
      },
      "Get list element type.");
}
void PyAnyTorchOptionalType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](MlirType containedType, DefaultingPyMlirContext context) {
        MlirType optionalType = torchMlirTorchOptionalTypeGet(containedType);
        return PyAnyTorchOptionalType(context->getRef(), optionalType);
      },
      py::arg("element_type"), py::arg("context") = py::none(),
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
        [](DefaultingPyMlirContext context) {                                  \
          MlirType containedType =                                             \
              torchMlir##CONCRETETYPE##TypeGet(context->get());                \
          MlirType listType =                                                  \
              torchMlirTorchListTypeGetContainedType(containedType);           \
          return PyAnyTorchListOf##CONCRETETYPE##Type(context->getRef(),       \
                                                      listType);               \
        },                                                                     \
        py::arg("context") = py::none(), "Create a " #CONCRETETYPE " type.");  \
  }

FORALL_LIST_BASE_CONCRETE_TYPES(DEFINE_LIST_BASE_CONCRETE_TYPE)
#undef DEFINE_LIST_BASE_CONCRETE_TYPE

#define DEFINE_OPTIONAL_BASE_CONCRETE_TYPE(CONCRETETYPE)                       \
  void PyAnyTorchOptional##CONCRETETYPE##Type::bindDerived(ClassTy &c) {       \
    c.def_static(                                                              \
        "get",                                                                 \
        [](DefaultingPyMlirContext context) {                                  \
          MlirType containedType =                                             \
              torchMlirTorch##CONCRETETYPE##TypeGet(context->get());           \
          MlirType optionalType =                                              \
              torchMlirTorchOptionalTypeGetContained(containedType);           \
          return PyAnyTorchOptional##CONCRETETYPE##Type(context->getRef(),     \
                                                        optionalType);         \
        },                                                                     \
        py::arg("context") = py::none(), "Create a " #CONCRETETYPE " type.");  \
  }

FORALL_OPTIONAL_BASE_CONCRETE_TYPES(DEFINE_OPTIONAL_BASE_CONCRETE_TYPE)
#undef DEFINE_OPTIONAL_BASE_CONCRETE_TYPE

// TODO(max): missing tensor and some others (check commented out types in
// mlir.__init__.py
// no underscores here because the CAPI getters don't have underscores...
#define FORALL_SCALAR_TYPES(_)                                                 \
  _(Bool)                                                                      \
  _(Device)                                                                    \
  _(Float)                                                                     \
  _(Int)                                                                       \
  _(LinearParams)                                                              \
  _(None)                                                                      \
  _(Number)                                                                    \
  _(String)

#define DEFINE_SCALAR_TYPE(SCALARTYPE)                                         \
  void PyTorch_##SCALARTYPE##Type::bindDerived(ClassTy &c) {                   \
    c.def_static(                                                              \
        "get",                                                                 \
        [](DefaultingPyMlirContext context) {                                  \
          MlirType type = torchMlirTorch##SCALARTYPE##TypeGet(context->get()); \
          return PyTorch_##SCALARTYPE##Type(context->getRef(), type);          \
        },                                                                     \
        py::arg("context") = py::none(), "Create a " #SCALARTYPE " type.");    \
  }
FORALL_SCALAR_TYPES(DEFINE_SCALAR_TYPE)
#undef DEFINE_SCALAR_TYPE

void PyTorch_DictType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](MlirType keyType, MlirType valueType,
         DefaultingPyMlirContext context) {
        MlirType dictType = torchMlirTorchDictTypeGet(keyType, valueType);
        return PyTorch_DictType(context->getRef(), dictType);
      },
      py::arg("key_type"), py::arg("value_type"),
      py::arg("context") = py::none(), "Create a dict type.");
  c.def("get_key_type", torchMlirTorchDictTypeGetKeyType);
  c.def("get_value_type", torchMlirTorchDictTypeGetValueType);
}

void PyTorch_TupleType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](const py::tuple &elementTypes, DefaultingPyMlirContext context) {
        auto types = elementTypes.cast<std::vector<MlirType>>();
        MlirType tupleType = torchMlirTorchTupleTypeGet(
            context->get(), types.size(), types.data());
        return PyTorch_TupleType(context->getRef(), tupleType);
      },
      py::arg("element_types"), py::arg("context") = py::none(),
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
      [](MlirStringRef name, DefaultingPyMlirContext context) {
        MlirType nnModuleType =
            torchMlirTorchNnModuleTypeGet(context->get(), name);
        return PyTorch_NnModuleType(context->getRef(), nnModuleType);
      },
      py::arg("element_types"), py::arg("context") = py::none(),
      "Create a tuple type.");
}

void PyTorch_NonValueTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> sizes, MlirType dtype,
         DefaultingPyMlirContext context) {
        MlirType nonValueTensorType = torchMlirTorchNonValueTensorTypeGet(
            context->get(), sizes.size(), sizes.data(), dtype);
        return PyTorch_NonValueTensorType(context->getRef(),
                                          nonValueTensorType);
      },
      py::arg("sizes"), py::arg("dtype"), py::arg("context") = py::none());
  c.def_static(
      "get_with_least_static_information",
      [](DefaultingPyMlirContext context) {
        MlirType nonValueTensorType =
            torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
                context->get());
        return PyTorch_NonValueTensorType(context->getRef(),
                                          nonValueTensorType);
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
        MlirType valueTensorType = torchMlirTorchValueTensorTypeGet(
            context->get(), sizes.size(), sizes.data(), dtype);
        return PyTorch_ValueTensorType(context->getRef(), valueTensorType);
      },
      py::arg("sizes"), py::arg("dtype"), py::arg("context") = py::none());
  c.def_static(
      "get_with_least_static_information",
      [](DefaultingPyMlirContext context) {
        MlirType valueTensorType =
            torchMlirTorchValueTensorTypeGetWithLeastStaticInformation(
                context->get());
        return PyTorch_ValueTensorType(context->getRef(), valueTensorType);
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
#define BIND_TYPE(TYPE) PyTorch##TYPE##Type::bind(m);
  FORALL_CONCRETE_TYPES(BIND_TYPE)
#undef BIND_TYPE
}

} // namespace mlir::torch
