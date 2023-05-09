#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "torch-mlir-c/TorchTypes.h"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace mlir::python;
using namespace mlir::python::adaptors;

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

PYBIND11_MODULE(_pi_mlir, m) {

  //============================================================================
  // Types
  //============================================================================

#define TORCH_MLIR_FORALL_NUMBER_TYPES(_)                                      \
  _(Any)                                                                       \
  _(Bool)                                                                      \
  _(Device)                                                                    \
  _(Float)                                                                     \
  _(Generator)                                                                 \
  _(Int)                                                                       \
  _(LinearParams)                                                              \
  _(None)                                                                      \
  _(Number)                                                                    \
  _(QInt8)                                                                     \
  _(QUInt8)                                                                    \
  _(String)

#define TORCH_MLIR_FORALL_CONTAINER_TYPES(_)                                   \
  _(Dict)                                                                      \
  _(List)                                                                      \
  _(NnModule)                                                                  \
  _(NonValueTensor)                                                            \
  _(Optional)                                                                  \
  _(Tuple)                                                                     \
  _(Union)                                                                     \
  _(ValueTensor)

#define DEFINE_SUBTYPE(TTT)                                                    \
  mlir_type_subclass(m, "Torch" #TTT "Type", torchMlirTypeIsATorch##TTT)       \
      .def_classmethod(                                                        \
          "get",                                                               \
          [](const py::object &cls, MlirContext context) {                     \
            return cls(torchMlirTorch##TTT##TypeGet(context));                 \
          },                                                                   \
          py::arg("cls"), py::arg("context") = py::none());
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_SUBTYPE)
#undef DEFINE_SUBTYPE

  mlir_type_subclass(m, "TorchNnModuleType", torchMlirTypeIsATorchNnModule)
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirStringRef name, MlirContext context) {
            return cls(torchMlirTorchNnModuleTypeGet(context, name));
          },
          py::arg("cls"), py::arg("name"), py::arg("context") = py::none());

  mlir_type_subclass(m, "TorchTupleType", torchMlirTypeIsATorchTuple)
      .def_classmethod(
          "get",
          [](const py::object &cls, const py::tuple &containedTypes,
             MlirContext context) {
            auto types = containedTypes.cast<std::vector<MlirType>>();
            return cls(torchMlirTorchTupleTypeGet(context, types.size(),
                                                  types.data()));
          },
          py::arg("cls"), py::arg("contained_types"),
          py::arg("context") = py::none())
      .def("__len__",
           [](MlirType self) {
             return torchMlirTorchTupleTypeGetNumTypes(self);
           })
      .def("__getitem__", [](MlirType self, intptr_t pos) {
        return torchMlirTorchTupleTypeGetType(self, pos);
      });

  mlir_type_subclass(m, "TorchOptionalType", torchMlirTypeIsATorchOptional)
      .def_classmethod(
          "get", [](const py::object &cls, MlirType containedType) {
            return cls(torchMlirTorchOptionalTypeGet(containedType));
          });

  mlir_type_subclass(m, "TorchUnionType", torchMlirTypeIsATorchUnion)
      .def_classmethod(
          "get",
          [](const py::object &cls, const py::tuple &containedTypes,
             MlirContext context) {
            auto types = containedTypes.cast<std::vector<MlirType>>();
            return cls(torchMlirTorchUnionTypeGet(context, types.size(),
                                                  types.data()));
          },
          py::arg("cls"), py::arg("contained_types"),
          py::arg("context") = py::none())
      .def("__len__",
           [](MlirType self) {
             return torchMlirTorchUnionTypeGetNumTypes(self);
           })
      .def("__getitem__", [](MlirType self, intptr_t pos) {
        return torchMlirTorchUnionTypeGetType(self, pos);
      });

  mlir_type_subclass(m, "TorchListType", torchMlirTypeIsATorchList)
      .def_classmethod("get",
                       [](const py::object &cls, MlirType containedType) {
                         return cls(torchMlirTorchListTypeGet(containedType));
                       });

  mlir_type_subclass(m, "TorchNonValueTensorType",
                     torchMlirTypeIsATorchNonValueTensor)
      .def_classmethod(
          "get",
          [](const py::object &cls, std::vector<int64_t> sizes, MlirType dtype,
             MlirContext context) {
            return cls(torchMlirTorchNonValueTensorTypeGet(
                context, sizes.size(), sizes.data(), dtype));
          },
          py::arg("cls"), py::arg("sizes"), py::arg("dtype"),
          py::arg("context") = py::none())
      .def_classmethod(
          "get_with_least_static_information",
          [](const py::object &cls, MlirContext context) {
            return cls(
                torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
                    context));
          },
          py::arg("cls"), py::arg("context") = py::none())
      .def("sizes",
           [](MlirType self) {
             std::vector<int64_t> sizes(
                 torchMlirTorchNonValueTensorTypeGetRank(self));
             if (torchMlirTorchNonValueTensorTypeGetSizes(self, sizes.data()))
               throw py::value_error("no sizes");
             return py::tuple(py::cast(sizes));
           })
      .def("dtype", [](MlirType self) {
        return torchMlirTorchNonValueTensorTypeGetDtype(self);
      });

  mlir_type_subclass(m, "TorchValueTensorType",
                     torchMlirTypeIsATorchValueTensor)
      .def_classmethod(
          "get",
          [](const py::object &cls, std::vector<int64_t> sizes, MlirType dtype,
             MlirContext context) {
            return cls(torchMlirTorchValueTensorTypeGet(context, sizes.size(),
                                                        sizes.data(), dtype));
          },
          py::arg("cls"), py::arg("sizes"), py::arg("dtype"),
          py::arg("context") = py::none())
      .def("sizes",
           [](MlirType self) {
             std::vector<int64_t> sizes(
                 torchMlirTorchValueTensorTypeGetRank(self));
             if (torchMlirTorchValueTensorTypeGetSizes(self, sizes.data()))
               throw py::value_error("no sizes");
             return py::tuple(py::cast(sizes));
           })
      .def("dtype", [](MlirType self) {
        return torchMlirTorchValueTensorTypeGetDtype(self);
      });

  mlir_type_subclass(m, "TorchDictType", torchMlirTypeIsATorchDict)
      .def_classmethod(
          "get",
          [](py::object cls, MlirType keyType, MlirType valueType) {
            return cls(torchMlirTorchDictTypeGet(keyType, valueType));
          })
      .def("get_key_type", torchMlirTorchDictTypeGetKeyType)
      .def("get_value_type", torchMlirTorchDictTypeGetValueType);

  //============================================================================
  // Values
  //============================================================================

#define DEFINE_SUBCLASS(TTT)                                                   \
  (void)mlir_value_subclass(m, "Torch" #TTT "Value", [](MlirValue value) {     \
    return torchMlirTypeIsATorch##TTT(mlirValueGetType(value));                \
  }).def("__str__", [](const py::object &self) {                               \
    auto Value =                                                               \
        py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Value");     \
    return py::str(Value(self))                                                \
        .attr("replace")("Value", "Torch" #TTT "Value");                       \
  });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_SUBCLASS)
  TORCH_MLIR_FORALL_CONTAINER_TYPES(DEFINE_SUBCLASS)

#undef DEFINE_SUBCLASS
}
