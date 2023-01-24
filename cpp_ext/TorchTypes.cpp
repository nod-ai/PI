//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#include "IRModule.h"

#include "TorchTypes.h"
#include "TorchTypesCAPI.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace mlir;
using namespace mlir::python;

py::object getPyType(MlirType rawType, const PyMlirContextRef &ctx) {
#define DEFINE_CAST_TYPE(TTT)                                                  \
  if (torchMlirTypeIsATorch##TTT(rawType)) {                                   \
    return py::cast<>(Torch_##TTT##Type(ctx, rawType));                        \
  }
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_CAST_TYPE)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_CAST_TYPE)
#undef DEFINE_CAST_TYPE

  if (torchMlirTypeIsATorchList(rawType)) {
    auto elType = torchMlirTorchListTypeGetContainedType(rawType);
#define DEFINE_CAST_LIST_TYPE(TTT)                                             \
  if (torchMlirTypeIsATorch##TTT(elType)) {                                    \
    return py::cast<>(                                                         \
        TorchListOfTorch##TTT##Type::createFromMlirType_(rawType));            \
  }
    TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_CAST_LIST_TYPE)
    TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_CAST_LIST_TYPE)
#undef DEFINE_CAST_LIST_TYPE

#define DEFINE_CAST_LIST_TYPE(TTT)                                             \
  if (torchMlirTypeIsATorch##TTT(elType)) {                                    \
    return py::cast<>(TorchListOf##TTT##Type::createFromMlirType_(rawType));   \
  }
    TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_CAST_LIST_TYPE)
#undef DEFINE_CAST_LIST_TYPE
  }

  if (torchMlirTypeIsATorchOptional(rawType)) {
    auto elType = torchMlirTorchOptionalTypeGetContained(rawType);
#define DEFINE_CAST_OPTIONAL_TYPE(TTT)                                         \
  if (torchMlirTypeIsATorch##TTT(elType)) {                                    \
    return py::cast<>(TorchOptional##TTT##Type::createFromMlirType_(rawType)); \
  }
    TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_CAST_OPTIONAL_TYPE)
    TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_CAST_OPTIONAL_TYPE)
    TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_CAST_OPTIONAL_TYPE)
#undef DEFINE_CAST_OPTIONAL_TYPE
  }

  mlirTypeDump(rawType);
  throw py::type_error("couldn't infer value's type");
}

void bindTypes(py::module &m) {
  py::object type_ =
      (py::object)py::module_::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Type");

  py::object location =
      (py::object)py::module_::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Location");

#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<Torch_##TTT##Type>(m, "Torch_" #TTT "Type", type_)                \
      .def(py::init<>([]() {                                                   \
        auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))         \
                       .attr("Context")                                        \
                       .attr("current");                                       \
        py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                  \
        MlirContext mlirContext = {ctxCapsule.get_pointer()};                  \
        auto torchIntType = torchMlirTorch##TTT##TypeGet(mlirContext);         \
        return Torch_##TTT##Type::createFromMlirType_(torchIntType);           \
      }))                                                                      \
      .def(py::init<>([](const py::handle apiObject) {                         \
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);    \
        return Torch_##TTT##Type::createFromCapsule_(capsule);                 \
      }))                                                                      \
      .def("__repr__",                                                         \
           [](PyType &self) {                                                  \
             PyPrintAccumulator printAccum;                                    \
             printAccum.parts.append("Torch" #TTT "(");                        \
             mlirTypePrint(self, printAccum.getCallback(),                     \
                           printAccum.getUserData());                          \
             printAccum.parts.append(")");                                     \
             return printAccum.join();                                         \
           })                                                                  \
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<Torch_##TTT##Type>(m, "Torch_" #TTT "Type", type_)                \
      .def(py::init<>([](const py::handle optionalSizesHandle,                 \
                         const py::handle optionalDtypeHandle) {               \
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))    \
                            .attr("Context")                                   \
                            .attr("current");                                  \
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);             \
             MlirContext mlirContext = {ctxCapsule.get_pointer()};             \
                                                                               \
             int64_t numSizes = -1;                                            \
             std::vector<int64_t> optionalSizes;                               \
             if (!optionalSizesHandle.is(py::none())) {                        \
               optionalSizes =                                                 \
                   py::cast<std::vector<int64_t>>(optionalSizesHandle);        \
               numSizes = optionalSizes.size();                                \
             }                                                                 \
             MlirType optionalDtype;                                           \
             if (!optionalDtypeHandle.is(py::none())) {                        \
               py::capsule optionalDtypeCapsule =                              \
                   mlirApiObjectToCapsule(optionalDtypeHandle);                \
               optionalDtype = {optionalDtypeCapsule.get_pointer()};           \
             } else {                                                          \
               optionalDtype = {nullptr};                                      \
             }                                                                 \
             auto tensorType = torchMlirTorch##TTT##TypeGet(                   \
                 mlirContext, numSizes, optionalSizes.data(), optionalDtype);  \
             return Torch_##TTT##Type::createFromMlirType_(tensorType);        \
           }),                                                                 \
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())       \
      .def(py::init<>([](const py::handle apiObject) {                         \
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);    \
        return Torch_##TTT##Type::createFromCapsule_(capsule);                 \
      }))                                                                      \
      .def_property_readonly(                                                  \
          "sizes",                                                             \
          [](const py::handle &self) {                                         \
            py::capsule capsule =                                              \
                pybind11::detail::mlirApiObjectToCapsule(self);                \
            MlirType rawType = {capsule.get_pointer()};                        \
            if (torchMlirTorch##TTT##TypeHasSizes(rawType)) {                  \
              auto rank = torchMlirTorch##TTT##TypeGetRank(rawType);           \
              std::vector<int64_t> sizes(rank);                                \
              torchMlirTorch##TTT##TypeGetSizes(rawType, sizes.data());        \
              return sizes;                                                    \
            } else {                                                           \
              std::vector<int64_t> sizes;                                      \
              return sizes;                                                    \
            }                                                                  \
          })                                                                   \
      .def_property_readonly("dtype", [](const py::handle &self) {             \
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(self);  \
        MlirType rawType = {capsule.get_pointer()};                            \
                                                                               \
        auto dtype = torchMlirTorch##TTT##TypeGetDtype(rawType);               \
        return Torch_DType::createFromMlirType_(dtype);                        \
      });

  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<TorchListOfTorch##TTT##Type>(m, "TorchListOfTorch" #TTT "Type",   \
                                          type_)                               \
      .def(py::init<>([]() {                                                   \
        auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))         \
                       .attr("Context")                                        \
                       .attr("current");                                       \
        py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                  \
        MlirContext mlirContext = {ctxCapsule.get_pointer()};                  \
        MlirType elType = torchMlirTorch##TTT##TypeGet(mlirContext);           \
        auto listType = torchMlirTorchListTypeGet(elType);                     \
        return TorchListOfTorch##TTT##Type::createFromMlirType_(listType);     \
      }))                                                                      \
      .def("__repr__",                                                         \
           [](PyType &self) {                                                  \
             PyPrintAccumulator printAccum;                                    \
             printAccum.parts.append("TorchListOfTorch" #TTT "(");             \
             mlirTypePrint(self, printAccum.getCallback(),                     \
                           printAccum.getUserData());                          \
             printAccum.parts.append(")");                                     \
             return printAccum.join();                                         \
           })                                                                  \
      .def_property_readonly(                                                  \
          "el_type",                                                           \
          [](PyType &self) {                                                   \
            return getPyType(                                                  \
                torchMlirTorchListTypeGetContainedType(self.get()),            \
                self.getContext());                                            \
          })                                                                   \
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<TorchListOf##TTT##Type>(m, "TorchListOf" #TTT "Type", type_)      \
      .def(py::init<>([](const py::handle optionalSizesHandle,                 \
                         const py::handle optionalDtypeHandle) {               \
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))    \
                            .attr("Context")                                   \
                            .attr("current");                                  \
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);             \
             MlirContext mlirContext = {ctxCapsule.get_pointer()};             \
                                                                               \
             int64_t numSizes = -1;                                            \
             std::vector<int64_t> optionalSizes;                               \
             if (!optionalSizesHandle.is(py::none())) {                        \
               optionalSizes =                                                 \
                   py::cast<std::vector<int64_t>>(optionalSizesHandle);        \
               numSizes = optionalSizes.size();                                \
             }                                                                 \
             MlirType optionalDtype;                                           \
             if (!optionalDtypeHandle.is(py::none())) {                        \
               py::capsule optionalDtypeCapsule =                              \
                   mlirApiObjectToCapsule(optionalDtypeHandle);                \
               optionalDtype = {optionalDtypeCapsule.get_pointer()};           \
             } else {                                                          \
               optionalDtype = {nullptr};                                      \
             }                                                                 \
             auto tensorType = torchMlirTorch##TTT##TypeGet(                   \
                 mlirContext, numSizes, optionalSizes.data(), optionalDtype);  \
             auto listType = torchMlirTorchListTypeGet(tensorType);            \
             return TorchListOf##TTT##Type::createFromMlirType_(listType);     \
           }),                                                                 \
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())       \
      .def("__repr__",                                                         \
           [](PyType &self) {                                                  \
             PyPrintAccumulator printAccum;                                    \
             printAccum.parts.append("TorchListOf" #TTT "(");                  \
             mlirTypePrint(self, printAccum.getCallback(),                     \
                           printAccum.getUserData());                          \
             printAccum.parts.append(")");                                     \
             return printAccum.join();                                         \
           })                                                                  \
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)
  py::class_<TorchListOfOptionalTensorType>(m, "TorchListOfOptionalTensorType",
                                            type_)
      .def(py::init<>([](const py::handle optionalSizesHandle,
                         const py::handle optionalDtypeHandle) {
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                            .attr("Context")
                            .attr("current");
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);
             MlirContext mlirContext = {ctxCapsule.get_pointer()};

             int64_t numSizes = -1;
             std::vector<int64_t> optionalSizes;
             if (!optionalSizesHandle.is(py::none())) {
               optionalSizes =
                   py::cast<std::vector<int64_t>>(optionalSizesHandle);
               numSizes = optionalSizes.size();
             }
             MlirType optionalDtype;
             if (!optionalDtypeHandle.is(py::none())) {
               py::capsule optionalDtypeCapsule =
                   mlirApiObjectToCapsule(optionalDtypeHandle);
               optionalDtype = {optionalDtypeCapsule.get_pointer()};
             } else {
               optionalDtype = {nullptr};
             }
             auto tensorType = torchMlirTorchNonValueTensorTypeGet(
                 mlirContext, numSizes, optionalSizes.data(), optionalDtype);
             auto optionalType = torchMlirTorchOptionalTypeGet(tensorType);
             auto listType = torchMlirTorchListTypeGet(optionalType);
             return TorchListOfOptionalTensorType::createFromMlirType_(
                 listType);
           }),
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())
      .def("__repr__",
           [](PyType &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("TorchListOfOptionalTensorType");
             mlirTypePrint(self, printAccum.getCallback(),
                           printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<TorchOptional##TTT##Type>(m, "TorchOptional" #TTT "Type", type_)  \
      .def(py::init<>([]() {                                                   \
        auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))         \
                       .attr("Context")                                        \
                       .attr("current");                                       \
        py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                  \
        MlirContext mlirContext = {ctxCapsule.get_pointer()};                  \
        MlirType elType = torchMlirTorch##TTT##TypeGet(mlirContext);           \
        auto optionalType = torchMlirTorchOptionalTypeGet(elType);             \
        return TorchOptional##TTT##Type::createFromMlirType_(optionalType);    \
      }))                                                                      \
      .def("__repr__",                                                         \
           [](PyType &self) {                                                  \
             PyPrintAccumulator printAccum;                                    \
             printAccum.parts.append("TorchOptional" #TTT "(");                \
             mlirTypePrint(self, printAccum.getCallback(),                     \
                           printAccum.getUserData());                          \
             printAccum.parts.append(")");                                     \
             return printAccum.join();                                         \
           })                                                                  \
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<TorchOptional##TTT##Type>(m, "TorchOptional" #TTT "Type", type_)  \
      .def(py::init<>([](const py::handle optionalSizesHandle,                 \
                         const py::handle optionalDtypeHandle) {               \
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))    \
                            .attr("Context")                                   \
                            .attr("current");                                  \
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);             \
             MlirContext mlirContext = {ctxCapsule.get_pointer()};             \
                                                                               \
             int64_t numSizes = -1;                                            \
             std::vector<int64_t> optionalSizes;                               \
             if (!optionalSizesHandle.is(py::none())) {                        \
               optionalSizes =                                                 \
                   py::cast<std::vector<int64_t>>(optionalSizesHandle);        \
               numSizes = optionalSizes.size();                                \
             }                                                                 \
             MlirType optionalDtype;                                           \
             if (!optionalDtypeHandle.is(py::none())) {                        \
               py::capsule optionalDtypeCapsule =                              \
                   mlirApiObjectToCapsule(optionalDtypeHandle);                \
               optionalDtype = {optionalDtypeCapsule.get_pointer()};           \
             } else {                                                          \
               optionalDtype = {nullptr};                                      \
             }                                                                 \
             auto tensorType = torchMlirTorch##TTT##TypeGet(                   \
                 mlirContext, numSizes, optionalSizes.data(), optionalDtype);  \
             auto optionalType = torchMlirTorchOptionalTypeGet(tensorType);    \
             return TorchOptional##TTT##Type::createFromMlirType_(             \
                 optionalType);                                                \
           }),                                                                 \
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())       \
      .def("__repr__",                                                         \
           [](PyType &self) {                                                  \
             PyPrintAccumulator printAccum;                                    \
             printAccum.parts.append("TorchOptional" #TTT "(");                \
             mlirTypePrint(self, printAccum.getCallback(),                     \
                           printAccum.getUserData());                          \
             printAccum.parts.append(")");                                     \
             return printAccum.join();                                         \
           })                                                                  \
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND
}

void bindTypeHelpers(py::module &m) {
  m.def("_torch_list_of_type", [](const py::handle elTypeHandle) {
    py::capsule elTypeCapsule = mlirApiObjectToCapsule(elTypeHandle);
    MlirType elType = {elTypeCapsule.get_pointer()};
    return py::cast<>(torchMlirTorchListTypeGet(elType));
  });

  m.def("_torch_optional_of_type", [](const py::handle elTypeHandle) {
    py::capsule elTypeCapsule = mlirApiObjectToCapsule(elTypeHandle);
    MlirType elType = {elTypeCapsule.get_pointer()};
    return py::cast<>(torchMlirTorchOptionalTypeGet(elType));
  });

  m.def("is_dtype", [](const py::handle apiObject) {
    py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);
    MlirType type = {capsule.get_pointer()};
    return mlirTypeIsAInteger(type) || mlirTypeIsABF16(type) ||
           mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type);
  });
}
