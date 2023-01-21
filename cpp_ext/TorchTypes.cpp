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

void bindTypes(py::module &m) {
  py::object type_ =
      (py::object) py::module_::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Type");

  py::object location =
      (py::object) py::module_::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Location");

#define DEFINE_PYBIND(TTT) py::class_<Torch_##TTT##Type>(m, "_Torch_" #TTT "Type", type_)                 \
                               .def(py::init<>([]() {                                                     \
                                 auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))           \
                                                .attr("Context")                                          \
                                                .attr("current");                                         \
                                 py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                    \
                                 MlirContext mlirContext = {ctxCapsule.get_pointer()};                    \
                                 auto torchIntType = torchMlirTorch##TTT##TypeGet(mlirContext);           \
                                 return Torch_##TTT##Type::createFromMlirType_(torchIntType);             \
                               }))                                                                        \
                               .def(py::init<>([](const py::handle apiObject) {                           \
                                 auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);      \
                                 return Torch_##TTT##Type::createFromCapsule_(capsule);                   \
                               }))                                                                        \
                               .def("__repr__", [](PyType &self) {                                        \
                                 PyPrintAccumulator printAccum;                                           \
                                 printAccum.parts.append("Torch" #TTT "(");                               \
                                 mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData()); \
                                 printAccum.parts.append(")");                                            \
                                 return printAccum.join();                                                \
                               })                                                                         \
                               .def("__str__", [](PyType &self) {                                         \
                                 PyPrintAccumulator printAccum;                                           \
                                 printAccum.parts.append("Torch" #TTT "(");                               \
                                 mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData()); \
                                 printAccum.parts.append(")");                                            \
                                 return printAccum.join();                                                \
                               });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                                                               \
  py::class_<Torch_##TTT##Type>(m, "_Torch_" #TTT "Type", type_)                                                         \
      .def(py::init<>([](const py::handle optionalSizesHandle, const py::handle optionalDtypeHandle) {                   \
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))                                              \
                            .attr("Context")                                                                             \
                            .attr("current");                                                                            \
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                                                       \
             MlirContext mlirContext = {ctxCapsule.get_pointer()};                                                       \
                                                                                                                         \
             int64_t numSizes = -1;                                                                                      \
             std::vector<int64_t> optionalSizes;                                                                         \
             if (!optionalSizesHandle.is(py::none())) {                                                                  \
               optionalSizes = py::cast<std::vector<int64_t>>(optionalSizesHandle);                                      \
               numSizes = optionalSizes.size();                                                                          \
             }                                                                                                           \
             MlirType optionalDtype;                                                                                     \
             if (!optionalDtypeHandle.is(py::none())) {                                                                  \
               py::capsule optionalDtypeCapsule = mlirApiObjectToCapsule(optionalDtypeHandle);                           \
               optionalDtype = {optionalDtypeCapsule.get_pointer()};                                                     \
             } else {                                                                                                    \
               optionalDtype = {nullptr};                                                                                \
             }                                                                                                           \
             auto tensorType = torchMlirTorch##TTT##TypeGet(mlirContext, numSizes, optionalSizes.data(), optionalDtype); \
             return Torch_##TTT##Type::createFromMlirType_(tensorType);                                                  \
           }),                                                                                                           \
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())                                                 \
      .def(py::init<>([](const py::handle apiObject) {                           \
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);      \
        return Torch_##TTT##Type::createFromCapsule_(capsule); }))                                                               \
      .def_property_readonly("sizes", [](const py::handle &self) {                                                       \
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(self);                                            \
        MlirType rawType = {capsule.get_pointer()};                                                                      \
        auto rank = torchMlirTorch##TTT##TypeGetRank(rawType);                                                           \
        const int64_t *sizes_ptr = torchMlirTorch##TTT##TypeGetSizes(rawType);                                           \
        std::vector<int64_t> sizes(sizes_ptr, sizes_ptr + rank);                                                         \
        return sizes;                                                                                                    \
      })                                                                                                                 \
      .def_property_readonly("dtype", [](const py::handle &self) {                                                       \
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(self);                                            \
        MlirType rawType = {capsule.get_pointer()};                                                                      \
                                                                                                                         \
        auto dtype = torchMlirTorch##TTT##TypeGetDtype(rawType);                                                         \
        return Torch_DType::createFromMlirType_(dtype);                                                                  \
      })                                                                                                                 \
      .def("__repr__", [](PyType &self) {                                                                                \
        PyPrintAccumulator printAccum;                                                                                   \
        mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());                                         \
        printAccum.parts.append(")");                                                                                    \
        return printAccum.join();                                                                                        \
      });

  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                                                               \
  py::class_<TorchListOf##TTT##Type>(m, "_TorchListOf" #TTT "Type", type_)                                               \
      .def(py::init<>([](const py::handle optionalSizesHandle, const py::handle optionalDtypeHandle) {                   \
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))                                              \
                            .attr("Context")                                                                             \
                            .attr("current");                                                                            \
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                                                       \
             MlirContext mlirContext = {ctxCapsule.get_pointer()};                                                       \
                                                                                                                         \
             int64_t numSizes = -1;                                                                                      \
             std::vector<int64_t> optionalSizes;                                                                         \
             if (!optionalSizesHandle.is(py::none())) {                                                                  \
               optionalSizes = py::cast<std::vector<int64_t>>(optionalSizesHandle);                                      \
               numSizes = optionalSizes.size();                                                                          \
             }                                                                                                           \
             MlirType optionalDtype;                                                                                     \
             if (!optionalDtypeHandle.is(py::none())) {                                                                  \
               py::capsule optionalDtypeCapsule = mlirApiObjectToCapsule(optionalDtypeHandle);                           \
               optionalDtype = {optionalDtypeCapsule.get_pointer()};                                                     \
             } else {                                                                                                    \
               optionalDtype = {nullptr};                                                                                \
             }                                                                                                           \
             auto tensorType = torchMlirTorch##TTT##TypeGet(mlirContext, numSizes, optionalSizes.data(), optionalDtype); \
             auto listType = torchMlirTorchListTypeGet(tensorType);                                                      \
             return TorchListOf##TTT##Type::createFromMlirType_(listType);                                               \
           }),                                                                                                           \
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())                                                 \
      .def("__repr__", [](PyType &self) {                                                                                \
        PyPrintAccumulator printAccum;                                                                                   \
        printAccum.parts.append("TorchListOf" #TTT "(");                                                                 \
        mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());                                         \
        printAccum.parts.append(")");                                                                                    \
        return printAccum.join();                                                                                        \
      })                                                                                                                 \
      .def("__str__", [](PyType &self) {                                                                                 \
        PyPrintAccumulator printAccum;                                                                                   \
        printAccum.parts.append("TorchListOf" #TTT "(");                                                                 \
        mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());                                         \
        printAccum.parts.append(")");                                                                                    \
        return printAccum.join();                                                                                        \
      });
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND
}

void bindTypeHelpers(py::module &m) {
  py::object type_ =
      (py::object) py::module_::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Type");

#define DEFINE_PYBIND(TTT)                                                           \
  py::class_<TorchListOfTorch##TTT##Type>(m, "_TorchListOfTorch" #TTT "Type", type_) \
      .def(py::init<>([]() {                                                         \
        auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))               \
                       .attr("Context")                                              \
                       .attr("current");                                             \
        py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                        \
        MlirContext mlirContext = {ctxCapsule.get_pointer()};                        \
        MlirType elType = torchMlirTorch##TTT##TypeGet(mlirContext);                 \
        auto listType = torchMlirTorchListTypeGet(elType);                           \
        return TorchListOfTorch##TTT##Type::createFromMlirType_(listType);           \
      }))                                                                            \
      .def("__repr__", [](PyType &self) {                                            \
        PyPrintAccumulator printAccum;                                               \
        printAccum.parts.append("TorchListOf" #TTT "(");                             \
        mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());     \
        printAccum.parts.append(")");                                                \
        return printAccum.join();                                                    \
      })                                                                             \
      .def("__str__", [](PyType &self) {                                             \
        PyPrintAccumulator printAccum;                                               \
        printAccum.parts.append("TorchListOf" #TTT "(");                             \
        mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());     \
        printAccum.parts.append(")");                                                \
        return printAccum.join();                                                    \
      });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

  m.def("_torch_list_of_type", [](const py::handle elTypeHandle) {
    py::capsule elTypeCapsule = mlirApiObjectToCapsule(elTypeHandle);
    MlirType elType = {elTypeCapsule.get_pointer()};
    return py::cast<>(torchMlirTorchListTypeGet(elType));
  });

#define DEFINE_PYBIND(TTT) m.def(                                                \
    "is_a_Torch_" #TTT "Type", [](const py::handle apiObject) {                  \
      py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject); \
      MlirType rawType = {capsule.get_pointer()};                                \
      return torchMlirTypeIsATorch##TTT(rawType);                                \
    });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_CONTAINER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT) m.def(                                                     \
    "is_a_TorchListOfTorch" #TTT "Type", [](const py::handle typeHandle) {            \
      py::capsule typeCapsule = pybind11::detail::mlirApiObjectToCapsule(typeHandle); \
      MlirType type = {typeCapsule.get_pointer()};                                    \
      if (torchMlirTypeIsATorchList(type)) {                                          \
        auto elType = torchMlirTorchListTypeGetContainedType(type);                   \
        return torchMlirTypeIsATorch##TTT(elType);                                    \
      }                                                                               \
      return false;                                                                   \
    });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT) m.def(                                                     \
    "is_a_TorchListOf" #TTT "Type", [](const py::handle typeHandle) {                 \
      py::capsule typeCapsule = pybind11::detail::mlirApiObjectToCapsule(typeHandle); \
      MlirType type = {typeCapsule.get_pointer()};                                    \
      if (torchMlirTypeIsATorchList(type)) {                                          \
        auto elType = torchMlirTorchListTypeGetContainedType(type);                   \
        return torchMlirTypeIsATorch##TTT(elType);                                    \
      }                                                                               \
      return false;                                                                   \
    });
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT) m.def(                                                     \
    "is_a_Torch_" #TTT "Type", [](const py::handle typeHandle) {                      \
      py::capsule typeCapsule = pybind11::detail::mlirApiObjectToCapsule(typeHandle); \
      MlirType type = {typeCapsule.get_pointer()};                                    \
      return torchMlirTypeIsATorch##TTT(type);                                        \
    });
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

  m.def(
      "is_a_TorchScalarType", [](const py::handle apiObject) {
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);
        MlirType rawType = {capsule.get_pointer()};
        auto isScalar = (
#define OR(TTT) torchMlirTypeIsATorch##TTT(rawType) ||
            TORCH_MLIR_FORALL_NUMBER_TYPES(OR) false
#undef OR
        );
        return isScalar;
      });
  m.def(
      "is_a_TorchType", [](const py::handle apiObject) {
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);
        MlirType rawType = {capsule.get_pointer()};
        auto isScalar = (
#define OR(TTT) torchMlirTypeIsATorch##TTT(rawType) ||
            TORCH_MLIR_FORALL_NUMBER_TYPES(OR)
                TORCH_MLIR_FORALL_CONTAINER_TYPES(OR)
                    TORCH_MLIR_FORALL_TENSOR_TYPES(OR)
                        TORCH_MLIR_FORALL_OTHER_TYPES(OR) false
#undef OR
        );
        return isScalar;
      });

  m.def("is_dtype", [](const py::handle apiObject) {
    py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);
    MlirType type = {capsule.get_pointer()};
    return mlirTypeIsAInteger(type) || mlirTypeIsABF16(type) || mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type);
  });
}
