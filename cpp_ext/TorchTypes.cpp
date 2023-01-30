// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#include "IRModule.h"

#include "TorchTypes.h"
#include "TorchTypesCAPI.h"

#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace mlir;
using namespace mlir::python;

py::str repr(MlirType self, const std::string &name) {
  PyPrintAccumulator printAccum;
  printAccum.parts.append(name + "(");
  mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());
  printAccum.parts.append(")");
  return printAccum.join();
}

py::object getPyType(MlirType rawType, const PyMlirContextRef &ctx) {
  // nonvaluetensor
  if (torchMlirTypeIsATorchNonValueTensor(rawType)) {
    return py::cast<>(Torch_NonValueTensorType(ctx, rawType));
  }

#define DEFINE_CAST_TYPE(TTT)                                                  \
  if (torchMlirTypeIsATorch##TTT(rawType)) {                                   \
    return py::cast<>(Torch_##TTT##Type(ctx, rawType));                        \
  }
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_CAST_TYPE)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_CAST_TYPE)
#undef DEFINE_CAST_TYPE

  // list of nonvaluetensor
  if (torchMlirTypeIsATorchList(rawType)) {
    auto elType = torchMlirTorchListTypeGetContainedType(rawType);
    if (torchMlirTypeIsATorchNonValueTensor(elType)) {
      return py::cast<>(
          TorchListOfNonValueTensorType::createFromMlirType(rawType));
    }
    // list of some other type
#define DEFINE_CAST_LIST_TYPE(TTT)                                             \
  if (torchMlirTypeIsATorch##TTT(elType)) {                                    \
    return py::cast<>(                                                         \
        TorchListOfTorch##TTT##Type::createFromMlirType(rawType));             \
  }
    TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_CAST_LIST_TYPE)
    TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_CAST_LIST_TYPE)
#undef DEFINE_CAST_LIST_TYPE

#define DEFINE_CAST_LIST_TYPE(TTT)                                             \
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_CAST_LIST_TYPE)
#undef DEFINE_CAST_LIST_TYPE
  }

  if (torchMlirTypeIsATorchOptional(rawType)) {
    auto elType = torchMlirTorchOptionalTypeGetContained(rawType);
    // optional of nonvaluetensor
    if (torchMlirTypeIsATorchNonValueTensor(elType)) {
      return py::cast<>(
          TorchOptionalNonValueTensorType::createFromMlirType(rawType));
    }
    // optional of other types
#define DEFINE_CAST_OPTIONAL_TYPE(TTT)                                         \
  if (torchMlirTypeIsATorch##TTT(elType)) {                                    \
    return py::cast<>(TorchOptional##TTT##Type::createFromMlirType(rawType));  \
  }
    TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_CAST_OPTIONAL_TYPE)
    TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_CAST_OPTIONAL_TYPE)
#undef DEFINE_CAST_OPTIONAL_TYPE
  }

  throw py::type_error("pi cpp_ext: couldn't infer value's type" +
                       repr(rawType, "").operator std::string());
}

MlirType getNonValueTensorType(const py::handle optionalSizesHandle,
                               const py::handle optionalDtypeHandle,
                               MlirContext mlirContext) {
  int64_t numSizes = -1;
  std::vector<int64_t> optionalSizes;
  if (!optionalSizesHandle.is(py::none())) {
    optionalSizes = py::cast<std::vector<int64_t>>(optionalSizesHandle);
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
  return torchMlirTorchNonValueTensorTypeGet(
      mlirContext, numSizes, optionalSizes.data(), optionalDtype);
}

void bindTensorType(py::module &m, const py::object &PyTypePyClass) {
  py::class_<Torch_NonValueTensorType>(m, "Torch_NonValueTensorType",
                                       PyTypePyClass)
      .def(py::init<>([](const py::handle optionalSizesHandle,
                         const py::handle optionalDtypeHandle) {
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                            .attr("Context")
                            .attr("current");
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);
             MlirContext mlirContext = {ctxCapsule.get_pointer()};
             auto tensorType = getNonValueTensorType(
                 optionalSizesHandle, optionalDtypeHandle, mlirContext);
             return Torch_NonValueTensorType::createFromMlirType(tensorType);
           }),
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())
      .def(py::init<>([](const py::handle apiObject) {
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);
        return Torch_NonValueTensorType::createFromCapsule_(capsule);
      }))
      .def_property_readonly(
          "sizes",
          [](const py::handle &self) {
            py::capsule capsule =
                pybind11::detail::mlirApiObjectToCapsule(self);
            MlirType rawType = {capsule.get_pointer()};
            if (torchMlirTorchNonValueTensorTypeHasSizes(rawType)) {
              auto rank = torchMlirTorchNonValueTensorTypeGetRank(rawType);
              std::vector<int64_t> sizes(rank);
              torchMlirTorchNonValueTensorTypeGetSizes(rawType, sizes.data());
              return sizes;
            } else {
              std::vector<int64_t> sizes;
              return sizes;
            }
          })
      .def_property_readonly("dtype", [](const py::handle &self) {
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(self);
        MlirType rawType = {capsule.get_pointer()};

        auto dtype = torchMlirTorchNonValueTensorTypeGetDtype(rawType);
        return Torch_DType::createFromMlirType(dtype);
      });

  py::class_<TorchListOfNonValueTensorType>(m, "TorchListOfNonValueTensorType",
                                            PyTypePyClass)
      .def(py::init<>([](const py::handle optionalSizesHandle,
                         const py::handle optionalDtypeHandle) {
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                            .attr("Context")
                            .attr("current");
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);
             MlirContext mlirContext = {ctxCapsule.get_pointer()};
             auto tensorType = getNonValueTensorType(
                 optionalSizesHandle, optionalDtypeHandle, mlirContext);
             auto listType = torchMlirTorchListTypeGet(tensorType);
             return TorchListOfNonValueTensorType::createFromMlirType(listType);
           }),
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())
      .def("__repr__",
           [](PyType &self) {
             return repr(self.get(), "TorchListOfNonValueTensorType");
           })
      .def("__str__", [](py::object &self) { return py::repr(self); });

  py::class_<TorchListOfOptionalTensorType>(m, "TorchListOfOptionalTensorType",
                                            PyTypePyClass)
      .def(py::init<>([](const py::handle optionalSizesHandle,
                         const py::handle optionalDtypeHandle) {
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                            .attr("Context")
                            .attr("current");
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);
             MlirContext mlirContext = {ctxCapsule.get_pointer()};
             auto tensorType = getNonValueTensorType(
                 optionalSizesHandle, optionalDtypeHandle, mlirContext);
             auto optionalType = torchMlirTorchOptionalTypeGet(tensorType);
             auto listType = torchMlirTorchListTypeGet(optionalType);
             return TorchListOfOptionalTensorType::createFromMlirType(listType);
           }),
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())
      .def("__repr__",
           [](PyType &self) {
             return repr(self.get(), "TorchListOfOptionalTensorType");
           })
      .def("__str__", [](py::object &self) { return py::repr(self); });

  py::class_<TorchOptionalNonValueTensorType>(
      m, "TorchOptionalNonValueTensorType", PyTypePyClass)
      .def(py::init<>([](const py::handle optionalSizesHandle,
                         const py::handle optionalDtypeHandle) {
             auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                            .attr("Context")
                            .attr("current");
             py::capsule ctxCapsule = mlirApiObjectToCapsule(src);
             MlirContext mlirContext = {ctxCapsule.get_pointer()};
             auto tensorType = getNonValueTensorType(
                 optionalSizesHandle, optionalDtypeHandle, mlirContext);
             auto optionalType = torchMlirTorchOptionalTypeGet(tensorType);
             return TorchOptionalNonValueTensorType::createFromMlirType(
                 optionalType);
           }),
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())
      .def("__repr__",
           [](PyType &self) {
             return repr(self.get(), "TorchOptionalNonValueTensorType");
           })
      .def("__str__", [](py::object &self) { return py::repr(self); });
}

void bindOtherTypes(py::module &m, const py::object &PyTypePyClass) {
#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<Torch_##TTT##Type>(m, "Torch_" #TTT "Type", PyTypePyClass)        \
      .def(py::init<>([]() {                                                   \
        auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))         \
                       .attr("Context")                                        \
                       .attr("current");                                       \
        py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                  \
        MlirContext mlirContext = {ctxCapsule.get_pointer()};                  \
        auto torchIntType = torchMlirTorch##TTT##TypeGet(mlirContext);         \
        return Torch_##TTT##Type::createFromMlirType(torchIntType);            \
      }))                                                                      \
      .def(py::init<>([](const py::handle apiObject) {                         \
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);    \
        return Torch_##TTT##Type::createFromCapsule_(capsule);                 \
      }))                                                                      \
      .def("__repr__",                                                         \
           [](PyType &self) { return repr(self.get(), "Torch" #TTT); })        \
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<TorchListOfTorch##TTT##Type>(m, "TorchListOfTorch" #TTT "Type",   \
                                          PyTypePyClass)                       \
      .def(py::init<>([]() {                                                   \
        auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))         \
                       .attr("Context")                                        \
                       .attr("current");                                       \
        py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                  \
        MlirContext mlirContext = {ctxCapsule.get_pointer()};                  \
        MlirType elType = torchMlirTorch##TTT##TypeGet(mlirContext);           \
        auto listType = torchMlirTorchListTypeGet(elType);                     \
        return TorchListOfTorch##TTT##Type::createFromMlirType(listType);      \
      }))                                                                      \
      .def_property_readonly(                                                  \
          "el_type",                                                           \
          [](PyType &self) {                                                   \
            return getPyType(                                                  \
                torchMlirTorchListTypeGetContainedType(self.get()),            \
                self.getContext());                                            \
          })                                                                   \
      .def("__repr__",                                                         \
           [](PyType &self) { return repr(self.get(), "TorchListOfTorch"); })  \
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                     \
  py::class_<TorchOptional##TTT##Type>(m, "TorchOptional" #TTT "Type",         \
                                       PyTypePyClass)                          \
      .def(py::init<>([]() {                                                   \
        auto src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))         \
                       .attr("Context")                                        \
                       .attr("current");                                       \
        py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                  \
        MlirContext mlirContext = {ctxCapsule.get_pointer()};                  \
        MlirType elType = torchMlirTorch##TTT##TypeGet(mlirContext);           \
        auto optionalType = torchMlirTorchOptionalTypeGet(elType);             \
        return TorchOptional##TTT##Type::createFromMlirType(optionalType);     \
      }))                                                                      \
      .def("__repr__",                                                         \
           [](PyType &self) { return repr(self.get(), "TorchOptional"); })     \
      .def("__str__", [](py::object &self) { return py::repr(self); });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND
}

void bindTypes(py::module &m) {
  py::object PyTypePyClass =
      (py::object)py::module_::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Type");
  bindTensorType(m, PyTypePyClass);
  bindOtherTypes(m, PyTypePyClass);
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
