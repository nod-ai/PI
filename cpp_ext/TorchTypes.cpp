//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "TorchTypes.h"
#include "IRModule.h"
#include "TorchTypesCAPI.h"
#include "mlir-c/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::python;

void bindTypes(py::module &m) {
  py::object type_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("Type");

  py::class_<Torch_IntType>(m, "_Torch_IntType", type_)
      .def(py::init<>([](py::capsule capsule) {
        return Torch_IntType::createFromCapsule_(capsule);
      }));

  py::class_<Torch_BoolType>(m, "_Torch_BoolType", type_)
      .def(py::init<>([](py::capsule capsule) {
        return Torch_BoolType::createFromCapsule_(capsule);
      }));

  py::class_<Torch_StringType>(m, "_Torch_StringType", type_)
      .def(py::init<>([](py::capsule capsule) {
        return Torch_StringType::createFromCapsule_(capsule);
      }));

  py::class_<Torch_FloatType>(m, "_Torch_FloatType", type_)
      .def(py::init<>([](py::capsule capsule) {
        return Torch_FloatType::createFromCapsule_(capsule);
      }));

  py::class_<Torch_ValueTensorType>(m, "_Torch_ValueTensorType", type_)
      .def(py::init<>([](py::capsule capsule) {
        return Torch_ValueTensorType::createFromCapsule_(capsule);
      }));

  py::class_<Torch_NonValueTensorType>(m, "_Torch_NonValueTensorType", type_)
      .def(py::init<>([](py::capsule capsule) {
        return Torch_NonValueTensorType::createFromCapsule_(capsule);
      }));
}

void bindTypeHelpers(py::module &m) {
  m.def(
      "is_a_torch_int_type", [](const py::capsule &type) {
        MlirType rawType = {type.get_pointer()};
        return torchMlirTypeIsATorchInt(rawType);
      });
  m.def(
      "is_a_torch_bool_type", [](const py::capsule &type) {
        MlirType rawType = {type.get_pointer()};
        return torchMlirTypeIsATorchBool(rawType);
      });
  m.def(
      "is_a_torch_string_type", [](const py::capsule &type) {
        MlirType rawType = {type.get_pointer()};
        return torchMlirTypeIsATorchString(rawType);
      });
  m.def(
      "is_a_torch_float_type", [](const py::capsule &type) {
        MlirType rawType = {type.get_pointer()};
        return torchMlirTypeIsATorchFloat(rawType);
      });
  m.def(
      "is_a_torch_value_tensor_type", [](const py::capsule &type) {
        MlirType rawType = {type.get_pointer()};
        return torchMlirTypeIsATorchValueTensor(rawType);
      });
  m.def(
      "is_a_torch_nonvalue_tensor_type", [](const py::capsule &type) {
        MlirType rawType = {type.get_pointer()};
        return torchMlirTypeIsATorchNonValueTensor(rawType);
      });
}
