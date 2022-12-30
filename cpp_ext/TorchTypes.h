//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "IRModule.h"
#include "TorchTypesCAPI.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::python;

struct Torch_IntType : public PyConcreteType<Torch_IntType> {
  Torch_IntType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<Torch_IntType>(std::move(contextRef), t) {}
};

struct Torch_BoolType : public PyConcreteType<Torch_BoolType> {
  Torch_BoolType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<Torch_BoolType>(std::move(contextRef), t) {}
};

struct Torch_StringType : public PyConcreteType<Torch_StringType> {
  Torch_StringType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<Torch_StringType>(std::move(contextRef), t) {}
};

struct Torch_FloatType : public PyConcreteType<Torch_FloatType> {
  Torch_FloatType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<Torch_FloatType>(std::move(contextRef), t) {}
};

struct Torch_ValueTensorType : public PyConcreteType<Torch_ValueTensorType> {
  Torch_ValueTensorType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<Torch_ValueTensorType>(std::move(contextRef), t) {}
};

struct Torch_NonValueTensorType : public PyConcreteType<Torch_NonValueTensorType> {
  Torch_NonValueTensorType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<Torch_NonValueTensorType>(std::move(contextRef), t) {}
};

void bindTypes(py::module &m);
void bindTypeHelpers(py::module &m);