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


struct Torch_NonValueTensorType : public PyConcreteType<Torch_NonValueTensorType> {
  Torch_NonValueTensorType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<Torch_NonValueTensorType>(std::move(contextRef), t) {}
};

struct TorchListOfNonValueTensorType
    : public PyConcreteType<TorchListOfNonValueTensorType> {
  TorchListOfNonValueTensorType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<TorchListOfNonValueTensorType>(std::move(contextRef), t) {}
};

struct TorchOptionalNonValueTensorType
    : public PyConcreteType<TorchOptionalNonValueTensorType> {
  TorchOptionalNonValueTensorType(PyMlirContextRef contextRef, MlirType t)
      : PyConcreteType<TorchOptionalNonValueTensorType>(std::move(contextRef), t) {
  }
};


// Note: TorchScript does not consider !torch.bool to be a Scalar.
#define TORCH_MLIR_FORALL_NUMBER_TYPES(_)                                      \
  _(Float)                                                                     \
  _(Int)                                                                       \
  _(Number)                                                                    \
  _(QInt8)                                                                     \
  _(QUInt8)

#define TORCH_MLIR_FORALL_CONTAINER_TYPES(_)                                   \
  _(Dict)                                                                      \
  _(List)                                                                      \
  _(Tuple)                                                                     \
  _(Optional)

#define TORCH_MLIR_FORALL_OTHER_TYPES(_)                                       \
  _(Any)                                                                       \
  _(Bool)                                                                      \
  _(Device)                                                                    \
  _(Generator)                                                                 \
  _(LinearParams)                                                              \
  _(None)                                                                      \
  _(String)
//  _(NnModule)                            \
//  _(Union)

#define DEFINE_STRUCT(TTT)                                                     \
  struct Torch_##TTT##Type : public PyConcreteType<Torch_##TTT##Type> {        \
    Torch_##TTT##Type(PyMlirContextRef contextRef, MlirType t)                 \
        : PyConcreteType<Torch_##TTT##Type>(std::move(contextRef), t) {}       \
  };
TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_STRUCT)
TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_STRUCT)
DEFINE_STRUCT(D)
#undef DEFINE_STRUCT

#define DEFINE_STRUCT(TTT)                                                     \
  struct TorchListOfTorch##TTT##Type                                           \
      : public PyConcreteType<TorchListOfTorch##TTT##Type> {                   \
    TorchListOfTorch##TTT##Type(PyMlirContextRef contextRef, MlirType t)       \
        : PyConcreteType<TorchListOfTorch##TTT##Type>(std::move(contextRef),   \
                                                      t) {}                    \
  };
TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_STRUCT)
TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_STRUCT)
#undef DEFINE_STRUCT

#define DEFINE_STRUCT(TTT)                                                     \
  struct TorchListOf##TTT##Type                                                \
      : public PyConcreteType<TorchListOf##TTT##Type> {                        \
    TorchListOf##TTT##Type(PyMlirContextRef contextRef, MlirType t)            \
        : PyConcreteType<TorchListOf##TTT##Type>(std::move(contextRef), t) {}  \
  };
DEFINE_STRUCT(OptionalTensor)
#undef DEFINE_STRUCT

#define DEFINE_STRUCT(TTT)                                                     \
  struct TorchOptional##TTT##Type                                              \
      : public PyConcreteType<TorchOptional##TTT##Type> {                      \
    TorchOptional##TTT##Type(PyMlirContextRef contextRef, MlirType t)          \
        : PyConcreteType<TorchOptional##TTT##Type>(std::move(contextRef), t) { \
    }                                                                          \
  };
TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_STRUCT)
TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_STRUCT)
#undef DEFINE_STRUCT

void bindTypes(py::module &m);
void bindTypeHelpers(py::module &typeHandle);
py::object getPyType(MlirType rawType, const PyMlirContextRef &ctx);