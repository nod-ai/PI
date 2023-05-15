#ifndef PI_TORCHTYPES_H
#define PI_TORCHTYPES_H

// hack
#include "IRModule.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "torch-mlir-c/TorchTypes.h"

namespace py = pybind11;
using namespace mlir::python;

namespace mlir::torch {
bool torchMlirTypeIsATorchBaseTensor(MlirType type);

bool isAAnyTorchDictKeyType(MlirType type);

bool isAAnyTorchListOfOptionalIntType(MlirType type);
bool isAAnyTorchListOfOptionalTensorType(MlirType type);

bool isAAnyTorchListOfTensorType(MlirType type);
bool isAAnyTorchListOfTorchBoolType(MlirType type);
bool isAAnyTorchListOfTorchIntType(MlirType type);
bool isAAnyTorchListOfTorchStringType(MlirType type);
bool isAAnyTorchListType(MlirType type);

bool isAAnyTorchOptionalBoolType(MlirType type);
bool isAAnyTorchOptionalDeviceType(MlirType type);
bool isAAnyTorchOptionalFloatType(MlirType type);
bool isAAnyTorchOptionalGeneratorType(MlirType type);
bool isAAnyTorchOptionalIntType(MlirType type);
bool isAAnyTorchOptionalStringType(MlirType type);
bool isAAnyTorchOptionalTensorType(MlirType type);
bool isAAnyTorchOptionalType(MlirType type);

bool isAAnyTorchOptionalListOfTorchIntType(MlirType type);

bool isAAnyTorchOptionalScalarType(MlirType type);
bool isAAnyTorchScalarType(MlirType type);
bool isAAnyTorchTensorType(MlirType type);
bool isAAnyTorchType(MlirType type);

bool isATorch_BoolType(MlirType type);
bool isATorch_DeviceType(MlirType type);
bool isATorch_DictType(MlirType type);
bool isATorch_FloatType(MlirType type);
bool isATorch_IntType(MlirType type);
bool isATorch_LinearParamsType(MlirType type);
bool isATorch_NnModuleType(MlirType type);
bool isATorch_NonValueTensorType(MlirType type);
bool isATorch_NoneType(MlirType type);
bool isATorch_NumberType(MlirType type);
bool isATorch_StringType(MlirType type);
bool isATorch_TupleType(MlirType type);
bool isATorch_ValueTensorType(MlirType type);

class PyAnyTorchListType : public PyConcreteType<PyAnyTorchListType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchListType;
  static constexpr const char *pyClassName = "AnyTorchListType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchListType(MlirType containedType, DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchListTypeGet(containedType)) {}

  static void bindDerived(ClassTy &containedType);
};

class PyAnyTorchOptionalType : public PyConcreteType<PyAnyTorchOptionalType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalType;
  static constexpr const char *pyClassName = "AnyTorchOptionalType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchOptionalType(MlirType containedType,
                         DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchOptionalTypeGet(containedType)) {}

  static void bindDerived(ClassTy &c);
};

#define FORALL_LIST_BASE_CONCRETE_TYPES(_)                                     \
  _(TorchBool)                                                                 \
  _(TorchInt)                                                                  \
  _(TorchString)

#define FORALL_OPTIONAL_BASE_CONCRETE_TYPES(_)                                 \
  _(Bool)                                                                      \
  _(Device)                                                                    \
  _(Float)                                                                     \
  _(Generator)                                                                 \
  _(Int)                                                                       \
  _(String)

#define DECLARE_LIST_BASE_CONCRETE_TYPE(CONCRETETYPE)                          \
  class PyAnyTorchListOf##CONCRETETYPE##Type                                   \
      : public PyConcreteType<PyAnyTorchListOf##CONCRETETYPE##Type,            \
                              PyAnyTorchListType> {                            \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction =                               \
        isAAnyTorchListOf##CONCRETETYPE##Type;                                 \
    static constexpr const char *pyClassName =                                 \
        "AnyTorchListOf" #CONCRETETYPE "Type";                                 \
    using PyConcreteType::PyConcreteType;                                      \
    PyAnyTorchListOf##CONCRETETYPE##Type(DefaultingPyMlirContext context)      \
        : PyConcreteType(                                                      \
              context->getRef(),                                               \
              torchMlirTorchListTypeGet(                                       \
                  torchMlir##CONCRETETYPE##TypeGet(context->get()))) {}        \
                                                                               \
    static void bindDerived(ClassTy &c);                                       \
  };
FORALL_LIST_BASE_CONCRETE_TYPES(DECLARE_LIST_BASE_CONCRETE_TYPE)
#undef DECLARE_LIST_BASE_CONCRETE_TYPE

#define DECLARE_OPTIONAL_BASE_CONCRETE_TYPE(CONCRETETYPE)                      \
  class PyAnyTorchOptional##CONCRETETYPE##Type                                 \
      : public PyConcreteType<PyAnyTorchOptional##CONCRETETYPE##Type,          \
                              PyAnyTorchOptionalType> {                        \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction =                               \
        isAAnyTorchOptional##CONCRETETYPE##Type;                               \
    static constexpr const char *pyClassName =                                 \
        "AnyTorchOptional" #CONCRETETYPE "Type";                               \
    using PyConcreteType::PyConcreteType;                                      \
    static void bindDerived(ClassTy &c);                                       \
    PyAnyTorchOptional##CONCRETETYPE##Type(DefaultingPyMlirContext context)    \
        : PyConcreteType(                                                      \
              context->getRef(),                                               \
              torchMlirTorchOptionalTypeGet(                                   \
                  torchMlirTorch##CONCRETETYPE##TypeGet(context->get()))) {}   \
  };
FORALL_OPTIONAL_BASE_CONCRETE_TYPES(DECLARE_OPTIONAL_BASE_CONCRETE_TYPE)
#undef DECLARE_OPTIONAL_BASE_CONCRETE_TYPE

#define FORALL_SCALAR_TYPES(_)                                                 \
  _(Bool)                                                                      \
  _(Device)                                                                    \
  _(Float)                                                                     \
  _(Int)                                                                       \
  _(LinearParams)                                                              \
  _(None)                                                                      \
  _(Number)                                                                    \
  _(String)

#define DECLARE_SCALAR_TYPE(SCALARTYPE)                                        \
  class PyTorch_##SCALARTYPE##Type                                             \
      : public PyConcreteType<PyTorch_##SCALARTYPE##Type> {                    \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction = isATorch_##SCALARTYPE##Type;  \
    static constexpr const char *pyClassName = "Torch_" #SCALARTYPE "Type";    \
    using PyConcreteType::PyConcreteType;                                      \
    static void bindDerived(ClassTy &c);                                       \
    PyTorch_##SCALARTYPE##Type(DefaultingPyMlirContext context)                \
        : PyConcreteType(                                                      \
              context->getRef(),                                               \
              torchMlirTorch##SCALARTYPE##TypeGet(context->get())) {}          \
  };
FORALL_SCALAR_TYPES(DECLARE_SCALAR_TYPE)
#undef DECLARE_SCALAR_TYPE

class PyTorch_DictType : public PyConcreteType<PyTorch_DictType> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_DictType;
  static constexpr const char *pyClassName = "Torch_DictType";
  using PyConcreteType::PyConcreteType;
  PyTorch_DictType(MlirType keyType, MlirType valueType,
                   DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchDictTypeGetChecked(context->get(), keyType,
                                                        valueType)) {}
  static void bindDerived(ClassTy &c);
};

class PyTorch_TupleType : public PyConcreteType<PyTorch_TupleType> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_TupleType;
  static constexpr const char *pyClassName = "Torch_TupleType";
  using PyConcreteType::PyConcreteType;
  PyTorch_TupleType(std::vector<MlirType> elementTypes,
                    DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchTupleTypeGet(context->get(),
                                                  elementTypes.size(),
                                                  elementTypes.data())) {}

  static void bindDerived(ClassTy &c);
};

class PyTorch_NnModuleType : public PyConcreteType<PyTorch_NnModuleType> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_NnModuleType;
  static constexpr const char *pyClassName = "Torch_NnModuleType";
  using PyConcreteType::PyConcreteType;
  PyTorch_NnModuleType(MlirStringRef name, DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchNnModuleTypeGet(context->get(), name)) {}
  static void bindDerived(ClassTy &c);
};

class PyTorch_NonValueTensorType
    : public PyConcreteType<PyTorch_NonValueTensorType> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_NonValueTensorType;
  static constexpr const char *pyClassName = "Torch_NonValueTensorType";
  using PyConcreteType::PyConcreteType;
  PyTorch_NonValueTensorType(std::vector<int64_t> sizes, MlirType dtype,
                             DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchNonValueTensorTypeGet(
                           context->get(), sizes.size(), sizes.data(), dtype)) {
  }
  static PyTorch_NonValueTensorType
  getWithLeastStaticInformation(DefaultingPyMlirContext context);

  static void bindDerived(ClassTy &c);
};

class PyTorch_ValueTensorType : public PyConcreteType<PyTorch_ValueTensorType> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_ValueTensorType;
  static constexpr const char *pyClassName = "Torch_ValueTensorType";
  using PyConcreteType::PyConcreteType;
  PyTorch_ValueTensorType(std::vector<int64_t> sizes, MlirType dtype,
                          DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchValueTensorTypeGet(
                           context->get(), sizes.size(), sizes.data(), dtype)) {
  }
  static PyTorch_ValueTensorType
  getWithLeastStaticInformation(DefaultingPyMlirContext context);

  static void bindDerived(ClassTy &c);
};

class PyAnyTorchTensorType
    : public PyConcreteType<PyAnyTorchTensorType, PyTorch_NonValueTensorType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchTensorType;
  static constexpr const char *pyClassName = "AnyTorchTensorType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchTensorType(std::vector<int64_t> sizes, MlirType dtype,
                       DefaultingPyMlirContext context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchNonValueTensorTypeGet(
                           context->get(), sizes.size(), sizes.data(), dtype)) {
  }
  static PyAnyTorchTensorType
  getWithLeastStaticInformation(DefaultingPyMlirContext context);

  static void bindDerived(ClassTy &c);
};

void populateTorchMLIRTypes(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHTYPES_H
