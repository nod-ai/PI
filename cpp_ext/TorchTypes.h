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

// bool isAAnyTorchOptionalScalarType(MlirType type) ;
// bool isAAnyTorchScalarType(MlirType type) ;
bool isAAnyTorchTensorType(MlirType type);
// bool isAAnyTorchType(MlirType type) ;

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

  static void bindDerived(ClassTy &element_type);
};

class PyAnyTorchOptionalType : public PyConcreteType<PyAnyTorchOptionalType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalType;
  static constexpr const char *pyClassName = "AnyTorchOptionalType";
  using PyConcreteType::PyConcreteType;

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
  };
FORALL_OPTIONAL_BASE_CONCRETE_TYPES(DECLARE_OPTIONAL_BASE_CONCRETE_TYPE)
#undef DECLARE_OPTIONAL_BASE_CONCRETE_TYPE

#define FORALL_CONCRETE_TYPES(_)                                               \
  _(_Bool)                                                                     \
  _(_Device)                                                                   \
  _(_Dict)                                                                     \
  _(_Float)                                                                    \
  _(_Int)                                                                      \
  _(_LinearParams)                                                             \
  _(_NnModule)                                                                 \
  _(_NonValueTensor)                                                           \
  _(_None)                                                                     \
  _(_Number)                                                                   \
  _(_String)                                                                   \
  _(_Tuple)                                                                    \
  _(_ValueTensor)

#define DECLARE_CONCRETE_TYPE(CONCRETETYPE)                                    \
  class PyTorch##CONCRETETYPE##Type                                            \
      : public PyConcreteType<PyTorch##CONCRETETYPE##Type> {                   \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction = isATorch##CONCRETETYPE##Type; \
    static constexpr const char *pyClassName = "Torch" #CONCRETETYPE "Type";   \
    using PyConcreteType::PyConcreteType;                                      \
    static void bindDerived(ClassTy &c);                                       \
  };
FORALL_CONCRETE_TYPES(DECLARE_CONCRETE_TYPE)
#undef DECLARE_CONCRETE_TYPE

void populateTorchMLIRTypes(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHTYPES_H
