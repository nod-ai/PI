#ifndef PI_TORCHTYPES_H
#define PI_TORCHTYPES_H

#include <regex>
#include <string>

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

bool isAAnyTorchListOfTorchBoolType(MlirType type);
bool isAAnyTorchListOfTorchIntType(MlirType type);
bool isAAnyTorchListOfTorchFloatType(MlirType type);
bool isAAnyTorchListOfTorchStringType(MlirType type);
bool isAAnyTorchListType(MlirType type);

bool isAAnyTorchOptionalBoolType(MlirType type);
bool isAAnyTorchOptionalDeviceType(MlirType type);
bool isAAnyTorchOptionalFloatType(MlirType type);
bool isAAnyTorchOptionalGeneratorType(MlirType type);
bool isAAnyTorchOptionalIntType(MlirType type);
bool isAAnyTorchOptionalStringType(MlirType type);
bool isAAnyTorchOptionalType(MlirType type);

bool isAAnyTorchOptionalListOfTorchIntType(MlirType type);

bool isAAnyTorchOptionalScalarType(MlirType type);
bool isAAnyTorchScalarType(MlirType type);
bool isAAnyTorchType(MlirType type);

#define FORALL_UNDERSCORE_TYPES(_)                                             \
  _(Any)                                                                       \
  _(Bool)                                                                      \
  _(Device)                                                                    \
  _(Dict)                                                                      \
  _(Float)                                                                     \
  _(Int)                                                                       \
  _(LinearParams)                                                              \
  _(NnModule)                                                                  \
  _(None)                                                                      \
  _(Number)                                                                    \
  _(String)                                                                    \
  _(Tuple)
#define DECLARE_ISA_UNDERSCORE_TYPE(UNDERSCORETYPE)                            \
  bool isATorch_##UNDERSCORETYPE##Type(MlirType type);
FORALL_UNDERSCORE_TYPES(DECLARE_ISA_UNDERSCORE_TYPE)
#undef DECLARE_ISA_UNDERSCORE_TYPE

class PyAnyTorchListType : public PyConcreteType<PyAnyTorchListType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchListType;
  static constexpr const char *pyClassName = "AnyTorchListType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchListType(MlirType containedType, PyMlirContext *context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchListTypeGet(containedType)) {}

  static void bindDerived(ClassTy &containedType);
};

class PyAnyTorchOptionalType : public PyConcreteType<PyAnyTorchOptionalType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalType;
  static constexpr const char *pyClassName = "AnyTorchOptionalType";
  using PyConcreteType::PyConcreteType;
  PyAnyTorchOptionalType(MlirType containedType, PyMlirContext *context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchOptionalTypeGet(containedType)) {}

  static void bindDerived(ClassTy &c);
};

#define FORALL_LIST_BASE_CONCRETE_TYPES(_)                                     \
  _(TorchBool)                                                                 \
  _(TorchFloat)                                                                \
  _(TorchInt)                                                                  \
  _(TorchString)

#define FORALL_LIST_BASE_CONCRETE_TYPES_WITH_TYPE(_)                           \
  _(TorchBool, Bool)                                                           \
  _(TorchFloat, Float)                                                         \
  _(TorchInt, Int)                                                             \
  _(TorchString, String)

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
    PyAnyTorchListOf##CONCRETETYPE##Type(PyMlirContext *context)               \
        : PyConcreteType(                                                      \
              context->getRef(),                                               \
              torchMlirTorchListTypeGet(                                       \
                  torchMlir##CONCRETETYPE##TypeGet(context->get()))) {}        \
                                                                               \
    static void bindDerived(ClassTy &c);                                       \
  };
FORALL_LIST_BASE_CONCRETE_TYPES(DECLARE_LIST_BASE_CONCRETE_TYPE)
#undef DECLARE_LIST_BASE_CONCRETE_TYPE

#define FORALL_OPTIONAL_BASE_CONCRETE_TYPES(_)                                 \
  _(Bool)                                                                      \
  _(Device)                                                                    \
  _(Float)                                                                     \
  _(Generator)                                                                 \
  _(Int)                                                                       \
  _(String)

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
    PyAnyTorchOptional##CONCRETETYPE##Type(PyMlirContext *context)             \
        : PyConcreteType(                                                      \
              context->getRef(),                                               \
              torchMlirTorchOptionalTypeGet(                                   \
                  torchMlirTorch##CONCRETETYPE##TypeGet(context->get()))) {}   \
  };
FORALL_OPTIONAL_BASE_CONCRETE_TYPES(DECLARE_OPTIONAL_BASE_CONCRETE_TYPE)
#undef DECLARE_OPTIONAL_BASE_CONCRETE_TYPE

#define FORALL_SCALAR_TYPES(_)                                                 \
  _(Any)                                                                       \
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
    PyTorch_##SCALARTYPE##Type(PyMlirContext *context)                         \
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
  PyTorch_DictType(MlirType keyType, MlirType valueType, PyMlirContext *context)
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
  PyTorch_TupleType(std::vector<MlirType> elementTypes, PyMlirContext *context)
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
  PyTorch_NnModuleType(MlirStringRef name, PyMlirContext *context)
      : PyConcreteType(context->getRef(),
                       torchMlirTorchNnModuleTypeGet(context->get(), name)) {}
  static void bindDerived(ClassTy &c);
};

class PyAnyTorchScalarType : public PyConcreteType<PyAnyTorchScalarType> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchScalarType;
  static constexpr const char *pyClassName = "AnyTorchScalarType";
  using PyConcreteType::PyConcreteType;
  static void bindDerived(ClassTy &c) {
    pybind11::implicitly_convertible<PyType, PyAnyTorchScalarType>();
    c.def("__repr__", [](PyAnyTorchScalarType &self) {
      auto origRepr =
          pybind11::repr(pybind11::cast(PyType(self))).cast<std::string>();
      return std::regex_replace(origRepr, std::regex("Type"),
                                "AnyTorchScalarType");
    });
  };
};

void populateTorchMLIRTypes(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHTYPES_H
