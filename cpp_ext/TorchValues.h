//
// Created by maksim on 5/13/23.
//

#ifndef PI_TORCHVALUES_H
#define PI_TORCHVALUES_H

#include "Globals.h"
#include "IRModule.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "torch-mlir-c/TorchTypes.h"

#include <functional>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "TorchDType.h"
#include "TorchTypes.h"

using llvm::Twine;
namespace py = pybind11;
using namespace mlir::python;

namespace mlir::torch {

bool isAAnyTorchDictKeyValue(MlirValue value);

bool isAAnyTorchListOfOptionalIntValue(MlirValue value);

bool isAAnyTorchListOfTorchBoolValue(MlirValue value);
bool isAAnyTorchListOfTorchIntValue(MlirValue value);
bool isAAnyTorchListOfTorchFloatValue(MlirValue value);
bool isAAnyTorchListOfTorchStringValue(MlirValue value);
bool isAAnyTorchListValue(MlirValue value);

bool isAAnyTorchOptionalBoolValue(MlirValue value);
bool isAAnyTorchOptionalDeviceValue(MlirValue value);
bool isAAnyTorchOptionalFloatValue(MlirValue value);
bool isAAnyTorchOptionalGeneratorValue(MlirValue value);
bool isAAnyTorchOptionalIntValue(MlirValue value);
bool isAAnyTorchOptionalStringValue(MlirValue value);
bool isAAnyTorchOptionalValue(MlirValue value);

bool isAAnyTorchOptionalListOfTorchIntValue(MlirValue value);

bool isAAnyTorchOptionalScalarValue(MlirValue value);
bool isAAnyTorchScalarValue(MlirValue value);
bool isAAnyTorchValue(MlirValue value);

std::vector<PyType> inferReturnTypes(
    const std::string &operationName,
    const std::vector<std::reference_wrapper<const PyValue>> &operands,
    PyMlirContext *pyContext, PyLocation *loc,
    const std::optional<PyAttribute> &attributes = {},
    void *properties = nullptr);

PyInsertionPoint *getInsertionPoint(const py::object &maybeIp = py::none());

PyOperationRef createOperation(
    const std::string &name,
    const std::vector<std::reference_wrapper<const PyType>> &results,
    const std::vector<std::reference_wrapper<const PyValue>> &operands,
    const std::map<std::string, MlirAttribute> &attributes, PyLocation *loc,
    PyInsertionPoint *ip);

PyLocation getValueLocation(const PyValue &value);

#define DECLARE_ISA_UNDERSCORE_VALUE(UNDERSCOREVALUE)                          \
  bool isATorch_##UNDERSCOREVALUE##Value(MlirValue value);
FORALL_UNDERSCORE_TYPES(DECLARE_ISA_UNDERSCORE_VALUE)
#undef DECLARE_ISA_UNDERSCORE_VALUE

template <typename DerivedTy, typename BaseTy = PyValue>
class PyConcreteValue : public BaseTy {
public:
  using ClassTy = py::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(MlirValue);

  PyConcreteValue() = default;
  PyConcreteValue(PyOperationRef operationRef, MlirValue value)
      : BaseTy(operationRef, value) {}
  PyConcreteValue(PyValue &orig)
      : PyConcreteValue(orig.getParentOperation(), castFrom(orig)) {}

  static MlirValue castFrom(PyValue &orig) {
    if (!DerivedTy::isaFunction(orig.get())) {
      auto origRepr = py::str(py::cast(orig)).cast<std::string>();
      auto errMsg = Twine("Cannot cast value to ") + DerivedTy::pyClassName +
                    " (from " + origRepr + ")";
      throw py::value_error(errMsg.str());
    }
    return orig.get();
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(py::init<PyValue &>(), py::keep_alive<0, 1>(), py::arg("value"));
    cls.def_static(
        "isinstance",
        [](PyValue &otherValue) -> bool {
          return DerivedTy::isaFunction(otherValue);
        },
        py::arg("other_value"));
    cls.def("__str__", [](const py::object &self) {
      auto Value =
          py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Value");
      return py::str(Value(self))
          .attr("replace")("Value", DerivedTy::pyClassName);
    });
    DerivedTy::bindDerived(cls);
    pybind11::implicitly_convertible<PyValue, DerivedTy>();
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
  //  // Keeps the parent operation alive
  //  pybind11::object parentOperationKeepAlive;
};

class PyTorch_NoneValue;
PyTorch_NoneValue makePyTorchNoneValue(PyLocation *loc, PyInsertionPoint *ip);

class PyTorch_BoolValue;
PyTorch_BoolValue makePyTorchBoolValue(bool b, PyLocation *loc,
                                       PyInsertionPoint *ip);

class PyTorch_DeviceValue;
PyTorch_DeviceValue makePyTorchDeviceValue(const std::string &b,
                                           PyLocation *loc,
                                           PyInsertionPoint *ip);

class PyTorch_FloatValue;
PyTorch_FloatValue makePyTorchFloatValue(float b, PyLocation *loc,
                                         PyInsertionPoint *ip);

class PyTorch_StringValue;
PyTorch_StringValue makePyTorchStringValue(const std::string &b,
                                           PyLocation *loc,
                                           PyInsertionPoint *ip);

class PyTorch_IntValue;
PyTorch_IntValue makePyTorchIntValue(int b, PyLocation *loc,
                                     PyInsertionPoint *ip);

class PyAnyTorchListValue;
PyAnyTorchListValue makePyAnyTorchListValue(const py::object &type,
                                            const py::list &operands,
                                            PyLocation *loc,
                                            PyInsertionPoint *ip);

class PyTorch_NoneValue : public PyConcreteValue<PyTorch_NoneValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_NoneValue;
  static constexpr const char *pyClassName = "Torch_NoneValue";
  PyTorch_NoneValue()
      : PyTorch_NoneValue(makePyTorchNoneValue(&DefaultingPyLocation::resolve(),
                                               getInsertionPoint())){};
  PyTorch_NoneValue(const py::none &n)
      : PyTorch_NoneValue(makePyTorchNoneValue(&DefaultingPyLocation::resolve(),
                                               getInsertionPoint())){};
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

#define DECLARE_SCALAR_VALUE(TORCHTYPE, CPPTYPE)                               \
  class PyTorch_##TORCHTYPE##Value                                             \
      : public PyConcreteValue<PyTorch_##TORCHTYPE##Value> {                   \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction = isATorch_##TORCHTYPE##Value;  \
    static constexpr const char *pyClassName = "Torch_" #TORCHTYPE "Value";    \
    using PyConcreteValue::PyConcreteValue;                                    \
    static void bindDerived(ClassTy &c);                                       \
                                                                               \
    PyTorch_##TORCHTYPE##Value(CPPTYPE b)                                      \
        : PyTorch_##TORCHTYPE##Value(makePyTorch##TORCHTYPE##Value(            \
              b, &DefaultingPyLocation::resolve(), getInsertionPoint())) {}    \
  };
DECLARE_SCALAR_VALUE(Bool, bool)
DECLARE_SCALAR_VALUE(Device, std::string)
DECLARE_SCALAR_VALUE(Float, float)
DECLARE_SCALAR_VALUE(String, std::string)
#undef DECLARE_SCALAR_VALUE

class PyTorch_IntValue : public PyConcreteValue<PyTorch_IntValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_IntValue;
  static constexpr const char *pyClassName = "Torch_IntValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
  PyTorch_IntValue(int b)
      : PyTorch_IntValue(makePyTorchIntValue(
            b, &DefaultingPyLocation::resolve(), getInsertionPoint())) {}
  PyTorch_IntValue(DType b)
      : PyTorch_IntValue(makePyTorchIntValue(to_underlying(b),
                                             &DefaultingPyLocation::resolve(),
                                             getInsertionPoint())) {}
};

template <class T> struct tag {
  using type = T;
};
// the above helper let you work with types as values.
// the tag<T> type is a variable with no state besides the type it caries.
// https://stackoverflow.com/a/31616949
// the sole use case here is templatizing the constructor of PyAnyTorchListValue

template <class T>
static inline py::list mapListElementsToPyType(py::list list) {
  for (unsigned long i = 0; i < list.size(); ++i) {
    if (list[i].is_none())
      list[i] = PyTorch_NoneValue(list[i]);
    else
      list[i] = py::cast<T>(list[i]);
  }
  return list;
}

py::object mapListToPyTorchListValue(const py::list &list, PyMlirContext &ctx);

class PyAnyTorchListValue : public PyConcreteValue<PyAnyTorchListValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchListValue;
  static constexpr const char *pyClassName = "AnyTorchListValue";
  using PyConcreteValue::PyConcreteValue;

  PyAnyTorchListValue(const py::object &type, const py::list &list)
      : PyAnyTorchListValue(makePyAnyTorchListValue(
            type, list, &DefaultingPyLocation::resolve(),
            getInsertionPoint())){};

  template <class T>
  PyAnyTorchListValue(py::object type, const py::list &list, tag<T>)
      : PyAnyTorchListValue(type, mapListElementsToPyType<T>(list)){};

  PyAnyTorchListValue(const py::list &list)
      : PyAnyTorchListValue(
            mapListToPyTorchListValue(list, DefaultingPyMlirContext::resolve()),
            list){};

  PyAnyTorchListValue(const py::tuple &list)
      : PyAnyTorchListValue(
            mapListToPyTorchListValue(list, DefaultingPyMlirContext::resolve()),
            list){};

  static void bindDerived(ClassTy &c);
};

#define DECLARE_SCALAR_VALUE(TORCHTYPE)                                        \
  class PyTorch_##TORCHTYPE##Value                                             \
      : public PyConcreteValue<PyTorch_##TORCHTYPE##Value> {                   \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction = isATorch_##TORCHTYPE##Value;  \
    static constexpr const char *pyClassName = "Torch_" #TORCHTYPE "Value";    \
    using PyConcreteValue::PyConcreteValue;                                    \
    static void bindDerived(ClassTy &c);                                       \
  };

DECLARE_SCALAR_VALUE(Any)
DECLARE_SCALAR_VALUE(LinearParams)
DECLARE_SCALAR_VALUE(Number)
#undef DECLARE_SCALAR_VALUE

#define DECLARE_LIST_BASE_CONCRETE_VALUE(TORCHTYPE)                            \
  class PyAnyTorchListOfTorch##TORCHTYPE##Value                                \
      : public PyConcreteValue<PyAnyTorchListOfTorch##TORCHTYPE##Value,        \
                               PyAnyTorchListValue> {                          \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction =                               \
        isAAnyTorchListOfTorch##TORCHTYPE##Value;                              \
    static constexpr const char *pyClassName =                                 \
        "AnyTorchListOfTorch" #TORCHTYPE "Value";                              \
    using PyConcreteValue::PyConcreteValue;                                    \
    /* you should be able to simplify this to just be PyAnyTorchListValue(l)   \
     * but for some reason it doesn't work for Bool*/                          \
    PyAnyTorchListOfTorch##TORCHTYPE##Value(const py::list &l)                 \
        : PyAnyTorchListOfTorch##TORCHTYPE##Value(                             \
              py::cast(PyAnyTorchListValue(                                    \
                           py::cast(PyAnyTorchListOfTorch##TORCHTYPE##Type(    \
                               DefaultingPyMlirContext::resolve())),           \
                           l, tag<PyTorch_##TORCHTYPE##Value>{}))              \
                  .cast<PyAnyTorchListOfTorch##TORCHTYPE##Value>()){};         \
                                                                               \
    static void bindDerived(ClassTy &c);                                       \
  };

DECLARE_LIST_BASE_CONCRETE_VALUE(Bool)
DECLARE_LIST_BASE_CONCRETE_VALUE(Float)
DECLARE_LIST_BASE_CONCRETE_VALUE(Int)
DECLARE_LIST_BASE_CONCRETE_VALUE(String)
#undef DECLARE_LIST_BASE_CONCRETE_VALUE

class PyAnyTorchOptionalValue
    : public PyConcreteValue<PyAnyTorchOptionalValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalValue;
  static constexpr const char *pyClassName = "AnyTorchOptionalValue";
  PyAnyTorchOptionalValue(const py::none &n)
      : PyAnyTorchOptionalValue(
            py::cast(PyTorch_NoneValue(n)).cast<PyAnyTorchOptionalValue>()){};
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

class PyAnyTorchOptionalGeneratorValue
    : public PyConcreteValue<PyAnyTorchOptionalGeneratorValue,
                             PyAnyTorchOptionalValue> {
public:
  static constexpr IsAFunctionTy isaFunction =
      isAAnyTorchOptionalGeneratorValue;
  static constexpr const char *pyClassName = "AnyTorchOptionalGeneratorValue";
  PyAnyTorchOptionalGeneratorValue(const py::none &n)
      : PyAnyTorchOptionalGeneratorValue(
            py::cast(PyTorch_NoneValue(n))
                .cast<PyAnyTorchOptionalGeneratorValue>()) {}
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

#define DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(TORCHTYPE, CPPTYPE)               \
  class PyAnyTorchOptional##TORCHTYPE##Value                                   \
      : public PyConcreteValue<PyAnyTorchOptional##TORCHTYPE##Value,           \
                               PyAnyTorchOptionalValue> {                      \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction =                               \
        isAAnyTorchOptional##TORCHTYPE##Value;                                 \
    static constexpr const char *pyClassName =                                 \
        "AnyTorchOptional" #TORCHTYPE "Value";                                 \
    PyAnyTorchOptional##TORCHTYPE##Value(const py::none &n)                    \
        : PyAnyTorchOptional##TORCHTYPE                                        \
          ##Value(py::cast(PyTorch_NoneValue(n))                               \
                      .cast<PyAnyTorchOptional##TORCHTYPE##Value>()) {}        \
    PyAnyTorchOptional##TORCHTYPE##Value(CPPTYPE n)                            \
        : PyAnyTorchOptional##TORCHTYPE                                        \
          ##Value(py::cast(PyTorch_##TORCHTYPE##Value(n))                      \
                      .cast<PyAnyTorchOptional##TORCHTYPE##Value>()) {}        \
    using PyConcreteValue::PyConcreteValue;                                    \
    static void bindDerived(ClassTy &c);                                       \
  };

DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(Bool, bool)
DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(Device, std::string)
DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(Float, float)
DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(String, std::string)
#undef DECLARE_OPTIONAL_BASE_CONCRETE_VALUE

class PyAnyTorchOptionalScalarValue
    : public PyConcreteValue<PyAnyTorchOptionalScalarValue,
                             PyAnyTorchOptionalValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalScalarValue;
  static constexpr const char *pyClassName = "AnyTorchOptionalScalarValue";
  PyAnyTorchOptionalScalarValue(const py::none &n)
      : PyAnyTorchOptionalScalarValue(
            py::cast(PyTorch_NoneValue(n))
                .cast<PyAnyTorchOptionalScalarValue>()) {}
  PyAnyTorchOptionalScalarValue(int n)
      : PyAnyTorchOptionalScalarValue(
            py::cast(PyTorch_IntValue(n))
                .cast<PyAnyTorchOptionalScalarValue>()) {}
  PyAnyTorchOptionalScalarValue(float n)
      : PyAnyTorchOptionalScalarValue(
            py::cast(PyTorch_FloatValue(n))
                .cast<PyAnyTorchOptionalScalarValue>()) {}
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyAnyTorchOptionalIntValue
    : public PyConcreteValue<PyAnyTorchOptionalIntValue,
                             PyAnyTorchOptionalValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalIntValue;
  static constexpr const char *pyClassName = "AnyTorchOptionalIntValue";
  PyAnyTorchOptionalIntValue(const py::none &n)
      : PyAnyTorchOptionalIntValue(
            py::cast(PyTorch_NoneValue(n)).cast<PyAnyTorchOptionalIntValue>()) {
  }
  PyAnyTorchOptionalIntValue(int n)
      : PyAnyTorchOptionalIntValue(
            py::cast(PyTorch_IntValue(n)).cast<PyAnyTorchOptionalIntValue>()) {}
  PyAnyTorchOptionalIntValue(DType n)
      : PyAnyTorchOptionalIntValue(
            py::cast(PyTorch_IntValue(n)).cast<PyAnyTorchOptionalIntValue>()) {}
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyAnyTorchOptionalListOfTorchIntValue
    : public PyConcreteValue<PyAnyTorchOptionalListOfTorchIntValue,
                             PyAnyTorchOptionalValue> {
public:
  static constexpr IsAFunctionTy isaFunction =
      isAAnyTorchOptionalListOfTorchIntValue;
  static constexpr const char *pyClassName =
      "AnyTorchOptionalListOfTorchIntValue";
  PyAnyTorchOptionalListOfTorchIntValue(const py::none &n)
      : PyAnyTorchOptionalListOfTorchIntValue(
            py::cast(PyTorch_NoneValue(n))
                .cast<PyAnyTorchOptionalListOfTorchIntValue>()) {}
  PyAnyTorchOptionalListOfTorchIntValue(const py::list &l)
      : PyAnyTorchOptionalListOfTorchIntValue(
            py::cast(PyAnyTorchListOfTorchIntValue(l))
                .cast<PyAnyTorchOptionalListOfTorchIntValue>()) {}
  PyAnyTorchOptionalListOfTorchIntValue(const py::tuple &l)
      : PyAnyTorchOptionalListOfTorchIntValue(
            py::cast(PyAnyTorchListOfTorchIntValue(l))
                .cast<PyAnyTorchOptionalListOfTorchIntValue>()) {}
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyTorch_DictValue : public PyConcreteValue<PyTorch_DictValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_DictValue;
  static constexpr const char *pyClassName = "Torch_DictValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyTorch_TupleValue : public PyConcreteValue<PyTorch_TupleValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_TupleValue;
  static constexpr const char *pyClassName = "Torch_TupleValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyTorch_NnModuleValue : public PyConcreteValue<PyTorch_NnModuleValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_NnModuleValue;
  static constexpr const char *pyClassName = "Torch_NnModuleValue";
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

class PyAnyTorchScalarValue : public PyConcreteValue<PyAnyTorchScalarValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchScalarValue;
  static constexpr const char *pyClassName = "AnyTorchScalarValue";
  PyAnyTorchScalarValue(int n)
      : PyAnyTorchScalarValue(
            py::cast(PyTorch_IntValue(n)).cast<PyAnyTorchScalarValue>()){};
  PyAnyTorchScalarValue(float n)
      : PyAnyTorchScalarValue(
            py::cast(PyTorch_FloatValue(n)).cast<PyAnyTorchScalarValue>()){};
  using PyConcreteValue::PyConcreteValue;
  static void bindDerived(ClassTy &c);
};

void populateTorchMLIRValues(py::module &m);

} // namespace mlir::torch

#endif // PI_TORCHVALUES_H
