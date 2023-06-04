//
// Created by maksim on 5/13/23.
//

#ifndef PI_TORCHVALUES_H
#define PI_TORCHVALUES_H

#include "Globals.h"
#include "IRModule.h"

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/IR/TypeSupport.h"

#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

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
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError, Twine("Cannot cast value to ") +
                                             DerivedTy::pyClassName +
                                             " (from " + origRepr + ")");
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
};

class PyTorch_NoneValue : public PyConcreteValue<PyTorch_NoneValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isATorch_NoneValue;
  static constexpr const char *pyClassName = "Torch_NoneValue";
  PyTorch_NoneValue()
      : PyTorch_NoneValue(mlir::python::PyGlobals::get()
                              .lookupOperationClass("torch.constant.none")
                              .value()()
                              .cast<PyTorch_NoneValue>()){};
  PyTorch_NoneValue(const py::none &n)
      : PyTorch_NoneValue(mlir::python::PyGlobals::get()
                              .lookupOperationClass("torch.constant.none")
                              .value()()
                              .cast<PyTorch_NoneValue>()){};
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

#define DECLARE_SCALAR_VALUE(TORCHTYPE, CPPTYPE, CONSTANTSTR)                  \
  class PyTorch_##TORCHTYPE##Value                                             \
      : public PyConcreteValue<PyTorch_##TORCHTYPE##Value> {                   \
  public:                                                                      \
    static constexpr IsAFunctionTy isaFunction = isATorch_##TORCHTYPE##Value;  \
    static constexpr const char *pyClassName = "Torch_" #TORCHTYPE "Value";    \
    using PyConcreteValue::PyConcreteValue;                                    \
    static void bindDerived(ClassTy &c);                                       \
                                                                               \
    PyTorch_##TORCHTYPE##Value(CPPTYPE b)                                      \
        : PyTorch_##TORCHTYPE                                                  \
          ##Value(mlir::python::PyGlobals::get()                               \
                      .lookupOperationClass("torch.constant." #CONSTANTSTR)    \
                      .value()(b)                                              \
                      .cast<PyTorch_##TORCHTYPE##Value>()) {}                  \
  };
DECLARE_SCALAR_VALUE(Bool, bool, bool)
DECLARE_SCALAR_VALUE(Device, int, device)
DECLARE_SCALAR_VALUE(Int, int, int)
DECLARE_SCALAR_VALUE(Float, float, float)
DECLARE_SCALAR_VALUE(String, std::string, str)
#undef DECLARE_SCALAR_VALUE

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

py::object mapListToPyTorchListValue(const py::list &list);

class PyAnyTorchListValue : public PyConcreteValue<PyAnyTorchListValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchListValue;
  static constexpr const char *pyClassName = "AnyTorchListValue";
  using PyConcreteValue::PyConcreteValue;

  PyAnyTorchListValue(const py::object &type, const py::list &list)
      : PyAnyTorchListValue(
            mlir::python::PyGlobals::get()
                .lookupOperationClass("torch.prim.ListConstruct")
                .value()(type, list)
                .template cast<PyAnyTorchListValue>()){};

  template <class T>
  PyAnyTorchListValue(py::object type, const py::list &list, tag<T>)
      : PyAnyTorchListValue(type, mapListElementsToPyType<T>(list)){};

  PyAnyTorchListValue(const py::list &list)
      : PyAnyTorchListValue(mapListToPyTorchListValue(list), list){};

  PyAnyTorchListValue(const py::tuple &list)
      : PyAnyTorchListValue(mapListToPyTorchListValue(list), list){};

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
DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(Device, int)
DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(Int, int)
DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(Float, float)
DECLARE_OPTIONAL_BASE_CONCRETE_VALUE(String, std::string)
#undef DECLARE_OPTIONAL_BASE_CONCRETE_VALUE

class PyAnyTorchOptionalScalarValue
    : public PyConcreteValue<PyAnyTorchOptionalScalarValue,
                             PyAnyTorchOptionalValue> {
public:
  static constexpr IsAFunctionTy isaFunction = isAAnyTorchOptionalScalarValue;
  static constexpr const char *pyClassName = "AnyTorchOptional"
                                             "Scalar"
                                             "Value";
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
