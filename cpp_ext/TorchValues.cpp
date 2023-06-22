//
// Created by maksim on 5/13/23.
//
#include "IRModule.h"
#include "PybindUtils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Interfaces.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "TorchDType.h"
#include "TorchOps.h"
#include "TorchTypes.h"
#include "TorchValues.h"
#include "torch-mlir-c/TorchTypes.h"

#include <functional>
#include <iostream>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace py = pybind11;
using namespace py::literals;
using namespace mlir::python;

using llvm::StringRef;
using llvm::Twine;
using std::vector;

namespace mlir::torch {

bool isAAnyTorchDictKeyValue(MlirValue value) {
  return isAAnyTorchDictKeyType(mlirValueGetType(value));
}

bool isAAnyTorchListOfOptionalIntValue(MlirValue value) {
  return isAAnyTorchListOfOptionalIntType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchBoolValue(MlirValue value) {
  return isAAnyTorchListOfTorchBoolType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchIntValue(MlirValue value) {
  return isAAnyTorchListOfTorchIntType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchFloatValue(MlirValue value) {
  return isAAnyTorchListOfTorchFloatType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTorchStringValue(MlirValue value) {
  return isAAnyTorchListOfTorchStringType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalFloatValue(MlirValue value) {
  return isAAnyTorchOptionalFloatType(mlirValueGetType(value));
}

bool isAAnyTorchListValue(MlirValue value) {
  return isAAnyTorchListType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalBoolValue(MlirValue value) {
  return isAAnyTorchOptionalBoolType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalDeviceValue(MlirValue value) {
  return isAAnyTorchOptionalDeviceType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalGeneratorValue(MlirValue value) {
  return isAAnyTorchOptionalGeneratorType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalIntValue(MlirValue value) {
  return isAAnyTorchOptionalIntType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalListOfTorchIntValue(MlirValue value) {
  return isAAnyTorchOptionalListOfTorchIntType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalStringValue(MlirValue value) {
  return isAAnyTorchOptionalStringType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalValue(MlirValue value) {
  return isAAnyTorchOptionalType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalScalarValue(MlirValue value) {
  return isAAnyTorchOptionalScalarType(mlirValueGetType(value));
}

bool isAAnyTorchScalarValue(MlirValue value) {
  return isAAnyTorchScalarType(mlirValueGetType(value));
}

bool isAAnyTorchValue(MlirValue value) {
  return isAAnyTorchType(mlirValueGetType(value));
}

#define DECLARE_ISA_UNDERSCORE_VALUE(TORCHTYPE)                                \
  bool isATorch_##TORCHTYPE##Value(MlirValue value) {                          \
    return isATorch_##TORCHTYPE##Type(mlirValueGetType(value));                \
  }
FORALL_UNDERSCORE_TYPES(DECLARE_ISA_UNDERSCORE_VALUE)
#undef DECLARE_ISA_UNDERSCORE_VALUE

py::object mapListToPyTorchListValue(const py::list &list, PyMlirContext &ctx) {
  if (list.empty())
    throw std::runtime_error("Can't cast empty list");

  MlirType containedType;

  if (py::isinstance<py::int_>(list[0])) {
    containedType = torchMlirTorchIntTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_IntValue>(list);
  } else if (py::isinstance<py::float_>(list[0])) {
    containedType = torchMlirTorchFloatTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_FloatValue>(list);
  } else if (py::isinstance<py::bool_>(list[0])) {
    containedType = torchMlirTorchBoolTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_BoolValue>(list);
  } else if (py::isinstance<py::str>(list[0])) {
    containedType = torchMlirTorchStringTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_StringValue>(list);
  } else if (list[0].is_none()) {
    containedType = torchMlirTorchNoneTypeGet(ctx.get());
    mapListElementsToPyType<PyTorch_NoneValue>(list);
  } else {
    if (!py::isinstance<PyValue>(list[0]))
      throw std::runtime_error("Can't infer list element type.");
    containedType = mlirValueGetType(list[0].cast<PyValue>().get());
  }

  auto pyType =
      py::cast(PyType(ctx.getRef(), torchMlirTorchListTypeGet(containedType)));
  return pyType;
}

////////////////////////////////////////////////////////////////////////////////

struct AppendResultsCallbackData {
  std::vector<PyType> &inferredTypes;
  PyMlirContext *pyMlirContext;
};

void appendResultsCallback(intptr_t nTypes, MlirType *types, void *userData) {
  auto *data = static_cast<AppendResultsCallbackData *>(userData);
  data->inferredTypes.reserve(data->inferredTypes.size() + nTypes);
  for (intptr_t i = 0; i < nTypes; ++i) {
    data->inferredTypes.emplace_back(data->pyMlirContext->getRef(), types[i]);
  }
}

std::vector<PyType> inferReturnTypes(
    const std::string &operationName,
    const std::vector<std::reference_wrapper<const PyValue>> &operands,
    PyMlirContext *pyContext, PyLocation *loc,
    const std::optional<PyAttribute> &attributes, void *properties) {
  std::vector<MlirValue> mlirOperands;
  std::vector<MlirRegion> mlirRegions;

  mlirOperands.reserve(operands.size());
  for (PyValue operand : operands) {
    mlirOperands.push_back(operand.get());
  }

  std::vector<PyType> inferredTypes;
  AppendResultsCallbackData data{inferredTypes, pyContext};
  MlirStringRef opNameRef =
      mlirStringRefCreate(operationName.data(), operationName.length());
  MlirAttribute attributeDict =
      attributes ? attributes->get() : mlirAttributeGetNull();

  MlirLogicalResult result = mlirInferTypeOpInterfaceInferReturnTypes(
      opNameRef, pyContext->get(), loc->get(), mlirOperands.size(),
      mlirOperands.data(), attributeDict, properties, mlirRegions.size(),
      mlirRegions.data(), &appendResultsCallback, &data);

  if (mlirLogicalResultIsFailure(result)) {
    throw py::value_error("Failed to infer result types");
  }

  return inferredTypes;
}

MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

PyInsertionPoint &DefaultingPyInsertionPoint::resolve() {
  auto *ip = PyThreadContextEntry::getDefaultInsertionPoint();
  if (!ip) {
    throw std::runtime_error(
        "An MLIR function requires a InsertionPoint but none was provided in "
        "the "
        "call or from the surrounding environment. Either pass to the function "
        "with a 'ip=' argument or establish a default using 'with ip:'");
  }
  return *ip;
}

PyOperationRef createOperation(
    const std::string &name,
    const std::vector<std::reference_wrapper<const PyType>> &results,
    const std::vector<std::reference_wrapper<const PyValue>> &operands,
    const std::map<std::string, MlirAttribute> &attributes, PyLocation *loc,
    PyInsertionPoint *ip) {
  std::vector<MlirValue> mlirOperands;
  std::vector<MlirType> mlirResults;
  std::vector<std::pair<std::string, MlirAttribute>> mlirAttributes;

  // Unpack/validate operands.
  mlirOperands.reserve(operands.size());
  for (PyValue operand : operands) {
    mlirOperands.push_back(operand.get());
  }

  // Unpack/validate results.
  mlirResults.reserve(results.size());
  for (PyType result : results) {
    mlirResults.push_back(result);
  }
  // Unpack/validate attributes.
  mlirAttributes.reserve(attributes.size());
  for (auto &it : attributes) {
    std::string key;
    try {
      key = it.first;
    } catch (py::cast_error &err) {
      std::string msg = "Invalid attribute key (not a string) when "
                        "attempting to create the operation \"" +
                        name + "\" (" + err.what() + ")";
      throw py::cast_error(msg);
    }
    try {
      auto &attribute = it.second;
      // TODO: Verify attribute originates from the same context.
      mlirAttributes.emplace_back(std::move(key), attribute);
    } catch (py::reference_cast_error &) {
      // This exception seems thrown when the value is "None".
      std::string msg =
          "Found an invalid (`None`?) attribute value for the key \"" + key +
          "\" when attempting to create the operation \"" + name + "\"";
      throw py::cast_error(msg);
    } catch (py::cast_error &err) {
      std::string msg = "Invalid attribute value for the key \"" + key +
                        "\" when attempting to create the operation \"" + name +
                        "\" (" + err.what() + ")";
      throw py::cast_error(msg);
    }
  }

  // Apply unpacked/validated to the operation state. Beyond this
  // point, exceptions cannot be thrown or else the state will leak.
  MlirOperationState state =
      mlirOperationStateGet(toMlirStringRef(name), loc->get());
  if (!mlirOperands.empty())
    mlirOperationStateAddOperands(&state, mlirOperands.size(),
                                  mlirOperands.data());
  if (!mlirResults.empty())
    mlirOperationStateAddResults(&state, mlirResults.size(),
                                 mlirResults.data());
  if (!mlirAttributes.empty()) {
    // Note that the attribute names directly reference bytes in
    // mlirAttributes, so that vector must not be changed from here
    // on.
    std::vector<MlirNamedAttribute> mlirNamedAttributes;
    mlirNamedAttributes.reserve(mlirAttributes.size());
    for (auto &it : mlirAttributes)
      mlirNamedAttributes.push_back(mlirNamedAttributeGet(
          mlirIdentifierGet(mlirAttributeGetContext(it.second),
                            toMlirStringRef(it.first)),
          it.second));
    mlirOperationStateAddAttributes(&state, mlirNamedAttributes.size(),
                                    mlirNamedAttributes.data());
  }

  // Construct the operation.
  MlirOperation operation = mlirOperationCreate(&state);
  PyOperationRef created =
      PyOperation::createDetached(loc->getContext(), operation);
  if (ip)
    ip->insert(*created.get());

  return created;
}

////////////////////////////////////////////////////////////////////////////////

PyTorch_NoneValue makePyTorchNoneValue(PyLocation *loc, PyInsertionPoint *ip) {

  auto resultType =
      py::cast(torchMlirTorchNoneTypeGet(loc->getContext()->get()))
          .cast<PyType>();
  PyOperationRef opRef = createOperation("torch.constant.none", {resultType},
                                         /*operands*/ {}, {}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

PyTorch_BoolValue makePyTorchBoolValue(bool b, PyLocation *loc,
                                       PyInsertionPoint *ip) {
  auto resultType =
      py::cast(torchMlirTorchBoolTypeGet(loc->getContext()->get()))
          .cast<PyType>();
  PyOperationRef opRef = createOperation(
      "torch.constant.bool", {resultType},
      /*operands*/ {},
      {{"value", mlirBoolAttrGet(loc->getContext()->get(), b)}}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

PyTorch_DeviceValue makePyTorchDeviceValue(const std::string &b,
                                           PyLocation *loc,
                                           PyInsertionPoint *ip) {
  auto resultType =
      py::cast(torchMlirTorchDeviceTypeGet(loc->getContext()->get()))
          .cast<PyType>();
  PyOperationRef opRef = createOperation(
      "torch.constant.device", {resultType},
      /*operands*/ {},
      {{"value", mlirStringAttrGet(loc->getContext()->get(),
                                   mlir::torch::toMlirStringRef(b))}},
      loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

PyTorch_FloatValue makePyTorchFloatValue(float b, PyLocation *loc,
                                         PyInsertionPoint *ip) {
  auto resultType =
      py::cast(torchMlirTorchFloatTypeGet(loc->getContext()->get()))
          .cast<PyType>();
  PyOperationRef opRef = createOperation(
      "torch.constant.float", {resultType},
      /*operands*/ {},
      {{"value",
        mlirFloatAttrDoubleGet(loc->getContext()->get(),
                               mlirF64TypeGet(loc->getContext()->get()), b)}},
      loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

PyTorch_StringValue makePyTorchStringValue(const std::string &b,
                                           PyLocation *loc,
                                           PyInsertionPoint *ip) {
  auto resultType =
      py::cast(torchMlirTorchStringTypeGet(loc->getContext()->get()))
          .cast<PyType>();
  PyOperationRef opRef = createOperation(
      "torch.constant.str", {resultType},
      /*operands*/ {},
      {{"value", mlirStringAttrGet(loc->getContext()->get(),
                                   mlir::torch::toMlirStringRef(b))}},
      loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

PyTorch_IntValue makePyTorchIntValue(int b, PyLocation *loc,
                                     PyInsertionPoint *ip) {
  auto resultType = py::cast(torchMlirTorchIntTypeGet(loc->getContext()->get()))
                        .cast<PyType>();
  PyOperationRef opRef = createOperation(
      "torch.constant.int", {resultType},
      /*operands*/ {},
      {{"value", mlirIntegerAttrGet(
                     mlirIntegerTypeGet(loc->getContext()->get(), 64), b)}},
      loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

PyAnyTorchListValue makePyAnyTorchListValue(const py::object &type,
                                            const py::list &operands,
                                            PyLocation *loc,
                                            PyInsertionPoint *ip) {
  std::vector<std::reference_wrapper<const PyValue>> operands_;
  for (const auto &operand : operands)
    operands_.emplace_back(operand.cast<PyValue &>());
  auto resultType = type.cast<PyType>();
  PyOperationRef opRef = createOperation("torch.prim.ListConstruct",
                                         {resultType}, operands_, {}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

template <typename T, typename U>
T makeGetItem(U &self, const PyTorch_IntValue &idx, PyLocation *loc,
              PyInsertionPoint *ip) {
  MlirType t;
  if (std::is_same<T, PyTorch_BoolValue>::value)
    t = torchMlirTorchBoolTypeGet(loc->getContext()->get());
  else if (std::is_same<T, PyTorch_FloatValue>::value)
    t = torchMlirTorchFloatTypeGet(loc->getContext()->get());
  else if (std::is_same<T, PyTorch_IntValue>::value)
    t = torchMlirTorchIntTypeGet(loc->getContext()->get());
  else if (std::is_same<T, PyTorch_StringValue>::value)
    t = torchMlirTorchStringTypeGet(loc->getContext()->get());
  else
    throw std::runtime_error("unknown element type");
  auto resultType = py::cast(t).cast<PyType>();
  PyOperationRef opRef = createOperation(
      "torch.aten.__getitem__.t", {resultType}, {self, idx}, {}, loc, ip);
  return {opRef, mlirOperationGetResult(opRef->get(), 0)};
}

MlirOperation getOwner(const PyValue &value) {
  MlirOperation owner;
  if (mlirValueIsAOpResult(value))
    owner = mlirOpResultGetOwner(value);
  else if (mlirValueIsABlockArgument(value))
    owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(value));
  else
    throw py::value_error("unknown value owner");
  return owner;
}

PyLocation getValueLocation(const PyValue &value) {
  auto location = mlirOperationGetLocation(getOwner(value));
  auto context = mlirLocationGetContext(location);
  auto ctx = PyMlirContext::forContext(context);
  return {ctx, location};
}

template <typename T, typename U>
std::vector<T> makeListIter(U &self, PyLocation *loc, PyInsertionPoint *ip) {
  if (!self.length)
    throw py::value_error("PyAnyTorchListValue has unknown length;");
  std::vector<PyType> _returnTypes;
  auto containedType =
      torchMlirTorchListTypeGetContainedType(mlirValueGetType(self));
  for (int i = 0; i < self.length.value(); ++i) {
    _returnTypes.push_back(py::cast(containedType).template cast<PyType>());
  }
  std::vector<std::reference_wrapper<const PyType>> returnTypes;
  for (const auto &returnType : _returnTypes)
    returnTypes.push_back(returnType);
  PyOperationRef opRef =
      createOperation("torch.prim.ListUnpack", returnTypes, {self},
                      /*attributes=*/{}, loc, ip);
  MlirOperation operation = opRef->get();
  std::vector<T> list;
  list.reserve(self.length.value());
  for (int i = 0; i < self.length.value(); ++i)
    list.push_back(py::cast(mlirOperationGetResult(operation, i)).cast<T>());
  return list;
}

// explicit template instantiation
template std::vector<PyAnyTorchTensorValue>
makeListIter<PyAnyTorchTensorValue, PyAnyTorchListOfTensorValue const>(
    PyAnyTorchListOfTensorValue const &, PyLocation *, PyInsertionPoint *);

void PyAnyTorchListValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::list>(), "value"_a);
  c.def(py::init<py::tuple>(), "value"_a);
  c.def(
      "__add__",
      [](const PyAnyTorchListValue &self,
         const PyAnyTorchListValue &other) -> PyAnyTorchListValue {
        auto loc = getValueLocation(self);
        return add(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);

  c.def(
      "__radd__",
      [](const PyAnyTorchListValue &self,
         const PyAnyTorchListValue &other) -> PyAnyTorchListValue {
        auto loc = getValueLocation(self);
        return add(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);

  py::implicitly_convertible<py::list, PyAnyTorchListValue>();
  py::implicitly_convertible<py::tuple, PyAnyTorchListValue>();
  c.def("__len__", [](const PyAnyTorchListValue &self) { return self.length; });
}

#define DEFINE_LIST_BASE_CONCRETE_VALUE(TORCHTYPE, SCALARTYPE)                 \
  void PyAnyTorchListOf##TORCHTYPE##Value::bindDerived(ClassTy &c) {           \
    c.def(py::init<py::list>(), "value"_a);                                    \
    c.def(py::init<py::tuple>(), "value"_a);                                   \
    c.def(                                                                     \
        "__getitem__",                                                         \
        [](PyAnyTorchListOf##TORCHTYPE##Value & self,                          \
           const PyTorch_IntValue &idx) -> PyTorch_##SCALARTYPE##Value {       \
          return makeGetItem<PyTorch_##SCALARTYPE##Value>(                     \
              self, idx, &DefaultingPyLocation::resolve(),                     \
              &DefaultingPyInsertionPoint::resolve());                         \
        },                                                                     \
        "idx"_a);                                                              \
    c.def("__iter__", [](const PyAnyTorchListOf##TORCHTYPE##Value &self) {     \
      return py::iter(py::cast(makeListIter<PyTorch_##SCALARTYPE##Value>(      \
          self, &DefaultingPyLocation::resolve(),                              \
          &DefaultingPyInsertionPoint::resolve())));                           \
    });                                                                        \
    py::implicitly_convertible<py::list,                                       \
                               PyAnyTorchListOf##TORCHTYPE##Value>();          \
    py::implicitly_convertible<py::tuple,                                      \
                               PyAnyTorchListOf##TORCHTYPE##Value>();          \
  }
FORALL_LIST_BASE_CONCRETE_TYPES_WITH_TYPE(DEFINE_LIST_BASE_CONCRETE_VALUE)
#undef DEFINE_LIST_BASE_CONCRETE_VALUE

void PyAnyTorchOptionalGeneratorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), "value"_a);
  py::implicitly_convertible<py::none, PyAnyTorchOptionalGeneratorValue>();
}

void PyAnyTorchOptionalValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), "value"_a);
  py::implicitly_convertible<py::none, PyAnyTorchOptionalValue>();
}

#define DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(TORCHTYPE, CPPTYPE)                \
  void PyAnyTorchOptional##TORCHTYPE##Value::bindDerived(ClassTy &c) {         \
    c.def(py::init<py::none>(), "value"_a);                                    \
    c.def(py::init<CPPTYPE>(), "value"_a);                                     \
    py::implicitly_convertible<py::none,                                       \
                               PyAnyTorchOptional##TORCHTYPE##Value>();        \
    py::implicitly_convertible<CPPTYPE,                                        \
                               PyAnyTorchOptional##TORCHTYPE##Value>();        \
  }
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Bool, bool)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Device, std::string)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(Float, float)
DEFINE_OPTIONAL_BASE_CONCRETE_VALUE(String, std::string)
#undef DEFINE_OPTIONAL_BASE_CONCRETE_VALUE

void PyAnyTorchOptionalIntValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), "value"_a);
  c.def(py::init<int>(), "value"_a);
  c.def(py::init<DType>(), "value"_a);
  py::implicitly_convertible<py::none, PyAnyTorchOptionalIntValue>();
  py::implicitly_convertible<int, PyAnyTorchOptionalIntValue>();
  py::implicitly_convertible<DType, PyAnyTorchOptionalIntValue>();
}

void PyAnyTorchOptionalScalarValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), "value"_a);
  c.def(py::init<int>(), "value"_a);
  c.def(py::init<float>(), "value"_a);
  py::implicitly_convertible<py::none, PyAnyTorchOptionalScalarValue>();
  py::implicitly_convertible<int, PyAnyTorchOptionalScalarValue>();
  py::implicitly_convertible<float, PyAnyTorchOptionalScalarValue>();
}

void PyAnyTorchOptionalListOfTorchIntValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), "value"_a);
  c.def(py::init<py::list>(), "value"_a);
  c.def(py::init<py::tuple>(), "value"_a);
  py::implicitly_convertible<py::none, PyAnyTorchOptionalListOfTorchIntValue>();
  py::implicitly_convertible<py::list, PyAnyTorchOptionalListOfTorchIntValue>();
  py::implicitly_convertible<py::tuple,
                             PyAnyTorchOptionalListOfTorchIntValue>();
}

#define DEFINE_BIND_SCALAR_VALUE(TORCHTYPE)                                    \
  void PyTorch_##TORCHTYPE##Value::bindDerived(ClassTy &c) {}
DEFINE_BIND_SCALAR_VALUE(Any)
DEFINE_BIND_SCALAR_VALUE(LinearParams)
DEFINE_BIND_SCALAR_VALUE(Number)
#undef DEFINE_BIND_SCALAR_VALUE

void PyTorch_NoneValue::bindDerived(ClassTy &c) {
  c.def(py::init<>());
  c.def(py::init<py::none>(), "none"_a);
}

#define DEFINE_BIND_SCALAR_VALUE(TORCHTYPE, CPPTYPE, DUNDER)                   \
  void PyTorch_##TORCHTYPE##Value::bindDerived(ClassTy &c) {                   \
    c.def(py::init<CPPTYPE>(), "value"_a);                                     \
    c.def("__" #DUNDER "__",                                                   \
          py::overload_cast<const PyTorch_##TORCHTYPE##Value &>(               \
              &getAttributeValue<PyTorch_##TORCHTYPE##Value>));                \
                                                                               \
    py::implicitly_convertible<CPPTYPE, PyTorch_##TORCHTYPE##Value>();         \
  }
DEFINE_BIND_SCALAR_VALUE(Bool, bool, bool)
DEFINE_BIND_SCALAR_VALUE(Device, std::string, str)
DEFINE_BIND_SCALAR_VALUE(String, std::string, str)
#undef DEFINE_BIND_SCALAR_VALUE

void PyTorch_IntValue::bindDerived(ClassTy &c) {
  c.def(py::init<int>(), "value"_a);
  c.def(py::init<DType>(), "value"_a);
  c.def("__int__", py::overload_cast<const PyTorch_IntValue &>(
                       &getAttributeValue<PyTorch_IntValue>));
  c.def(
      "__add__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_IntValue {
        auto loc = getValueLocation(self);
        return add(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__sub__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_IntValue {
        auto loc = getValueLocation(self);
        return sub(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__mul__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_IntValue {
        auto loc = getValueLocation(self);
        return mul(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__truediv__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_FloatValue {
        auto loc = getValueLocation(self);
        return div(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__floordiv__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_IntValue {
        auto loc = getValueLocation(self);
        return floordiv(self, other, &loc,
                        &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__lt__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_BoolValue {
        auto loc = getValueLocation(self);
        return lt(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__le__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_BoolValue {
        auto loc = getValueLocation(self);
        return le(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__gt__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_BoolValue {
        auto loc = getValueLocation(self);
        return gt(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__ge__",
      [](const PyTorch_IntValue &self,
         const PyTorch_IntValue &other) -> PyTorch_BoolValue {
        auto loc = getValueLocation(self);
        return ge(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  py::implicitly_convertible<int, PyTorch_IntValue>();
  py::implicitly_convertible<DType, PyTorch_IntValue>();
}

void PyTorch_FloatValue::bindDerived(ClassTy &c) {
  c.def(py::init<float>(), "value"_a);
  c.def("__float__", py::overload_cast<const PyTorch_FloatValue &>(
                         &getAttributeValue<PyTorch_FloatValue>));
  c.def(
      "__sub__",
      [](const PyTorch_FloatValue &self,
         const PyTorch_FloatValue &other) -> PyTorch_FloatValue {
        auto loc = getValueLocation(self);
        return sub(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__mul__",
      [](const PyTorch_FloatValue &self,
         const PyTorch_FloatValue &other) -> PyTorch_FloatValue {
        auto loc = getValueLocation(self);
        return mul(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__truediv__",
      [](const PyTorch_FloatValue &self,
         const PyTorch_FloatValue &other) -> PyTorch_FloatValue {
        auto loc = getValueLocation(self);
        return div(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__lt__",
      [](const PyTorch_FloatValue &self,
         const PyTorch_FloatValue &other) -> PyTorch_BoolValue {
        auto loc = getValueLocation(self);
        return lt(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__gt__",
      [](const PyTorch_FloatValue &self,
         const PyTorch_FloatValue &other) -> PyTorch_BoolValue {
        auto loc = getValueLocation(self);
        return gt(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  c.def(
      "__ge__",
      [](const PyTorch_FloatValue &self,
         const PyTorch_FloatValue &other) -> PyTorch_BoolValue {
        auto loc = getValueLocation(self);
        return ge(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);
  py::implicitly_convertible<float, PyTorch_FloatValue>();
}

void PyTorch_DictValue::bindDerived(ClassTy &c) {}
void PyTorch_TupleValue::bindDerived(ClassTy &c) {}
void PyTorch_NnModuleValue::bindDerived(ClassTy &c) {}
void PyAnyTorchScalarValue::bindDerived(ClassTy &c) {
  c.def("__repr__", [](PyAnyTorchScalarValue &self) {
    auto origRepr =
        pybind11::repr(pybind11::cast(PyValue(self))).cast<std::string>();
    return std::regex_replace(origRepr, std::regex("Value"),
                              "AnyTorchScalarValue");
  });
  c.def(py::init<int>(), "value"_a);
  c.def(py::init<float>(), "value"_a);
  py::implicitly_convertible<int, PyAnyTorchScalarValue>();
  py::implicitly_convertible<float, PyAnyTorchScalarValue>();
}

////////////////////////////////////////////////////////////////////////////////

void populateTorchMLIRValues(py::module &m) {
  PyAnyTorchListValue::bind(m);
  PyAnyTorchOptionalValue::bind(m);

#define BIND_VALUE(VALUE) PyAnyTorchListOf##VALUE##Value::bind(m);
  FORALL_LIST_BASE_CONCRETE_TYPES(BIND_VALUE)
#undef BIND_VALUE

#define BIND_VALUE(VALUE) PyAnyTorchOptional##VALUE##Value::bind(m);
  FORALL_OPTIONAL_BASE_CONCRETE_TYPES(BIND_VALUE)
  BIND_VALUE(Scalar)
#undef BIND_VALUE

  PyAnyTorchOptionalListOfTorchIntValue::bind(m);

#define BIND_VALUE(VALUE) PyTorch_##VALUE##Value::bind(m);
  FORALL_SCALAR_TYPES(BIND_VALUE)
#undef BIND_VALUE

  PyTorch_DictValue::bind(m);
  PyTorch_TupleValue::bind(m);
  PyTorch_NnModuleValue::bind(m);
  PyAnyTorchScalarValue::bind(m);
}

void populateTorchConstants(py::module &m) { m.attr("nan") = std::nanf(""); }

} // namespace mlir::torch