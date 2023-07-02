//
// Created by mlevental on 5/15/23.
//

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <exception>

#include "TorchOps.h"
#include "TorchTensor.h"
#include "TorchValues.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlir::python;

namespace {
using namespace mlir::torch;
#include "TorchTensor.pybinds_tramps.cpp"
} // namespace

namespace mlir::torch {

// type constraints

bool isAAnyTorchListOfOptionalTensorType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           (((torchMlirTypeIsATorchBaseTensor(
                torchMlirTorchListTypeGetContainedType(type)))) ||
            ((torchMlirTypeIsATorchOptional(
                torchMlirTorchListTypeGetContainedType(type)))) ||
            ((torchMlirTypeIsATorchNone(
                torchMlirTorchListTypeGetContainedType(type)))))));
}

bool isAAnyTorchListOfTensorType(MlirType type) {
  return ((((torchMlirTypeIsATorchList(type))) &&
           ((torchMlirTypeIsATorchBaseTensor(
               torchMlirTorchListTypeGetContainedType(type))))));
}

bool isAAnyTorchOptionalTensorType(MlirType type) {
  return ((((torchMlirTypeIsATorchBaseTensor(type))) ||
           ((torchMlirTypeIsATorchOptional(type))) ||
           ((torchMlirTypeIsATorchNone(type)))));
}

bool isAAnyTorchTensorType(MlirType type) {
  return (((torchMlirTypeIsATorchBaseTensor(type))));
}

bool isATorch_NonValueTensorType(MlirType type) {
  return torchMlirTypeIsATorchNonValueTensor(type);
}

bool isATorch_ValueTensorType(MlirType type) {
  return torchMlirTypeIsATorchValueTensor(type);
}

// value constraints

bool isAAnyTorchListOfOptionalTensorValue(MlirValue value) {
  return isAAnyTorchListOfOptionalTensorType(mlirValueGetType(value));
}

bool isAAnyTorchListOfTensorValue(MlirValue value) {
  return isAAnyTorchListOfTensorType(mlirValueGetType(value));
}

bool isAAnyTorchOptionalTensorValue(MlirValue value) {
  return isAAnyTorchOptionalTensorType(mlirValueGetType(value));
}

bool isAAnyTorchTensorValue(MlirValue value) {
  return isAAnyTorchTensorType(mlirValueGetType(value));
}

bool isATorch_ValueTensorValue(MlirValue value) {
  return isATorch_ValueTensorType(mlirValueGetType(value));
}

bool isATorch_NonValueTensorValue(MlirValue value) {
  return isATorch_NonValueTensorType(mlirValueGetType(value));
}

// helpers

PyTorch_NonValueTensorType
PyTorch_NonValueTensorType::getWithLeastStaticInformation(
    PyMlirContext *context) {
  MlirType valueTensorType =
      torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
          context->get());
  return {context->getRef(), valueTensorType};
}

PyTorch_ValueTensorType
PyTorch_ValueTensorType::getWithLeastStaticInformation(PyMlirContext *context) {
  MlirType valueTensorType =
      torchMlirTorchValueTensorTypeGetWithLeastStaticInformation(
          context->get());
  return {context->getRef(), valueTensorType};
}

// type binders

void PyTorch_NonValueTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> sizes, MlirType dtype,
         DefaultingPyMlirContext &context) {
        return PyTorch_NonValueTensorType(std::move(sizes), dtype,
                                          context.get());
      },
      "sizes"_a, "dtype"_a, py::kw_only(), "context"_a = py::none());
  c.def_static(
      "get_with_least_static_information",
      [](DefaultingPyMlirContext &context) {
        return PyTorch_NonValueTensorType::getWithLeastStaticInformation(
            context.get());
      },
      py::kw_only(), "context"_a = py::none());
  c.def("sizes", [](MlirType self) {
    std::vector<int64_t> sizes(torchMlirTorchNonValueTensorTypeGetRank(self));
    if (torchMlirTorchNonValueTensorTypeGetSizes(self, sizes.data()))
      throw py::value_error("no sizes");
    return py::tuple(py::cast(sizes));
  });
  c.def("dtype", [](MlirType self) {
    return torchMlirTorchNonValueTensorTypeGetDtype(self);
  });
}

void PyTorch_ValueTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> sizes, MlirType dtype,
         DefaultingPyMlirContext &context) {
        return PyTorch_ValueTensorType(sizes, dtype, context.get());
      },
      "sizes"_a, "dtype"_a, py::kw_only(), "context"_a = py::none());
  c.def_static(
      "get_with_least_static_information",
      [](DefaultingPyMlirContext &context) {
        return PyTorch_ValueTensorType::getWithLeastStaticInformation(
            context.get());
      },
      py::kw_only(), "context"_a = py::none());
  c.def("sizes", [](MlirType self) {
    std::vector<int64_t> sizes(torchMlirTorchValueTensorTypeGetRank(self));
    if (torchMlirTorchValueTensorTypeGetSizes(self, sizes.data()))
      throw py::value_error("no sizes");
    return py::tuple(py::cast(sizes));
  });
  c.def("dtype", [](MlirType self) {
    return torchMlirTorchValueTensorTypeGetDtype(self);
  });
}

PyAnyTorchTensorType
PyAnyTorchTensorType::getWithLeastStaticInformation(PyMlirContext *context) {
  MlirType nonValueTensorType =
      torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
          context->get());
  return {context->getRef(), nonValueTensorType};
}

void PyAnyTorchTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> sizes, MlirType dtype,
         DefaultingPyMlirContext &context) {
        return PyAnyTorchTensorType(std::move(sizes), dtype, context.get());
      },
      "sizes"_a, "dtype"_a, py::kw_only(), "context"_a = py::none());
  c.def_static(
      "get_with_least_static_information",
      [](DefaultingPyMlirContext &context) {
        return PyAnyTorchTensorType::getWithLeastStaticInformation(
            context.get());
      },
      py::kw_only(), "context"_a = py::none());
}

// value binders

void PyAnyTorchOptionalTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::none>(), "value"_a);
  py::implicitly_convertible<py::none, PyAnyTorchOptionalTensorValue>();
}

void PyTorch_NonValueTensorValue::bindDerived(ClassTy &c) {}

void PyTorch_ValueTensorValue::bindDerived(ClassTy &c) {}

void PyAnyTorchTensorValue::bindDerived(ClassTy &c) {

  // @overload __add__(self, other Tensor) -> Tensor
  // aten::__add__.Tensor : (Tensor, Tensor) -> (Tensor)
  c.def(
      "__add__",
      [](const PyAnyTorchTensorValue &self,
         const PyAnyTorchTensorValue &other) -> PyAnyTorchTensorValue {
        auto loc = getValueLocation(self);
        return add(self, other, 1.0f, &loc,
                   &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);

  c.def(
      "__iadd__",
      [](const PyAnyTorchTensorValue &self,
         const PyAnyTorchTensorValue &other) -> PyAnyTorchTensorValue {
        auto loc = getValueLocation(self);
        return add(self, other, 1.0f, &loc,
                   &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);

  c.def(
      "__sub__",
      [](const PyAnyTorchTensorValue &self,
         const PyAnyTorchTensorValue &other) -> PyAnyTorchTensorValue {
        auto loc = getValueLocation(self);
        return sub(self, other, 1.0f, &loc,
                   &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);

  // @overload view(self, dtype _dtype) -> Tensor
  c.def(
      "view",
      [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype,
         DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) {
        return view_copy(self, dtype, loc.get(), ip.get());
      },
      "dtype"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

  // @overload view(self, size Sequence[Union[_int, SymInt]]) -> Tensor
  c.def(
      "view",
      [](PyAnyTorchTensorValue &self, const PyAnyTorchListOfTorchIntValue &size,
         DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) {
        return view(self, size, loc.get(), ip.get());
      },
      "size"_a, py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

  // @overload view(self, *size _int) -> Tensor
  c.def(
      "view",
      [](PyAnyTorchTensorValue &self, const py::args &size,
         DefaultingPyLocation &loc, const DefaultingPyInsertionPoint &ip) {
        return view(self, PyAnyTorchListOfTorchIntValue(size), loc.get(),
                    ip.get());
      },
      // When combining *args or **kwargs with Keyword arguments you should not
      // include py::arg tags for the py::args and py::kwargs arguments.
      py::kw_only(), "loc"_a = py::none(), "ip"_a = py::none());

  // aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)
  c.def(
      "to",
      [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &dtype,
         const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy,
         const PyAnyTorchOptionalIntValue &memory_format,
         DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        return to(self, dtype, non_blocking, copy, memory_format, loc.get(),
                  ip.get());
      },
      "dtype"_a = py::none(), "non_blocking"_a = false, "copy"_a = false,
      "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());

  // chunk(self, chunks: _int, dim: _int=0) -> List[Tensor]
  // aten::chunk : (Tensor, int, int) -> (Tensor[])
  c.def(
      "chunk",
      [](const PyAnyTorchTensorValue &self, const PyTorch_IntValue &chunks,
         const PyTorch_IntValue &dim, DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchListOfTensorValue {
        return chunk(self, chunks, dim, loc.get(), ip.get());
      },
      "chunks"_a, "dim"_a = 0, py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());

  // __truediv__(self, other Any) -> Tensor
  // aten::div.Scalar : (Tensor, Scalar) -> (Tensor)
  c.def(
      "__truediv__",
      [](const PyAnyTorchTensorValue &self,
         PyAnyTorchScalarValue &other) -> PyAnyTorchTensorValue {
        auto loc = getValueLocation(self);
        return div(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);

  // aten::div.Tensor : (Tensor, Tensor) -> (Tensor)
  c.def(
      "__truediv__",
      [](const PyAnyTorchTensorValue &self,
         PyAnyTorchTensorValue &other) -> PyAnyTorchTensorValue {
        auto loc = getValueLocation(self);
        return div(self, other, &loc, &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);

  // __rtruediv__(self, other Any) -> Tensor
  c.def(
      "__rtruediv__",
      [](const PyAnyTorchTensorValue &self,
         PyAnyTorchScalarValue &other) -> PyAnyTorchTensorValue {
        auto loc = getValueLocation(self);
        PyAnyTorchTensorValue recip =
            reciprocal(self, &loc, &DefaultingPyInsertionPoint::resolve());
        auto recip_loc = getValueLocation(recip);
        return mul(recip, other, &recip_loc,
                   &DefaultingPyInsertionPoint::resolve());
      },
      "other"_a);

  // __getitem__(self, indices: Union[None, _int, slice, Tensor, List, Tuple])
  // -> Tensor
  // __getitem__(self, int) -> Tensor
  c.def(
      "__getitem__",
      [](const PyAnyTorchTensorValue &self,
         const PyTorch_IntValue &index) -> PyAnyTorchTensorValue {
        auto loc = getValueLocation(self);
        return select(self, 0, index, &loc,
                      &DefaultingPyInsertionPoint::resolve());
      },
      "index"_a);

  // __getitem__(self, None) -> Tensor
  c.def("__getitem__",
        [](const PyAnyTorchTensorValue &self,
           const py::none &noneValue) -> PyAnyTorchTensorValue {
          auto loc = getValueLocation(self);
          return unsqueeze(self, 0, &loc,
                           &DefaultingPyInsertionPoint::resolve());
        });

  // __getitem__(self, slice) -> Tensor
  c.def("__getitem__",
        [](const PyAnyTorchTensorValue &self,
           py::slice &sliceObject) -> PyAnyTorchTensorValue {
          int dim = 0;

          auto parseAttr =
              [](const py::object &obj) -> PyAnyTorchOptionalIntValue {
            if (py::isinstance<py::none>(obj)) {
              return obj.cast<PyAnyTorchOptionalIntValue>();
            } else if (py::isinstance<py::int_>(obj)) {
              return obj.cast<int>();
            } else {
              throw std::invalid_argument(
                  "Invalid: aten.slice.Tensor expects either an integer or "
                  "None type as indices");
            }
          };

          PyAnyTorchOptionalIntValue start =
              parseAttr(getattr(sliceObject, "start"));
          PyAnyTorchOptionalIntValue stop =
              parseAttr(getattr(sliceObject, "stop"));
          py::object step_attr = getattr(sliceObject, "step");
          PyTorch_IntValue step =
              py::isinstance<py::none>(step_attr) ? 1 : step_attr.cast<int>();

          auto loc = getValueLocation(step);

          return slice(self, dim, start, stop, step, &loc,
                       &DefaultingPyInsertionPoint::resolve());
        });

  // @overload reshape(self, shape: Sequence[Union[_int, SymInt]]) -> Tensor
  // aten::reshape : (Tensor, int...) -> (Tensor)
  c.def("reshape",
        [](const PyAnyTorchTensorValue &self,
           const py::args &args) -> PyAnyTorchTensorValue {
          auto shape = PyAnyTorchListOfTorchIntValue(args);
          auto loc = getValueLocation(shape);
          return reshape(self, shape, &loc,
                         &DefaultingPyInsertionPoint::resolve());
        });

  // @overload double(self) -> Tensor
  c.def(
      "double",
      [](const PyAnyTorchTensorValue &self,
         const PyTorch_BoolValue &non_blocking, const PyTorch_BoolValue &copy,
         const PyAnyTorchOptionalIntValue &memory_format,
         const DefaultingPyLocation &loc,
         const DefaultingPyInsertionPoint &ip) -> PyAnyTorchTensorValue {
        const PyTorch_IntValue dtype = 7;
        return to(self, dtype, non_blocking, copy, memory_format, loc.get(),
                  ip.get());
      },
      "non_blocking"_a = false, "copy"_a = false,
      "memory_format"_a = py::none(), py::kw_only(), "loc"_a = py::none(),
      "ip"_a = py::none());

#include "TorchTensor.pybinds.cpp"
}

PyAnyTorchListOfTensorValue mapListToTorchListOfTensorValue(const py::list &l) {
  if (l.empty())
    throw py::value_error("can't map empty list of tensor values");
  MlirValue element = l[0].cast<PyValue>().get();
  MlirType elType = mlirValueGetType(element);
  auto context = PyMlirContext::forContext(mlirTypeGetContext(elType));
  auto t = PyAnyTorchListOfTensorType(elType, context.get());
  return py::cast(
             PyAnyTorchListValue(py::cast(t), l, tag<PyAnyTorchTensorValue>{}))
      .cast<PyAnyTorchListOfTensorValue>();
}

void PyAnyTorchListOfTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::list>(), "value"_a);
  c.def(py::init<py::tuple>(), "value"_a);
  py::implicitly_convertible<py::list, PyAnyTorchListOfTensorValue>();
  py::implicitly_convertible<py::tuple, PyAnyTorchListOfTensorValue>();
  c.def("__len__", [](const PyAnyTorchListValue &self) { return self.length; });
  c.def("__iter__", [](const PyAnyTorchListOfTensorValue &self) {
    return py::iter(py::cast(makeListIter<PyAnyTorchTensorValue>(
        self, &DefaultingPyLocation::resolve(),
        &DefaultingPyInsertionPoint::resolve())));
  });
  c.def(
      "__getitem__",
      [](const PyAnyTorchListOfTensorValue &self,
         const PyTorch_IntValue &idx) -> PyAnyTorchTensorValue {
        return makeGetItem<PyAnyTorchTensorValue>(
            self, idx, &DefaultingPyLocation::resolve(),
            &DefaultingPyInsertionPoint::resolve());
      },
      "idx"_a);
}

PyAnyTorchListOfOptionalTensorValue
mapListToTorchListOfOptionalTensorValue(const py::list &l) {
  MlirValue element;
  if (l[0].is_none())
    element = PyTorch_NoneValue().get();
  else
    element = l[0].cast<PyValue>().get();
  MlirType elType = mlirValueGetType(element);
  auto context = PyMlirContext::forContext(mlirTypeGetContext(elType));
  return (l.empty() ||
          std::all_of(l.begin(), l.end(), [](auto o) { return o.is_none(); }))
             ? py::cast(PyAnyTorchListValue(l))
                   .cast<PyAnyTorchListOfOptionalTensorValue>()
             : py::cast(PyAnyTorchListValue(
                            py::cast(PyAnyTorchListOfOptionalTensorType(
                                elType, context.get())),
                            l, tag<PyAnyTorchTensorValue>{}))
                   .cast<PyAnyTorchListOfOptionalTensorValue>();
}

void PyAnyTorchListOfOptionalTensorValue::bindDerived(ClassTy &c) {
  c.def(py::init<py::list>(), "value"_a);
  c.def(py::init<py::tuple>(), "value"_a);
  py::implicitly_convertible<py::list, PyAnyTorchListOfOptionalTensorValue>();
  py::implicitly_convertible<py::tuple, PyAnyTorchListOfOptionalTensorValue>();
}

void populateTorchTensorOps(py::module &m) {
  // bind types
  PyTorch_NonValueTensorType::bind(m);
  PyTorch_ValueTensorType::bind(m);
  PyAnyTorchTensorType::bind(m);
  PyAnyTorchOptionalTensorType::bind(m);
  PyAnyTorchListOfOptionalTensorType::bind(m);
  PyAnyTorchListOfTensorType::bind(m);

  // bind values
  PyAnyTorchTensorValue::bind(m);
  PyAnyTorchListOfOptionalTensorValue::bind(m);
  PyAnyTorchListOfTensorValue::bind(m);
  PyAnyTorchOptionalTensorValue::bind(m);
  PyTorch_NonValueTensorValue::bind(m);
  PyTorch_ValueTensorValue::bind(m);
}
} // namespace mlir::torch
