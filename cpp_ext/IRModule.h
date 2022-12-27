//===- IRModules.h - IR Submodules of pybind module -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRMODULES_H
#define MLIR_BINDINGS_PYTHON_IRMODULES_H

#include <utility>
#include <vector>
#include <iostream>

#include <mlir-c/Bindings/Python/Interop.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"

namespace py = pybind11;

namespace mlir::python {

class PyOperation;

/// Template for a reference to a concrete type which captures a python
/// reference to its underlying python object.
template<typename T>
class PyObjectRef {
public:
  PyObjectRef(T *referrent, pybind11::object object)
      : referrent(referrent), object(std::move(object)) {
    assert(this->referrent && "cannot construct PyObjectRef with null referrent");
    assert(this->object && "cannot construct PyObjectRef with null object");
  }
  PyObjectRef(PyObjectRef &&other)
      : referrent(other.referrent), object(std::move(other.object)) {
    other.referrent = nullptr;
    assert(!other.object);
  }
  PyObjectRef(const PyObjectRef &other)
      : referrent(other.referrent), object(other.object /* copies */) {}
  ~PyObjectRef() = default;

  T *operator->() {
    assert(referrent && object);
    return referrent;
  }
  pybind11::object getObject() {
    assert(referrent && object);
    return object;
  }
  explicit operator bool() const { return referrent && object; }

private:
  T *referrent;
  pybind11::object object;
};

/// Wrapper around MlirContext.
class PyMlirContext {
public:
  PyMlirContext() = delete;
  PyMlirContext(const PyMlirContext &) = delete;
  PyMlirContext(PyMlirContext &&) = delete;

  explicit PyMlirContext(MlirContext context) : context(context){};

  MlirContext context;
  friend class PyModule;
  friend class PyOperation;
};

using PyMlirContextRef = PyObjectRef<PyMlirContext>;

class BaseContextObject {
public:
  explicit BaseContextObject(PyMlirContextRef ref) : contextRef(std::move(ref)) {
    assert(this->contextRef && "context object constructed with null context ref");
  }
  PyMlirContextRef contextRef;
};

class PyOperation : public BaseContextObject {
public:
  PyOperation &getOperation() { return *this; }
  PyOperation(PyMlirContextRef contextRef, MlirOperation operation) : BaseContextObject(std::move(contextRef)), operation(operation){};

  pybind11::handle handle;
  MlirOperation operation;
  pybind11::object parentKeepAlive;
  bool attached = true;
  bool valid = true;

  friend class PyOperationBase;
  friend class PySymbolTable;
};

using PyOperationRef = PyObjectRef<PyOperation>;

class PyValue {
public:
  PyValue(PyOperationRef parentOperation, MlirValue value)
      : parentOperation(std::move(parentOperation)), value(value) {}
  explicit operator MlirValue() const { return value; }

private:
  PyOperationRef parentOperation;
  MlirValue value;
};

struct PyType : public BaseContextObject {
  PyType(PyMlirContextRef contextRef, MlirType type)
      : BaseContextObject(std::move(contextRef)), type(type) {}
  explicit operator MlirType() const { return type; }
  [[nodiscard]] MlirType get() const { return type; }


  MlirType type;
};



template <typename DerivedTy, typename BaseTy = PyType>
struct PyConcreteType : public BaseTy {
//  using ClassTy = pybind11::class_<DerivedTy, BaseTy>;

  PyConcreteType() = default;
  PyConcreteType(PyMlirContextRef contextRef, MlirType t)
      : BaseTy(std::move(contextRef), t) {}

  static DerivedTy createFromCapsule_(py::capsule& capsule) {
    MlirType rawType = {capsule.get_pointer()};
    if (mlirTypeIsNull(rawType))
      throw py::error_already_set();

    MlirContext ctx = mlirTypeGetContext(rawType);
    auto *unownedContextWrapper = new PyMlirContext(ctx);
    auto pyCtxRef = py::reinterpret_steal<py::object>(mlirPythonContextToCapsule(ctx));
    assert(pyCtxRef && "cast to py::object failed");
    auto ctxRef = PyMlirContextRef(unownedContextWrapper, std::move(pyCtxRef));

    return {std::move(ctxRef), rawType};
  }

};

void populateTorchTypes(py::module &m);

}// namespace mlir::python

#endif// MLIR_BINDINGS_PYTHON_IRMODULES_H
