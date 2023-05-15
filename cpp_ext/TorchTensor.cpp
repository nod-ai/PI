//
// Created by mlevental on 5/15/23.
//

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <exception>
#include <utility>

#include "TorchOps.h"
#include "TorchTensor.h"

namespace py = pybind11;
using namespace mlir::python;

struct NotImplementedError : public std::exception {
  NotImplementedError(std::string msg) : message(std::move(msg)) {}
  [[nodiscard]] const char *what() const noexcept override {
    return message.data();
  }

  std::string message;
};

namespace {
#include "TorchTensor.pybinds_tramps.cpp"
}

namespace mlir::torch {
void PyAnyTorchTensorValue::bindDerived(ClassTy &c) {
#include "TorchTensor.pybinds.cpp"
}

void populateTorchTensorOps(py::module &m) {
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const NotImplementedError &e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    }
  });

  PyAnyTorchTensorValue::bind(m);
}
} // namespace mlir::torch
