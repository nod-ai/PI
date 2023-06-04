#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "torch-mlir-c/Dialects.h"
#include "torch-mlir-c/Registration.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

namespace py = pybind11;
using namespace mlir;

#include <iostream>

PYBIND11_MODULE(_site_initialize_0, m) {
  m.def("context_init_hook", [](MlirContext &context) {
    //    torchMlirRegisterAllPasses();
    const char *testingWithTorchMlir =
        std::getenv("TESTING_WITH_UPSTREAM_TORCH_MLIR");
    if (!testingWithTorchMlir) {
      MlirDialectHandle handle = mlirGetDialectHandle__torch__();
      mlirDialectHandleRegisterDialect(handle, context);
      mlirDialectHandleLoadDialect(handle, context);
    }
  });
}
