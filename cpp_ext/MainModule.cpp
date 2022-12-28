#include "IRModule.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "TorchTypes.h"
#include "TensorValue.h"
#include "TorchTypesCAPI.h"

#include <iostream>

namespace py = pybind11;
using namespace mlir::python;

// no clue why but without this i get a missing symbol error
namespace llvm {
int DisableABIBreakingChecks = 1;
int EnableABIBreakingChecks = 0;
}// namespace llvm


PYBIND11_MODULE(_mlir, m) {
  bindValues(m);
  bindTypes(m);
  bindTypeHelpers(m);
}
