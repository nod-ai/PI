#include "dylib.hpp"
#include "IRModule.h"
#include "TensorValue.h"
#include "TorchTypes.h"
#include "TorchTypesCAPI.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include <iostream>

namespace py = pybind11;
using namespace mlir::python;

// no clue why but without this i get a missing symbol error
namespace llvm {
int DisableABIBreakingChecks = 1;
int EnableABIBreakingChecks = 0;
}// namespace llvm

PYBIND11_MODULE(_mlir, m) {
//  dylib lib1("_torchMlir.cpython-310-darwin.so", dylib::no_filename_decorations);
  dylib lib2("TorchMLIRAggregateCAPI");

//  if (!lib1.has_symbol("mlirValueIsAOpResult"))
//    std::cerr << "symbol 'mlirValueIsAOpResult' not found in '_torchMlir' lib" << std::endl;
  if (!lib2.has_symbol("mlirValueIsAOpResult"))
    std::cerr << "symbol 'mlirValueIsAOpResult' not found in 'TorchMLIRAggregateCAPI' lib" << std::endl;
  else
    std::cerr << "found symbol 'mlirValueIsAOpResult' in 'TorchMLIRAggregateCAPI' lib" << std::endl;

  bindValues(m);
  bindTypes(m);
  bindTypeHelpers(m);
}
