//===- RegisterEverything.cpp - API to register all dialects/passes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Conversion.h"
#include "mlir-c/Transforms.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "RefBackend.h"
//#include "Pass.h"

inline void registerMungeMemrefCopyPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::python::createMungeMemrefCopyPass();
  });
}

inline void registerMungeCallingConventionsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::python::createMungeCallingConventionsPass();
  });
}

inline void registerExpandOpsForLLVMPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::python::createExpandOpsForLLVMPass();
  });
}



PYBIND11_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "MLIR All Upstream Dialects and Passes Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
  });

  // Register all passes on load.
  mlirRegisterAllPasses();
  mlirRegisterConversionPasses();
  mlirRegisterTransformsPasses();
  registerMungeMemrefCopyPass();
  registerExpandOpsForLLVMPass();
  registerMungeCallingConventionsPass();
}
