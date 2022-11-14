#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::python {

std::unique_ptr<OperationPass<func::FuncOp>> createMungeMemrefCopyPass();
std::unique_ptr<OperationPass<func::FuncOp>> createExpandOpsForLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createMungeCallingConventionsPass();

}// namespace mlir::python
