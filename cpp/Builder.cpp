//
// Created by mlevental on 11/28/22.
//

#include "Builder.h"
#include "IRModule.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

//#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace mlir;
using namespace mlir::python;
namespace py = pybind11;
//------------------------------------------------------------------------------
// PyBuilder
//------------------------------------------------------------------------------

void mlir::python::populateSharkPy(pybind11::module &m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

//  py::class_<mlir::Type>(m, "type")
//      .def("is_integer", &mlir::Type::isInteger)
//      .def("is_fp16", &mlir::Type::isF16)
//      .def("__str__", [](mlir::Type &self) {
//        std::string str;
//        llvm::raw_string_ostream os(str);
//        self.print(os);
//        return os.str();
//      });
//
//  py::class_<mlir::FunctionType>(m, "function_type")
//      .def("param_types", [](mlir::FunctionType &self) {
//        return std::vector<mlir::Type>(self.getInputs().begin(),
//                                       self.getInputs().end());
//      });

  py::class_<mlir::Value>(m, "value")
      .def("set_attr",
           [](mlir::Value &self, std::string &name,
              mlir::Attribute &attr) -> void {
             if (mlir::Operation *definingOp = self.getDefiningOp())
               definingOp->setAttr(name, attr);
             else {
               /* issue a warning */
             }
           })
      .def("replace_all_uses_with",
           [](mlir::Value &self, mlir::Value &newValue) {
             self.replaceAllUsesWith(newValue);
           });

  py::class_<mlir::BlockArgument, mlir::Value>(m, "block_argument");

  py::class_<mlir::Region>(m, "region")
      .def("get_parent_region", &mlir::Region::getParentRegion, ret::reference)
      .def("size", [](mlir::Region &self) { return self.getBlocks().size(); })
      .def("empty", &mlir::Region::empty);

  py::class_<mlir::Block>(m, "block")
      .def("arg",
           [](mlir::Block &self, int index) -> mlir::BlockArgument {
             return self.getArgument(index);
           })
      .def("get_num_arguments", &mlir::Block::getNumArguments)
      .def("dump", &mlir::Block::dump)
      .def("move_before", &mlir::Block::moveBefore)
      .def("insert_before", &mlir::Block::insertBefore)
      .def("get_parent", &mlir::Block::getParent, ret::reference)
      .def("merge_block_before",
           [](mlir::Block &self, mlir::Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with", [](mlir::Block &self, mlir::Value &v, mlir::Value &newVal) {
        v.replaceUsesWithIf(newVal, [&](mlir::OpOperand &operand) {
          mlir::Operation *user = operand.getOwner();
          mlir::Block *currentBlock = user->getBlock();
          while (currentBlock) {
            if (currentBlock == &self)
              return true;
            // Move up one level
            currentBlock = currentBlock->getParent()->getParentOp()->getBlock();
          }
          return false;
        });
      });

  // using eattr = ir::attribute_kind_t;
  // py::enum_<eattr>(m, "attribute_kind")
  //     .value("readonly", eattr::readonly)
  //     .value("writeonly", eattr::writeonly)
  //     .value("noalias", eattr::noalias)
  //     .value("aligned", eattr::aligned)
  //     .value("multiple_of", eattr::multiple_of)
  //     .value("retune", eattr::retune)
  //     .value("not_implemented", eattr::not_implemented);

  py::class_<mlir::Attribute>(m, "attribute");
  py::class_<mlir::IntegerAttr, mlir::Attribute>(m, "integer_attr");
  py::class_<mlir::BoolAttr, mlir::Attribute>(m, "bool_attr");

  // Ops
  py::class_<mlir::OpState>(m, "OpState")
      .def("set_attr",
           [](mlir::OpState &self, std::string &name,
              mlir::Attribute &attr) -> void { self->setAttr(name, attr); })
      .def(
          "get_num_results",
          [](mlir::OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](mlir::OpState &self, unsigned idx) -> mlir::Value {
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](mlir::OpState &self, unsigned idx) -> mlir::Region & {
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](mlir::scf::ForOp &self, unsigned idx) -> mlir::Block * {
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](mlir::OpState &self) { self->dump(); })
      .def("__str__",
           [](mlir::OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self->print(os);
             return str;
           })
      .def("append_operand",
           [](mlir::OpState &self, mlir::Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify", [](mlir::OpState &self) -> bool {
        return mlir::succeeded(mlir::verify(self.getOperation()));
      });
  // scf Ops
  py::class_<mlir::scf::ForOp, mlir::OpState>(m, "ForOp")
      .def("get_induction_var", &mlir::scf::ForOp::getInductionVar);

  py::class_<mlir::scf::IfOp, mlir::OpState>(m, "IfOp")
      .def("get_then_block", &mlir::scf::IfOp::thenBlock, ret::reference)
      .def("get_else_block", &mlir::scf::IfOp::elseBlock, ret::reference)
      .def("get_then_yield", &mlir::scf::IfOp::thenYield)
      .def("get_else_yield", &mlir::scf::IfOp::elseYield);
  py::class_<mlir::scf::YieldOp, mlir::OpState>(m, "YieldOp");
  py::class_<mlir::scf::WhileOp, mlir::OpState>(m, "WhileOp")
      .def("get_before", &mlir::scf::WhileOp::getBefore, ret::reference)
      .def("get_after", &mlir::scf::WhileOp::getAfter, ret::reference);
  py::class_<mlir::scf::ConditionOp, mlir::OpState>(m, "ConditionOp");

  // dynamic_attr is used to transfer ownership of the MLIR context to the
  // module
  py::class_<mlir::ModuleOp, mlir::OpState>(m, "module", py::dynamic_attr())
      .def("dump", &mlir::ModuleOp::dump)
      .def("str",
           [](mlir::ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("push_back",
           [](mlir::ModuleOp &self, func::FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("has_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> func::FuncOp {
             return self.lookupSymbol<func::FuncOp>(funcName);
           })
      .def("get_single_function", [](mlir::ModuleOp &self) -> func::FuncOp {
        llvm::SmallVector<func::FuncOp> funcs;
        self.walk([&](func::FuncOp func) { funcs.push_back(func); });
        if (funcs.size() != 1)
          throw std::runtime_error("Expected a single function");
        return funcs[0];
      });

  py::class_<func::FuncOp, mlir::OpState>(m, "function")
      // .def_property_readonly("attrs", &ir::function::attrs)
      // .def("add_attr", &ir::function::add_attr);
      .def("args",
           [](func::FuncOp &self, unsigned idx) -> mlir::BlockArgument {
             return self.getArgument(idx);
           })
      .def(
          "add_entry_block",
          [](func::FuncOp &self) -> mlir::Block * {
            return self.addEntryBlock();
          },
          ret::reference)
      .def(
          "set_arg_attr",
          [](func::FuncOp &self, int arg_no, const std::string &name, int val) {
            // set arg attributes "name" to value "val"
            auto attrTy = mlir::IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, mlir::IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      .def_property_readonly("type", &func::FuncOp::getFunctionTypeAttr)
      .def("reset_type", &func::FuncOp::setType);

  py::class_<mlir::OpBuilder>(m, "SharkBuilder", py::dynamic_attr())
      .def(py::init<>([](PyMlirContext &context) {
        auto *mlirContext = unwrap<mlir::MLIRContext, MlirContext>(context.get());
        mlir::OpBuilder builder(mlirContext);
        return builder;
      }))
      // // getters
      .def_property_readonly("context", &mlir::OpBuilder::getContext,
                             ret::reference)
      .def("create_module",
           [](mlir::OpBuilder &self) {
             auto loc = self.getUnknownLoc();
             //             MlirModule module = mlirModuleCreateEmpty(loc);
//             auto module = self.create<mlir::ModuleOp>(loc);
//             auto *mlirModule = wrap<MlirModule, mlir::ModuleOp>(context.get());
//             return PyModule::forModule(module).releaseObject();
             return self.create<mlir::ModuleOp>(loc);
           })
      .def("ret",
           [](mlir::OpBuilder &self, std::vector<mlir::Value> &vals) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::func::ReturnOp>(loc, vals);
           })
      .def("call",
           [](mlir::OpBuilder &self, func::FuncOp &func,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             auto loc = self.getUnknownLoc();
             return self.create<func::CallOp>(loc, func, args);
           })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](mlir::OpBuilder &self, mlir::Block &block) -> void {
             self.setInsertionPointToStart(&block);
           })
      .def("set_insertion_point_to_end",
           [](mlir::OpBuilder &self, mlir::Block &block) {
             self.setInsertionPointToEnd(&block);
           })
      .def(
          "get_insertion_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            return self.getInsertionBlock();
          },
          ret::reference)
      .def("get_insertion_point", &mlir::OpBuilder::saveInsertionPoint)
      .def("restore_insertion_point", &mlir::OpBuilder::restoreInsertionPoint)
      // .def("set_insert_point", [](ir::builder *self,
      // std::pair<ir::basic_block*, ir::instruction*> pt) {
      //   ir::basic_block *bb = pt.first;
      //   ir::instruction *instr = pt.second;
      //   if (instr) {
      //     if (bb != instr->get_parent())
      //       throw std::runtime_error("invalid insertion point, instr not in
      //       bb");
      //     self->set_insert_point(instr);
      //   } else {
      //     assert(bb);
      //     self->set_insert_point(bb);
      //   }
      // })
      // Attr
      .def("get_bool_attr", &mlir::OpBuilder::getBoolAttr)
      .def("get_int32_attr", &mlir::OpBuilder::getI32IntegerAttr)
      // Use arith.ConstantOp to create constants
      // Constants
      .def("get_int1",
           [](mlir::OpBuilder &self, bool v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI1Type()));
           })
      .def("get_int32",
           [](mlir::OpBuilder &self, int64_t v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI32Type()));
           })
      // .def("get_uint32", &ir::builder::get_int32, ret::reference)
      // .def("get_int64", [](ir::builder *self, int64_t v) { return
      // self->get_int64((uint64_t)v); }, ret::reference) .def("get_uint64",
      // &ir::builder::get_int64, ret::reference) .def("get_float16",
      // &ir::builder::get_float16, ret::reference)
      .def("get_float32",
           [](mlir::OpBuilder &self, float v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::ConstantOp>(
                 loc, self.getF32FloatAttr(v));
           })
      .def("get_null_value",
           [](mlir::OpBuilder &self, mlir::Type type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             if (auto floatTy = type.dyn_cast<mlir::FloatType>())
               return self.create<mlir::arith::ConstantFloatOp>(
                   loc, mlir::APFloat(floatTy.getFloatSemantics(), 0), floatTy);
             else if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(loc, 0, intTy);
             else
               throw std::runtime_error("Not implemented");
           })
      .def("get_all_ones_value",
           [](mlir::OpBuilder &self, mlir::Type type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             uint64_t val = 0xFFFFFFFFFFFFFFFF;
             if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(loc, val, intTy);
             else
               throw std::runtime_error("Not implemented");
           })

      // Types
      .def("get_void_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getNoneType();
           })
      .def("get_int1_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getI1Type();
           })// or ret::copy?
      .def("get_int8_ty",
           [](mlir::OpBuilder &self) -> mlir::Type { return self.getI8Type(); })
      .def("get_int16_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getType<mlir::IntegerType>(16);
           })
      .def(
          "get_int32_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getI32Type(); })
      .def(
          "get_int64_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getI64Type(); })
      //      .def("get_fp8_ty",
      //           [](mlir::OpBuilder &self) -> mlir::Type {
      //             return self.getType<mlir::triton::Float8Type>();
      //           })
      .def(
          "get_half_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF16Type(); })
      .def("get_bf16_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getBF16Type();
           })
      .def(
          "get_float_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF32Type(); })
      .def(
          "get_double_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF64Type(); })
      //      .def("get_ptr_ty",
      //           [](mlir::OpBuilder &self, mlir::Type &type,
      //              int addrSpace) -> mlir::Type {
      //             return mlir::triton::PointerType::get(type, addrSpace);
      //           })
      .def("get_block_ty",
           [](mlir::OpBuilder &self, mlir::Type &elementType,
              std::vector<int64_t> &shape) -> mlir::Type {
             return mlir::RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> inTypes,
              std::vector<mlir::Type> outTypes) -> mlir::Type {
             return self.getFunctionType(inTypes, outTypes);
           })

      // Ops
      .def("get_or_insert_function",
           [](mlir::OpBuilder &self, mlir::ModuleOp &module,
              std::string &funcName, MlirType &funcType,
              std::string &visibility) -> func::FuncOp {
             if (mlir::Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<func::FuncOp>(funcOperation);
             auto loc = self.getUnknownLoc();
             // TODO(max) PyFuctionType isn't exposed grrrrrrrrr
             if (auto funcTy = funcType.dyn_cast<mlir::FunctionType>()) {
               llvm::SmallVector<mlir::NamedAttribute> attrs = {
                   mlir::NamedAttribute(self.getStringAttr("sym_visibility"),
                                        self.getStringAttr(visibility))};
               return self.create<func::FuncOp>(loc, funcName, funcTy, attrs);
             }
             throw std::runtime_error("invalid function type");
           })
      .def(
          "create_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            mlir::Region *parent = self.getBlock()->getParent();
            return self.createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](mlir::OpBuilder &self, mlir::Region &parent,
             std::vector<mlir::Type> &argTypes) -> mlir::Block * {
            auto argLoc = self.getUnknownLoc();
            llvm::SmallVector<mlir::Location, 8> argLocs(argTypes.size(),
                                                         argLoc);
            return self.createBlock(&parent, {}, argTypes, argLocs);
          },
          ret::reference)
      .def(
          "new_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            return new mlir::Block();
          },
          ret::reference)
      // Structured control flow
      .def("create_for_op",
           [](mlir::OpBuilder &self, mlir::Value &lb, mlir::Value &ub,
              mlir::Value &step,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::ForOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::ForOp>(loc, lb, ub, step, initArgs);
           })
      .def("create_if_op",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> &retTypes,
              mlir::Value &condition, bool withElse) -> mlir::scf::IfOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::IfOp>(loc, retTypes, condition,
                                                 withElse);
           })
      .def("create_yield_op",
           [](mlir::OpBuilder &self,
              std::vector<mlir::Value> &yields) -> mlir::scf::YieldOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::YieldOp>(loc, yields);
           })
      .def("create_while_op",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> &retTypes,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::WhileOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::WhileOp>(loc, retTypes, initArgs);
           })
      .def("create_condition_op",
           [](mlir::OpBuilder &self, mlir::Value &cond,
              std::vector<mlir::Value> &args) -> mlir::scf::ConditionOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::ConditionOp>(loc, cond, args);
           })

      // miscellaneous
      //      .def("create_make_range",
      //           [](mlir::OpBuilder &self, int start, int end) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             auto retType =
      //                 mlir::RankedTensorType::get({end - start}, self.getI32Type());
      //             return self.create<mlir::triton::MakeRangeOp>(loc, retType, start,
      //                                                           end);
      //           })
      //      .def("create_get_program_id",
      //           [](mlir::OpBuilder &self, int axis) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::GetProgramIdOp>(
      //                 loc, self.getI32Type(), axis);
      //           })

      // Cast instructions
      // Conversions for custom FP types (FP8)
      //      .def("create_fp_to_fp",
      //           [](mlir::OpBuilder &self, mlir::Value &src,
      //              mlir::Type &dstType) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::FpToFpOp>(loc, dstType, src);
      //           })
      // Conversions for standard LLVM builtin types
      //      .def("create_bitcast",
      //           [](mlir::OpBuilder &self, mlir::Value &src,
      //              mlir::Type &dstType) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::BitcastOp>(loc, dstType, src);
      //           })
      .def("create_si_to_fp",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::SIToFPOp>(loc, dstType, src);
           })
      .def("create_ui_to_fp",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::UIToFPOp>(loc, dstType, src);
           })
      .def("create_fp_to_si",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::FPToSIOp>(loc, dstType, src);
           })
      .def("create_fp_to_ui",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::FPToUIOp>(loc, dstType, src);
           })
      .def("create_fp_ext",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::ExtFOp>(loc, dstType, src);
           })
      .def("create_fp_trunc",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::TruncFOp>(loc, dstType, src);
           })
      .def("create_int_cast",
           [](mlir::OpBuilder &self, mlir::Value &src, mlir::Type &dstType,
              bool isSigned) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             // get element type if necessary
             mlir::Type srcType = src.getType();
             auto srcTensorType = srcType.dyn_cast<mlir::RankedTensorType>();
             auto dstTensorType = dstType.dyn_cast<mlir::RankedTensorType>();
             mlir::Type srcEltType = srcType;
             mlir::Type dstEltType = dstType;
             if (dstTensorType && srcTensorType) {
               dstEltType = dstTensorType.getElementType();
               srcEltType = srcTensorType.getElementType();
             }
             unsigned srcWidth = srcEltType.getIntOrFloatBitWidth();
             unsigned dstWidth = dstEltType.getIntOrFloatBitWidth();
             if (srcWidth == dstWidth)
               return self.create<mlir::arith::BitcastOp>(loc, dstType, src);
             else if (srcWidth > dstWidth)
               return self.create<mlir::arith::TruncIOp>(loc, dstType, src);
             else if (isSigned)
               return self.create<mlir::arith::ExtSIOp>(loc, dstType, src);
             else
               return self.create<mlir::arith::ExtUIOp>(loc, dstType, src);
           })
      .def("create_to_index",
           [](mlir::OpBuilder &self, mlir::Value &input) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::IndexCastOp>(loc,
                                                          self.getIndexType(), input);
           })
      .def("create_index_to_si",
           [](mlir::OpBuilder &self, mlir::Value &input) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::IndexCastOp>(loc,
                                                          self.getI32Type(), input);
           })
      .def("create_fmul",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::MulFOp>(loc, lhs, rhs);
           })
      .def("create_fdiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivFOp>(loc, lhs, rhs);
           })
      .def("create_frem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemFOp>(loc, lhs, rhs);
           })
      .def("create_fadd",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AddFOp>(loc, lhs, rhs);
           })
      .def("create_fsub",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::SubFOp>(loc, lhs, rhs);
           })
      .def("create_mul",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::MulIOp>(loc, lhs, rhs);
           })
      .def("create_sdiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
           })
      .def("create_udiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivUIOp>(loc, lhs, rhs);
           })
      .def("create_srem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
           })
      .def("create_urem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemUIOp>(loc, lhs, rhs);
           })
      .def("create_add",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AddIOp>(loc, lhs, rhs);
           })
      .def("create_sub",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::SubIOp>(loc, lhs, rhs));
           })
      .def("create_shl",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShLIOp>(loc, lhs, rhs));
           })
      .def("create_lshr",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShRUIOp>(loc, lhs, rhs));
           })
      .def("create_ashr",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShRSIOp>(loc, lhs, rhs));
           })
      // AddPtr (similar to GEP)
      //      .def("create_addptr",
      //           [](mlir::OpBuilder &self, mlir::Value &ptr,
      //              mlir::Value &offset) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::AddPtrOp>(loc, ptr.getType(), ptr,
      //                                                        offset);
      //           })
      // Comparison (int)
      .def("create_icmpSLE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
           })
      .def("create_icmpSLT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
           })
      .def("create_icmpSGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
           })
      .def("create_icmpSGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
           })
      .def("create_icmpULE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ule, lhs, rhs);
           })
      .def("create_icmpULT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ult, lhs, rhs);
           })
      .def("create_icmpUGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::uge, lhs, rhs);
           })
      .def("create_icmpUGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs);
           })
      .def("create_icmpEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
           })
      .def("create_icmpNE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
           })
      // Comparison (float)
      .def("create_fcmpOLT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
           })
      .def("create_fcmpOGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
           })
      .def("create_fcmpOLE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
           })
      .def("create_fcmpOGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
           })
      .def("create_fcmpOEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
           })
      .def("create_fcmpONE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs);
           })
      .def("create_fcmpULT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ULT, lhs, rhs);
           })
      .def("create_fcmpUGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UGT, lhs, rhs);
           })
      .def("create_fcmpULE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ULE, lhs, rhs);
           })
      .def("create_fcmpUGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UGE, lhs, rhs);
           })
      .def("create_fcmpUEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UEQ, lhs, rhs);
           })
      .def("create_fcmpUNE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UNE, lhs, rhs);
           })
      // // Logical
      .def("create_and",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AndIOp>(loc, lhs, rhs);
           })
      .def("create_xor",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
           })
      .def("create_or",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::OrIOp>(loc, lhs, rhs);
           })
      // Input/Output
      //      .def("create_load",
      //           [](mlir::OpBuilder &self, mlir::Value &ptrs,
      //              mlir::triton::CacheModifier cacheModifier,
      //              mlir::triton::EvictionPolicy evictionPolicy,
      //              bool isVolatile) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::LoadOp>(
      //                 loc, ptrs, cacheModifier, evictionPolicy, isVolatile);
      //           })
      //      .def("create_store",
      //           [](mlir::OpBuilder &self, mlir::Value &ptrs,
      //              mlir::Value &value) -> void {
      //             auto loc = self.getUnknownLoc();
      //             self.create<mlir::triton::StoreOp>(loc, ptrs, value);
      //           })
      //      .def("create_masked_load",
      //           [](mlir::OpBuilder &self, mlir::Value &ptrs, mlir::Value &mask,
      //              std::optional<mlir::Value> &other,
      //              mlir::triton::CacheModifier cacheModifier,
      //              mlir::triton::EvictionPolicy evictionPolicy,
      //              bool isVolatile) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::LoadOp>(
      //                 loc, ptrs, mask, other.value_or(mlir::Value()), cacheModifier,
      //                 evictionPolicy, isVolatile);
      //           })
      //      .def("create_masked_store",
      //           [](mlir::OpBuilder &self, mlir::Value &ptrs, mlir::Value &val,
      //              mlir::Value &mask) -> void {
      //             auto loc = self.getUnknownLoc();
      //             self.create<mlir::triton::StoreOp>(loc, ptrs, val, mask);
      //           })
      //      .def("create_view",
      //           [](mlir::OpBuilder &self, mlir::Value &arg,
      //              std::vector<int64_t> &shape) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             auto argType = arg.getType()
      //                 .dyn_cast<mlir::RankedTensorType>()
      //                 .getElementType();
      //             return self.create<mlir::triton::ViewOp>(
      //                 loc, mlir::RankedTensorType::get(shape, argType), arg);
      //           })
      //      .def(
      //          "create_expand_dims",
      //          [](mlir::OpBuilder &self, mlir::Value &arg, int axis) -> mlir::Value {
      //            auto loc = self.getUnknownLoc();
      //            auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
      //            auto argEltType = argType.getElementType();
      //            std::vector<int64_t> retShape = argType.getShape();
      //            retShape.insert(retShape.begin() + axis, 1);
      //            return self.create<mlir::triton::ExpandDimsOp>(
      //                loc, mlir::RankedTensorType::get(retShape, argEltType), arg,
      //                axis);
      //          })
      //      .def("create_cat",
      //           [](mlir::OpBuilder &self, mlir::Value &lhs,
      //              mlir::Value &rhs) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             auto lhsType = lhs.getType().dyn_cast<mlir::RankedTensorType>();
      //             auto rhsType = rhs.getType().dyn_cast<mlir::RankedTensorType>();
      //             if (!(lhsType.getShape().size() == 1 &&
      //                 rhsType.getShape().size() == 1))
      //               throw std::runtime_error(
      //                   "shape not supported by cat. Expecting rank-1 inputs");
      //             std::vector<int64_t> shape{lhsType.getShape()[0] +
      //                 rhsType.getShape()[0]};
      //             return self.create<mlir::triton::CatOp>(
      //                 loc,
      //                 mlir::RankedTensorType::get(shape, lhsType.getElementType()),
      //                 lhs, rhs);
      //           })
      //      .def("create_broadcast",
      //           [](mlir::OpBuilder &self, mlir::Value &arg,
      //              std::vector<int64_t> &shape) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             if (auto argType =
      //                 arg.getType().dyn_cast<mlir::RankedTensorType>())
      //               return self.createOrFold<mlir::triton::BroadcastOp>(
      //                   loc,
      //                   mlir::RankedTensorType::get(shape, argType.getElementType()),
      //                   arg);
      //             throw std::runtime_error(
      //                 "arg is not of RankedTensorType, use create_splat");
      //           })
      //      .def("create_splat",
      //           [](mlir::OpBuilder &self, mlir::Value &arg,
      //              std::vector<int64_t> &shape) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             auto argType = arg.getType();
      //             auto ret = self.createOrFold<mlir::triton::SplatOp>(
      //                 loc, mlir::RankedTensorType::get(shape, argType), arg);
      //             return ret;
      //           })
      //          // // atomic
      //      .def("create_atomic_cas",
      //           [](mlir::OpBuilder &self, mlir::Value &ptr, mlir::Value &cmp,
      //              mlir::Value &val) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             mlir::Type dstType;
      //             if (auto srcTensorType = ptr.getType().dyn_cast<mlir::RankedTensorType>()) {
      //               mlir::Type dstElemType = srcTensorType.getElementType()
      //                   .cast<mlir::triton::PointerType>()
      //                   .getPointeeType();
      //               dstType = mlir::RankedTensorType::get(srcTensorType.getShape(),
      //                                                     dstElemType);
      //             } else {
      //               auto ptrType = mlir::getElementTypeOrSelf(ptr)
      //                   .cast<mlir::triton::PointerType>();
      //               dstType = ptrType.getPointeeType();
      //             }
      //             return self.create<mlir::triton::AtomicCASOp>(loc, dstType, ptr,
      //                                                           cmp, val);
      //           })
      //      .def("create_atomic_rmw",
      //           [](mlir::OpBuilder &self, mlir::triton::RMWOp rmwOp,
      //              mlir::Value &ptr, mlir::Value &val,
      //              mlir::Value &mask) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             mlir::Type dstType;
      //             if (auto srcTensorType =
      //                 ptr.getType().dyn_cast<mlir::RankedTensorType>()) {
      //               mlir::Type dstElemType = srcTensorType.getElementType()
      //                   .cast<mlir::triton::PointerType>()
      //                   .getPointeeType();
      //               dstType = mlir::RankedTensorType::get(srcTensorType.getShape(),
      //                                                     dstElemType);
      //             } else {
      //               auto ptrType = mlir::getElementTypeOrSelf(ptr)
      //                   .cast<mlir::triton::PointerType>();
      //               dstType = ptrType.getPointeeType();
      //             }
      //             return self.create<mlir::triton::AtomicRMWOp>(loc, dstType, rmwOp,
      //                                                           ptr, val, mask);
      //           })
      //          // External
      //      .def("create_external_elementwise",
      //           [](mlir::OpBuilder &self, const std::string &libName,
      //              const std::string &libPath, const std::string &symbol,
      //              std::vector<mlir::Value> &argList,
      //              mlir::Type retType) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::ExtElemwiseOp>(
      //                 loc, retType, argList, libName, libPath, symbol);
      //           })
      //          // Built-in instruction
      //      .def("create_get_program_id",
      //           [](mlir::OpBuilder &self, int axis) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::GetProgramIdOp>(
      //                 loc, self.getI32Type(), self.getI32IntegerAttr(axis));
      //           })
      //      .def("create_get_num_programs",
      //           [](mlir::OpBuilder &self, int axis) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::GetNumProgramsOp>(
      //                 loc, self.getI32Type(), self.getI32IntegerAttr(axis));
      //           })
      //      .def("create_dot",
      //           [](mlir::OpBuilder &self, mlir::Value &a, mlir::Value &b,
      //              mlir::Value &c, bool allowTF32, bool transA,
      //              bool transB) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::DotOp>(loc, c.getType(), a, b, c,
      //                                                     allowTF32, transA, transB);
      //           })
      //      .def("create_exp",
      //           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::math::ExpOp>(loc, val);
      //           })
      //      .def("create_cos",
      //           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::math::CosOp>(loc, val);
      //           })
      //      .def("create_sin",
      //           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::math::SinOp>(loc, val);
      //           })
      //      .def("create_log",
      //           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::math::LogOp>(loc, val);
      //           })
      //      .def("create_sqrt",
      //           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::math::SqrtOp>(loc, val);
      //           })
      //      .def("create_reduce",
      //           [](mlir::OpBuilder &self, mlir::Value &operand,
      //              mlir::triton::RedOp redOp, int axis) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             auto inputTensorType =
      //                 operand.getType().dyn_cast<mlir::RankedTensorType>();
      //             std::vector<int64_t> shape = inputTensorType.getShape();
      //             shape.erase(shape.begin() + axis);
      //             bool withIndex = mlir::triton::ReduceOp::withIndex(redOp);
      //             mlir::Type resType = withIndex ? self.getI32Type()
      //                                            : inputTensorType.getElementType();
      //             if (!shape.empty()) {
      //               resType = mlir::RankedTensorType::get(shape, resType);
      //             }
      //             return self.create<mlir::triton::ReduceOp>(loc, resType, redOp,
      //                                                        operand, axis);
      //           })
      //      .def("create_ptr_to_int",
      //           [](mlir::OpBuilder &self, mlir::Value &val,
      //              mlir::Type &type) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::PtrToIntOp>(loc, type, val);
      //           })
      //      .def("create_int_to_ptr",
      //           [](mlir::OpBuilder &self, mlir::Value &val,
      //              mlir::Type &type) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::triton::IntToPtrOp>(loc, type, val);
      //           })
      //      .def("create_select",
      //           [](mlir::OpBuilder &self, mlir::Value &condition,
      //              mlir::Value &trueValue, mlir::Value &falseValue) -> mlir::Value {
      //             auto loc = self.getUnknownLoc();
      //             return self.create<mlir::SelectOp>(loc, condition, trueValue,
      //                                                falseValue);
      //           })
      //      .def("create_printf",
      //           [](mlir::OpBuilder &self, const std::string &prefix,
      //              const std::vector<mlir::Value> &values) -> void {
      //             auto loc = self.getUnknownLoc();
      //             self.create<mlir::triton::PrintfOp>(
      //                 loc,
      //                 mlir::StringAttr::get(self.getContext(),
      //                                       llvm::StringRef(prefix)),
      //                 values);
      //           });
      ;
}
