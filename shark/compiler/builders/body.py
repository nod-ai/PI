from __future__ import annotations

from typing import Callable, Union

from shark.ir import (
    Type as MLIRType,
    Value as MLIRValue,
    Location,
    Attribute,
    IntegerAttr,
    IntegerType,
    Context,
    F64Type,
    MemRefType,
    AffineExpr,
    AffineMap,
)
from shark.dialects import arith, linalg, math, memref, affine_ as affine
from shark.dialects.linalg import ScalarAssign, ScalarExpression, FunctionKind
from shark.dialects.linalg.opdsl.lang.emitter import (
    _is_integer_type,
    _is_floating_point_type,
    _is_index_type,
    _get_floating_point_width,
    _is_complex_type,
)


class BodyBuilder:
    """Constructs a structured op body by evaluating assignments."""

    def __init__(
        self,
        type_mapping: dict[str, MLIRType],
        block_arg_mapping: dict[str, MLIRValue],
        fn_attr_mapping: dict[str, str],
        context: Context,
        location: Location,
    ):
        self.type_mapping = type_mapping
        self.block_arg_mapping = block_arg_mapping
        self.fn_attr_mapping = fn_attr_mapping
        self.yield_mapping = dict()
        self.context = context
        self.location = location

    def get_affine_expr_constant(self, c):
        return AffineExpr.get_constant(c)

    def get_affine_map_constant(self, c):
        return AffineMap.get_constant(c)

    def get_int_attr(self, bits: int, value: Union[IntegerAttr, int]) -> IntegerAttr:
        """Converts the given value to signless integer attribute of given bit width."""
        if isinstance(value, int):
            ty = IntegerType.get_signless(bits)
            return IntegerAttr.get(ty, value)
        else:
            return value

    def constant(
        self,
        py_cst: Union[int, float, bool],
        index_type: bool = False,
    ):
        if index_type:
            constant = arith.ConstantOp.create_index(py_cst).result
        else:
            constant = arith.ConstantOp(infer_mlir_type(py_cst), py_cst).result

        return constant

    def memref_alloc(
        self, dim_sizes: Union[list[int], tuple[int]], el_type: MLIRType
    ) -> MLIRValue:
        res_type = MemRefType.get(dim_sizes, el_type)
        # dim_sizes = [self.constant(dim, True) for dim in dim_sizes]
        # num_dims = self.constant(len(dim_sizes), True)
        res = memref.AllocaOp(res_type, [], []).memref
        return res

    def memref_store(
        self,
        store_value: MLIRValue,
        dst_memref: MLIRValue,
        indices: Union[tuple[MLIRValue], list[MLIRValue]],
    ) -> MLIRValue:
        # value, memref, indices
        return memref.StoreOp(store_value, dst_memref, indices).memref

    def memref_load(
        self,
        src_memref: MLIRValue,
        indices: Union[tuple[MLIRValue], list[MLIRValue]],
    ) -> MLIRValue:
        # result_type, memref, indices
        return memref.LoadOp(src_memref, indices).result

    def affine_store(
        self,
        store_value: MLIRValue,
        dst_memref: MLIRValue,
        indices: Union[tuple[MLIRValue], list[MLIRValue]],
    ) -> MLIRValue:
        # value, memref, indices
        return affine.AffineStoreOp(store_value, dst_memref, indices).memref

    def affine_load(
        self,
        src_memref: MLIRValue,
        indices: Union[tuple[MLIRValue], list[MLIRValue]],
    ) -> MLIRValue:
        # result_type, memref, indices
        return affine.AffineLoadOp(src_memref, indices).result

    def assign(self, assignment: ScalarAssign):
        if assignment.arg in self.yield_mapping:
            raise ValueError(
                f"Multiple assignments to the same argument are forbidden: "
                f"{assignment}"
            )
        self.yield_mapping[assignment.arg] = self.expression(assignment.value)

    def expression(self, expr: ScalarExpression) -> MLIRValue:
        if expr.scalar_arg:
            try:
                return self.block_arg_mapping[expr.scalar_arg.arg]
            except KeyError:
                raise ValueError(
                    f"Argument {expr.scalar_arg.arg} is not bound for "
                    f"this structured op."
                )
        elif expr.scalar_const:
            value_attr = Attribute.parse(expr.scalar_const.value)
            return arith.ConstantOp(value_attr.type, value_attr).result
        elif expr.scalar_index:
            dim_attr = IntegerAttr.get(
                IntegerType.get_signless(64), expr.scalar_index.dim
            )
            return linalg.IndexOp(dim_attr).result
        elif expr.scalar_fn:
            kind = expr.scalar_fn.kind.name.lower()
            fn_name = expr.scalar_fn.fn_name
            if expr.scalar_fn.attr_name:
                fn_name, _ = self.fn_attr_mapping[expr.scalar_fn.attr_name]
            fn = self.get_function(f"_{kind}_{fn_name}")
            operand_values = [
                self.expression(operand) for operand in expr.scalar_fn.operands
            ]
            if expr.scalar_fn.kind == FunctionKind.TYPE:
                operand_values = [expr.scalar_fn.type_var.name] + operand_values
            return fn(*operand_values)
        raise NotImplementedError(f"Unimplemented scalar body expression: {expr}")

    def yield_outputs(self, *output_names: str):
        output_values = []
        for n in output_names:
            try:
                output_values.append(self.yield_mapping[n])
            except KeyError:
                raise ValueError(
                    f"Body assignments do not assign all outputs: " f"missing '{n}'"
                )
        linalg.YieldOp(output_values)

    def get_function(self, fn_name: str) -> Callable:
        try:
            fn = getattr(self, f"{fn_name}")
        except AttributeError:
            raise ValueError(f"Function '{fn_name}' is not a known function")
        return fn

    def cast(
        self, type_var_name: str, operand: MLIRValue, is_unsigned_cast: bool = False
    ) -> MLIRValue:
        try:
            to_type = self.type_mapping[type_var_name]
        except KeyError:
            raise ValueError(
                f"Unbound type variable '{type_var_name}' ("
                f"expected one of {self.type_mapping.keys()}"
            )
        if operand.type == to_type:
            return operand
        if _is_integer_type(to_type):
            return self.cast_to_integer(to_type, operand, is_unsigned_cast)
        elif _is_floating_point_type(to_type):
            return self.cast_to_floating_point(to_type, operand, is_unsigned_cast)

    def cast_to_integer(
        self, to_type: MLIRType, operand: MLIRValue, is_unsigned_cast: bool
    ) -> MLIRValue:
        to_width = IntegerType(to_type).width
        operand_type = operand.type
        if _is_floating_point_type(operand_type):
            if is_unsigned_cast:
                return arith.FPToUIOp(to_type, operand, loc=self.location).result
            return arith.FPToSIOp(to_type, operand, loc=self.location).result
        if _is_index_type(operand_type):
            return arith.IndexCastOp(to_type, operand, loc=self.location).result
        # Assume integer.
        from_width = IntegerType(operand_type).width
        if to_width > from_width:
            if is_unsigned_cast:
                return arith.ExtUIOp(to_type, operand, loc=self.location).result
            return arith.ExtSIOp(to_type, operand, loc=self.location).result
        elif to_width < from_width:
            return arith.TruncIOp(to_type, operand, loc=self.location).result
        raise ValueError(
            f"Unable to cast body expression from {operand_type} to " f"{to_type}"
        )

    def cast_to_floating_point(
        self, to_type: MLIRType, operand: MLIRValue, is_unsigned_cast: bool
    ) -> MLIRValue:
        operand_type = operand.type
        if _is_integer_type(operand_type):
            if is_unsigned_cast:
                return arith.UIToFPOp(to_type, operand, loc=self.location).result
            return arith.SIToFPOp(to_type, operand, loc=self.location).result
        # Assume FloatType.
        to_width = _get_floating_point_width(to_type)
        from_width = _get_floating_point_width(operand_type)
        if to_width > from_width:
            return arith.ExtFOp(to_type, operand, loc=self.location).result
        elif to_width < from_width:
            return arith.TruncFOp(to_type, operand, loc=self.location).result
        raise ValueError(
            f"Unable to cast body expression from {operand_type} to " f"{to_type}"
        )

    def type_cast_signed(self, type_var_name: str, operand: MLIRValue) -> MLIRValue:
        return self.cast(type_var_name, operand, False)

    def type_cast_unsigned(self, type_var_name: str, operand: MLIRValue) -> MLIRValue:
        return self.cast(type_var_name, operand, True)

    def unary_exp(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.ExpOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'exp' operand: {x}")

    def unary_log(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.LogOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'log' operand: {x}")

    def unary_abs(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.AbsFOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'abs' operand: {x}")

    def unary_ceil(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.CeilOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'ceil' operand: {x}")

    def unary_floor(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.FloorOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'floor' operand: {x}")

    def unary_negf(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return arith.NegFOp(x, loc=self.location).result
        if _is_complex_type(x.type):
            return complex.NegOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'negf' operand: {x}")

    def binary_add(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.AddFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            if _is_index_type(lhs.type):
                lhs = self.cast_to_integer(
                    IntegerType.get_signed(64), lhs, is_unsigned_cast=True
                )
            if _is_index_type(rhs.type):
                rhs = self.cast_to_integer(
                    IntegerType.get_signed(64), rhs, is_unsigned_cast=True
                )
            return arith.AddIOp(lhs, rhs, loc=self.location).result
        if _is_complex_type(lhs.type):
            return complex.AddOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'add' operands: {lhs}, {rhs}")

    def binary_sub(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.SubFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.SubIOp(lhs, rhs, loc=self.location).result
        if _is_complex_type(lhs.type):
            return complex.SubOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'sub' operands: {lhs}, {rhs}")

    def binary_mul(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MulFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MulIOp(lhs, rhs, loc=self.location).result
        if _is_complex_type(lhs.type):
            return complex.MulOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'mul' operands: {lhs}, {rhs}")

    def binary_max_signed(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MaxFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MaxSIOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'max' operands: {lhs}, {rhs}")

    def binary_max_unsigned(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MaxFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MaxUIOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'max_unsigned' operands: {lhs}, {rhs}")

    def binary_min_signed(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MinFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MinSIOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'min' operands: {lhs}, {rhs}")

    def binary_min_unsigned(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MinFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MinUIOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'min_unsigned' operands: {lhs}, {rhs}")

    def __getitem__(self, bin_op: str):
        return {
            "Multiply": self.binary_mul,
            "Add": self.binary_add,
            "Subtract": self.binary_sub,
        }[bin_op]


def infer_mlir_type(py_val) -> MLIRType:
    if isinstance(py_val, int):
        # return IntegerType.get_signed(64)
        return IntegerType.get_signless(64)
    elif isinstance(py_val, float):
        return F64Type.get()
    else:
        raise Exception(f"unsupported val type {type(py_val)} {py_val}")
