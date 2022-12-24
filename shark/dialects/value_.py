from typing import Any, List

from torch_mlir.dialects import arith, math
from ._ods_common import get_op_result_or_value
from torch_mlir.dialects.linalg.opdsl.lang.emitter import (
    _is_integer_type,
    _is_floating_point_type,
    _is_index_type,
    _get_floating_point_width,
    _is_complex_type,
)
from torch_mlir.ir import (
    Type as MLIRType,
    Value as MLIRValue,
    IntegerType,
    OpView,
    F64Type,
    IndexType,
    BF16Type,
    F16Type,
    F32Type,
)


def _isa(obj: Any, cls: type):
    try:
        cls(obj)
    except ValueError:
        return False
    return True


def _is_any_of(obj: Any, classes: List[type]):
    return any(_isa(obj, cls) for cls in classes)


def _is_integer_like_type(type: MLIRType):
    return _is_any_of(type, [IntegerType, IndexType])


def _is_float_type(type: MLIRType):
    return _is_any_of(type, [BF16Type, F16Type, F32Type, F64Type])


def cast_to_integer(
    to_type: MLIRType, operand: OpView, is_unsigned_cast: bool
) -> OpView:
    operand: MLIRValue = get_op_result_or_value(operand)
    to_width = IntegerType(to_type).width
    operand_type = operand.type
    if _is_floating_point_type(operand_type):
        if is_unsigned_cast:
            return arith.FPToUIOp(to_type, operand)
        return arith.FPToSIOp(to_type, operand)
    if _is_index_type(operand_type):
        return arith.IndexCastOp(to_type, operand)
    # Assume integer.
    from_width = IntegerType(operand_type).width
    if to_width > from_width:
        if is_unsigned_cast:
            return arith.ExtUIOp(to_type, operand)
        return arith.ExtSIOp(to_type, operand)
    elif to_width < from_width:
        return arith.TruncIOp(to_type, operand)
    raise ValueError(
        f"Unable to cast body expression from {operand_type} to " f"{to_type}"
    )


def cast_to_floating_point(
    to_type: MLIRType, operand: OpView, is_unsigned_cast: bool
) -> OpView:
    operand: MLIRValue = get_op_result_or_value(operand)
    if _is_index_type(operand.type):
        operand = arith.IndexCastOp(IntegerType.get_signless(64), operand).out
    operand_type: MLIRType = operand.type
    if _is_integer_type(operand_type):
        if is_unsigned_cast:
            return arith.UIToFPOp(to_type, operand)
        return arith.SIToFPOp(to_type, operand)
    # Assume FloatType.
    to_width = _get_floating_point_width(to_type)
    from_width = _get_floating_point_width(operand_type)
    if to_width > from_width:
        return arith.ExtFOp(to_type, operand)
    elif to_width < from_width:
        return arith.TruncFOp(to_type, operand)
    raise ValueError(
        f"Unable to cast body expression from {operand_type} to " f"{to_type}"
    )


def type_promotion(x, y):
    if not _is_floating_point_type(x.type):
        x = cast_to_floating_point(F64Type.get(), x, is_unsigned_cast=True)
    if not _is_floating_point_type(y.type):
        y = cast_to_floating_point(F64Type.get(), y, is_unsigned_cast=True)

    return x, y


# def type_cast_signed(self, type_var_name: str, operand: OpView)-> OpView:
#     return self.cast(type_var_name, operand, False)
#
# def type_cast_unsigned(self, type_var_name: str, operand: OpView)-> OpView:
#     return self.cast(type_var_name, operand, True)

# def unary_exp(self, x: OpView)-> OpView:
#     if _is_floating_point_type(x.type):
#         return math.ExpOp(x)
#     raise NotImplementedError("Unsupported 'exp' operand: {x}")
#
# def unary_log(self, x: OpView)-> OpView:
#     if _is_floating_point_type(x.type):
#         return math.LogOp(x)
#     raise NotImplementedError("Unsupported 'log' operand: {x}")


def abs(x) -> OpView:
    x = get_op_result_or_value(x)
    if _is_floating_point_type(x.type):
        return math.AbsFOp(x)
    raise NotImplementedError("Unsupported 'abs' operand: {x}")


def ceil(x) -> OpView:
    x = get_op_result_or_value(x)
    if _is_floating_point_type(x.type):
        return math.CeilOp(x)
    raise NotImplementedError("Unsupported 'ceil' operand: {x}")


def floor(x) -> OpView:
    x = get_op_result_or_value(x)
    if _is_floating_point_type(x.type):
        return math.FloorOp(x)
    raise NotImplementedError("Unsupported 'floor' operand: {x}")


def neg(x) -> OpView:
    x = get_op_result_or_value(x)
    if _is_floating_point_type(x.type):
        return arith.NegFOp(x)
    if _is_complex_type(x.type):
        return complex.NegOp(x)
    raise NotImplementedError("Unsupported 'negf' operand: {x}")


def add(lhs, rhs) -> OpView:
    lhs = get_op_result_or_value(lhs)
    rhs = get_op_result_or_value(rhs)

    if _is_floating_point_type(lhs.type):
        return arith.AddFOp(lhs, rhs)
    # TODO(max): type promotion isn't right
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
        if _is_index_type(lhs.type):
            lhs = cast_to_integer(
                IntegerType.get_signed(64), lhs, is_unsigned_cast=True
            )
        if _is_index_type(rhs.type):
            rhs = cast_to_integer(
                IntegerType.get_signed(64), rhs, is_unsigned_cast=True
            )
        return arith.AddIOp(lhs, rhs)
    if _is_complex_type(lhs.type):
        return complex.AddOp(lhs, rhs)
    raise NotImplementedError("Unsupported 'add' operands: {lhs}, {rhs}")


def sub(lhs, rhs) -> OpView:
    lhs = get_op_result_or_value(lhs)
    rhs = get_op_result_or_value(rhs)

    if _is_floating_point_type(lhs.type):
        return arith.SubFOp(lhs, rhs)
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
        return arith.SubIOp(lhs, rhs)
    if _is_complex_type(lhs.type):
        return complex.SubOp(lhs, rhs)
    raise NotImplementedError("Unsupported 'sub' operands: {lhs}, {rhs}")


def mul(lhs, rhs) -> OpView:
    lhs = get_op_result_or_value(lhs)
    rhs = get_op_result_or_value(rhs)

    if _is_floating_point_type(lhs.type):
        if _is_integer_like_type(rhs.type):
            rhs = cast_to_floating_point(lhs.type, rhs, is_unsigned_cast=True)
        return arith.MulFOp(lhs, rhs)
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
        return arith.MulIOp(lhs, rhs)
    if _is_complex_type(lhs.type):
        return complex.MulOp(lhs, rhs)
    raise NotImplementedError("Unsupported 'mul' operands: {lhs}, {rhs}")


def gt(lhs, rhs) -> OpView:
    lhs = get_op_result_or_value(lhs)
    rhs = get_op_result_or_value(rhs)

    if _is_floating_point_type(lhs.type):
        if _is_integer_type(rhs.type) or _is_index_type(rhs.type):
            rhs = cast_to_floating_point(F64Type.get(), rhs, is_unsigned_cast=True)
        return arith.CmpFOp("ogt", lhs, rhs)
    if _is_integer_type(lhs.type):
        return arith.CmpIOp("sgt", lhs, rhs)
    if _is_index_type(lhs.type):
        return arith.CmpIOp("ugt", lhs, rhs)
    raise NotImplementedError("Unsupported 'mul' operands: {lhs}, {rhs}")


def lt(lhs, rhs) -> OpView:
    lhs = get_op_result_or_value(lhs)
    rhs = get_op_result_or_value(rhs)

    if _is_floating_point_type(lhs.type):
        return arith.CmpFOp("olt", lhs, rhs)
    if _is_integer_type(lhs.type):
        return arith.CmpIOp("slt", lhs, rhs)
    if _is_index_type(lhs.type):
        return arith.CmpIOp("ult", lhs, rhs)
    raise NotImplementedError("Unsupported 'mul' operands: {lhs}, {rhs}")


def ge(lhs, rhs) -> OpView:
    lhs = get_op_result_or_value(lhs)
    rhs = get_op_result_or_value(rhs)

    if _is_floating_point_type(lhs.type):
        return arith.CmpFOp("oge", lhs, rhs)
    if _is_integer_type(lhs.type):
        return arith.CmpIOp("sge", lhs, rhs)
    if _is_index_type(lhs.type):
        return arith.CmpIOp("uge", lhs, rhs)
    raise NotImplementedError("Unsupported 'mul' operands: {lhs}, {rhs}")


def le(lhs, rhs) -> OpView:
    lhs = get_op_result_or_value(lhs)
    rhs = get_op_result_or_value(rhs)

    if _is_floating_point_type(lhs.type):
        return arith.CmpFOp("ole", lhs, rhs)
    if _is_integer_type(lhs.type):
        return arith.CmpIOp("sle", lhs, rhs)
    if _is_index_type(lhs.type):
        return arith.CmpIOp("ule", lhs, rhs)
    raise NotImplementedError("Unsupported 'mul' operands: {lhs}, {rhs}")


class _Value:
    # def cast(
    #     self, type_var_name: str, operand: OpView, is_unsigned_cast: bool = False
    # )-> OpView:
    #     try:
    #         to_type = self.type_mapping[type_var_name]
    #     except KeyError:
    #         raise ValueError(
    #             f"Unbound type variable '{type_var_name}' ("
    #             f"expected one of {self.type_mapping.keys()}"
    #         )
    #     if operand.type == to_type:
    #         return operand
    #     if _is_integer_type(to_type):
    #         return self.cast_to_integer(to_type, operand, is_unsigned_cast)
    #     elif _is_floating_point_type(to_type):
    #         return self.cast_to_floating_point(to_type, operand, is_unsigned_cast)

    def cast_to_integer(
        self, to_type: MLIRType, operand: OpView, is_unsigned_cast: bool
    ) -> OpView:
        operand: MLIRValue = get_op_result_or_value(operand)
        to_width = IntegerType(to_type).width
        operand_type = operand.type
        if _is_floating_point_type(operand_type):
            if is_unsigned_cast:
                return arith.FPToUIOp(to_type, operand)
            return arith.FPToSIOp(to_type, operand)
        if _is_index_type(operand_type):
            return arith.IndexCastOp(to_type, operand)
        # Assume integer.
        from_width = IntegerType(operand_type).width
        if to_width > from_width:
            if is_unsigned_cast:
                return arith.ExtUIOp(to_type, operand)
            return arith.ExtSIOp(to_type, operand)
        elif to_width < from_width:
            return arith.TruncIOp(to_type, operand)
        raise ValueError(
            f"Unable to cast body expression from {operand_type} to " f"{to_type}"
        )

    def cast_to_floating_point(
        self, to_type: MLIRType, operand: OpView, is_unsigned_cast: bool
    ) -> OpView:
        operand: MLIRValue = get_op_result_or_value(operand)
        operand_type: MLIRType = operand.type
        if _is_integer_type(operand_type):
            if is_unsigned_cast:
                return arith.UIToFPOp(to_type, operand)
            return arith.SIToFPOp(to_type, operand)
        # Assume FloatType.
        to_width = _get_floating_point_width(to_type)
        from_width = _get_floating_point_width(operand_type)
        if to_width > from_width:
            return arith.ExtFOp(to_type, operand)
        elif to_width < from_width:
            return arith.TruncFOp(to_type, operand)
        raise ValueError(
            f"Unable to cast body expression from {operand_type} to " f"{to_type}"
        )

    __abs__ = abs
    __ceil__ = ceil
    __floor__ = floor
    __neg__ = neg
    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __gt__ = gt
    __lt__ = lt
    __ge__ = ge
    __le__ = le

    # def binary_max_signed(self, lhs: OpView, rhs: OpView)-> OpView:
    #     if _is_floating_point_type(lhs.type):
    #         return arith.MaxFOp(lhs, rhs)
    #     if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
    #         return arith.MaxSIOp(lhs, rhs)
    #     raise NotImplementedError("Unsupported 'max' operands: {lhs}, {rhs}")
    #
    # def binary_max_unsigned(self, lhs: OpView, rhs: OpView)-> OpView:
    #     if _is_floating_point_type(lhs.type):
    #         return arith.MaxFOp(lhs, rhs)
    #     if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
    #         return arith.MaxUIOp(lhs, rhs)
    #     raise NotImplementedError("Unsupported 'max_unsigned' operands: {lhs}, {rhs}")
    #
    # def binary_min_signed(self, lhs: OpView, rhs: OpView)-> OpView:
    #     if _is_floating_point_type(lhs.type):
    #         return arith.MinFOp(lhs, rhs)
    #     if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
    #         return arith.MinSIOp(lhs, rhs)
    #     raise NotImplementedError("Unsupported 'min' operands: {lhs}, {rhs}")
    #
    # def binary_min_unsigned(self, lhs: OpView, rhs: OpView)-> OpView:
    #     if _is_floating_point_type(lhs.type):
    #         return arith.MinFOp(lhs, rhs)
    #     if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
    #         return arith.MinUIOp(lhs, rhs)
    #     raise NotImplementedError("Unsupported 'min_unsigned' operands: {lhs}, {rhs}")
