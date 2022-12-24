#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from torch_mlir.dialects._ods_common import (
    get_default_loc_context,
    get_op_result_or_value,
)
from torch_mlir.ir import *
from typing import Any, List, Union

try:
    from shark.dialects.value_ import _Value

except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


def _isa(obj: Any, cls: type):
    try:
        cls(obj)
    except ValueError:
        return False
    return True


def _is_any_of(obj: Any, classes: List[type]):
    return any(_isa(obj, cls) for cls in classes)


def _is_integer_like_type(type: Type):
    return _is_any_of(type, [IntegerType, IndexType])


def _is_float_type(type: Type):
    return _is_any_of(type, [BF16Type, F16Type, F32Type, F64Type])


class ConstantOp:
    """Specialization for the constant op class."""

    def __init__(
        self, result: Type, value: Union[int, float, Attribute], *, loc=None, ip=None
    ):
        if isinstance(value, int):
            super().__init__(IntegerAttr.get(result, value), loc=loc, ip=ip)
        elif isinstance(value, float):
            super().__init__(FloatAttr.get(result, value), loc=loc, ip=ip)
        else:
            super().__init__(value, loc=loc, ip=ip)

    @classmethod
    def create_index(cls, value: int, *, loc=None, ip=None):
        """Create an index-typed constant."""
        return cls(
            IndexType.get(context=get_default_loc_context(loc)), value, loc=loc, ip=ip
        )

    @property
    def type(self):
        return self.results[0].type

    @property
    def literal_value(self) -> Union[int, float]:
        if _is_integer_like_type(self.type):
            return IntegerAttr(self.value).value
        elif _is_float_type(self.type):
            return FloatAttr(self.value).value
        else:
            raise ValueError("only integer and float constants have literal values")


class CmpFOp:
    """Specialization for the constant op class."""

    def __init__(
        self,
        predicate: str,
        lhs: Union[Operation, OpView, Value],
        rhs: Union[Operation, OpView, Value],
        *,
        loc=None,
        ip=None,
    ):

        predicates = {
            "false": 0,
            "oeq": 1,
            "ogt": 2,
            "oge": 3,
            "olt": 4,
            "ole": 5,
            "one": 6,
            "ord": 7,
            "ueq": 8,
            "ugt": 9,
            "uge": 10,
            "ult": 11,
            "ule": 12,
            "une": 13,
            "uno": 14,
            "true": 15,
        }
        assert predicate in predicates, f"predicate {predicate} not in predicates"
        context = get_default_loc_context(loc)
        lhs = get_op_result_or_value(lhs)
        rhs = get_op_result_or_value(rhs)
        predicate = IntegerAttr.get(IntegerType.get_signless(64), predicates[predicate])
        result_type = IntegerType.get_signless(1, context=context)
        super().__init__(result_type, predicate, lhs, rhs, loc=loc, ip=ip)


class CmpIOp:
    """Specialization for the constant op class."""

    def __init__(
        self,
        predicate: str,
        lhs: Union[Operation, OpView, Value],
        rhs: Union[Operation, OpView, Value],
        *,
        loc=None,
        ip=None,
    ):
        assert predicate in {
            "eq",
            "ne",
            "slt",
            "sle",
            "sgt",
            "sge",
            "ult",
            "ule",
            "ugt",
            "uge",
        }

        context = get_default_loc_context(loc)
        lhs = get_op_result_or_value(lhs)
        rhs = get_op_result_or_value(rhs)
        predicate = StringAttr.get(predicate, context=context)
        result_type = IntegerType.get_signless(1, context=context)
        super().__init__(result_type, predicate, lhs, rhs, loc=loc, ip=ip)


class MulFOp(_Value):
    pass
