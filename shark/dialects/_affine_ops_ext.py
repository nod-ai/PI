#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from torch_mlir.ir import *
    from ._ods_common import get_default_loc_context as _get_default_loc_context
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union

from ._ods_common import (
    get_op_result_or_value as _get_op_result_or_value,
    get_op_results_or_values as _get_op_results_or_values,
)


class AffineForOp:
    """Specialization for the Affine for op class."""

    def __init__(
        self,
        lower_bound,
        upper_bound,
        step,
        *,
        loc=None,
        ip=None,
    ):
        attributes = {
            "lower_bound": AffineMapAttr.get(AffineMap.get_constant(lower_bound)),
            "upper_bound": AffineMapAttr.get(AffineMap.get_constant(upper_bound)),
            "step": IntegerAttr.get(IntegerType.get_signless(64), step),
        }

        results = []
        super().__init__(
            self.build_generic(
                regions=1,
                results=[],
                attributes=attributes,
                operands=[],
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(IndexType.get(), *results)

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def induction_variable(self):
        """Returns the induction variable of the loop."""
        return self.body.arguments[0]

    @property
    def inner_iter_args(self):
        """Returns the loop-carried arguments usable within the loop.

        To obtain the loop-carried operands, use `iter_args`.
        """
        return self.body.arguments[1:]


class AffineLoadOp:
    """Specialization for the MemRef load operation."""

    def __init__(
        self,
        memref: Union[Operation, OpView, Value],
        indices: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Creates a memref load operation.

        Args:
          memref: the buffer to load from.
          indices: the list of subscripts, may be empty for zero-dimensional
            buffers.
          loc: user-visible location of the operation.
          ip: insertion point.
        """
        memref_resolved = _get_op_result_or_value(memref)
        indices_resolved = [] if indices is None else _get_op_results_or_values(indices)
        return_type = MemRefType(memref_resolved.type).element_type
        super().__init__(return_type, memref, indices_resolved, loc=loc, ip=ip)


# class IfOp:
#   """Specialization for the SCF if op class."""
#
#   def __init__(self,
#                cond,
#                results_=[],
#                *,
#                hasElse=False,
#                loc=None,
#                ip=None):
#     """Creates an SCF `if` operation.
#
#     - `cond` is a MLIR value of 'i1' type to determine which regions of code will be executed.
#     - `hasElse` determines whether the if operation has the else branch.
#     """
#     operands = []
#     operands.append(cond)
#     results = []
#     results.extend(results_)
#     super().__init__(
#         self.build_generic(
#             regions=2,
#             results=results,
#             operands=operands,
#             loc=loc,
#             ip=ip))
#     self.regions[0].blocks.append(*[])
#     if hasElse:
#         self.regions[1].blocks.append(*[])
#
#   @property
#   def then_block(self):
#     """Returns the then block of the if operation."""
#     return self.regions[0].blocks[0]
#
#   @property
#   def else_block(self):
#     """Returns the else block of the if operation."""
#     return self.regions[1].blocks[0]
