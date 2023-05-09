import atexit

import numpy as np

from ._mlir_libs._pi_mlir import *
from .dialects import _torch_ops_gen as torch_dialect
from .ir import (
    Value,
    Type,
    RankedTensorType,
    ShapedType,
    Operation,
    F16Type,
    F32Type,
    F64Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    Location,
    Context,
    DenseElementsAttr,
    register_attribute_builder,
)

DefaultContext = Context()
# Push a default context onto the context stack at import time.
DefaultContext.__enter__()
DefaultContext.allow_unregistered_dialects = False


@atexit.register
def __exit_ctxt():
    DefaultContext.__exit__(None, None, None)


DefaultLocation = Location.unknown()
DefaultLocation.__enter__()


@atexit.register
def __exit_loc():
    DefaultLocation.__exit__(None, None, None)


F16 = F16Type.get()
F32 = F32Type.get()
F64 = F64Type.get()
I16 = IntegerType.get_signless(16)
I32 = IntegerType.get_signless(32)
I64 = IntegerType.get_signless(64)
Index = IndexType.get()


@register_attribute_builder("F64Attr")
def _f64Attr(x, context):
    return FloatAttr.get(F64Type.get(context=context), x)


@register_attribute_builder("AnyI64Attr")
def _anyI64Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), x)


@register_attribute_builder("I1Attr")
def _i1Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(1, context=context), x)


def _fp64ElementsAttr(x):
    return DenseElementsAttr.get(
        np.array(x, dtype=np.float64),
        type=F64Type.get(),
    )
