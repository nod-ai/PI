import atexit
import inspect

import numpy as np

from ._mlir_libs._pi_mlir import (
    TorchAnyType,
    TorchAnyValue,
    TorchBoolType,
    TorchBoolValue,
    TorchDeviceType,
    TorchDeviceValue,
    TorchDictType,
    TorchDictValue,
    TorchFloatType,
    TorchFloatValue,
    TorchGeneratorType,
    TorchGeneratorValue,
    TorchIntType,
    TorchIntValue,
    TorchLinearParamsType,
    TorchLinearParamsValue,
    TorchListType,
    TorchListValue,
    TorchNnModuleType,
    TorchNnModuleValue,
    TorchNonValueTensorType,
    TorchNonValueTensorValue,
    TorchNoneType,
    TorchNoneValue,
    TorchNumberType,
    TorchNumberValue,
    TorchOptionalType,
    TorchOptionalValue,
    TorchQInt8Type,
    TorchQInt8Value,
    TorchQUInt8Type,
    TorchQUInt8Value,
    TorchStringType,
    TorchStringValue,
    TorchTupleType,
    TorchTupleValue,
    TorchUnionType,
    TorchUnionValue,
    TorchValueTensorType,
    TorchValueTensorValue,
)
from .dialects import _ods_common
from .dialects._ods_common import get_op_result_or_value
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


class ValueMeta(type(Value)):
    def __call__(cls, *args, **kwargs):
        cls_obj = cls.__new__(cls, *args, **kwargs)
        cls.__init__(cls_obj, *args, **kwargs)
        if len(cls_obj.results) == 1:
            val = get_op_result_or_value(cls_obj)
            if TorchAnyType.isinstance(val.type):
                return TorchAnyValue(val)
            if TorchBoolType.isinstance(val.type):
                return TorchBoolValue(val)
            if TorchDeviceType.isinstance(val.type):
                return TorchDeviceValue(val)
            if TorchDictType.isinstance(val.type):
                return TorchDictValue(val)
            if TorchFloatType.isinstance(val.type):
                return TorchFloatValue(val)
            if TorchGeneratorType.isinstance(val.type):
                return TorchGeneratorValue(val)
            if TorchIntType.isinstance(val.type):
                return TorchIntValue(val)
            if TorchLinearParamsType.isinstance(val.type):
                return TorchLinearParamsValue(val)
            if TorchListType.isinstance(val.type):
                return TorchListValue(val)
            if TorchNnModuleType.isinstance(val.type):
                return TorchNnModuleValue(val)
            if TorchNonValueTensorType.isinstance(val.type):
                return TorchNonValueTensorValue(val)
            if TorchNoneType.isinstance(val.type):
                return TorchNoneValue(val)
            if TorchNumberType.isinstance(val.type):
                return TorchNumberValue(val)
            if TorchOptionalType.isinstance(val.type):
                return TorchOptionalValue(val)
            if TorchQInt8Type.isinstance(val.type):
                return TorchQInt8Value(val)
            if TorchQUInt8Type.isinstance(val.type):
                return TorchQUInt8Value(val)
            if TorchStringType.isinstance(val.type):
                return TorchStringValue(val)
            if TorchTupleType.isinstance(val.type):
                return TorchTupleValue(val)
            if TorchUnionType.isinstance(val.type):
                return TorchUnionValue(val)
            if TorchValueTensorType.isinstance(val.type):
                return TorchValueTensorValue(val)
        return cls_obj


def rebuild_with_meta(parent_opview_cls, mixin=False):
    v = ValueMeta(
        f"{parent_opview_cls.__name__}",
        parent_opview_cls.__bases__,
        dict(parent_opview_cls.__dict__),
    )

    # mixins (extensions) for some reasons don't suffer from this problem
    # i.e., the __class__ is the correctly patched/hacked one
    if not mixin:
        # some ops don't have __init__ but one is inherited from OpView (as an instancemethod)
        if not inspect.ismethoddescriptor(v.__init__):
            v.__init__.__closure__[0].cell_contents = v
    return v


def extend_opview_class(ext_module):
    def class_decorator(parent_opview_cls: type):
        if ext_module is None:
            return rebuild_with_meta(parent_opview_cls)
        mixin_cls = NotImplemented
        # First try to resolve by name.
        try:
            mixin_cls = getattr(ext_module, parent_opview_cls.__name__)
        except AttributeError:
            # Fall back to a select_opview_mixin hook.
            try:
                select_mixin = getattr(ext_module, "select_opview_mixin")
            except AttributeError:
                pass
            else:
                mixin_cls = select_mixin(parent_opview_cls)

        if mixin_cls is NotImplemented or mixin_cls is None:
            return rebuild_with_meta(parent_opview_cls)

        # Have a mixin_cls. Create an appropriate subclass.
        try:

            class LocalOpView(mixin_cls, parent_opview_cls):
                pass

        except TypeError as e:
            raise TypeError(
                f"Could not mixin {mixin_cls} into {parent_opview_cls}"
            ) from e
        LocalOpView.__name__ = parent_opview_cls.__name__
        LocalOpView.__qualname__ = parent_opview_cls.__qualname__
        return rebuild_with_meta(LocalOpView, mixin=True)

    return class_decorator


_ods_common.extend_opview_class = extend_opview_class
# note this has to come after the extend_opview_class is monkey-patched
from .dialects import _torch_ops_gen as torch_dialect

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
