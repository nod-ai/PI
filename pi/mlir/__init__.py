import atexit
import contextlib
import ctypes
import inspect
import sys

import numpy as np


@contextlib.contextmanager
def dl_open_guard():
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    sys.setdlopenflags(old_flags)


with dl_open_guard():
    # noinspection PyUnresolvedReferences
    from ._mlir_libs import _mlir

    # noinspection PyUnresolvedReferences
    from ._mlir_libs import _pi_mlir

    # noinspection PyUnresolvedReferences
    from ._mlir_libs import _torchMlir

from ._mlir_libs._pi_mlir import ops

from ._mlir_libs._pi_mlir import (
    # AnyTorchDictKeyType,
    # AnyTorchListOfOptionalIntType,
    # AnyTorchListOfOptionalTensorType,
    # AnyTorchListOfTensorType,
    AnyTorchListOfTorchBoolType,
    AnyTorchListOfTorchIntType,
    AnyTorchListOfTorchStringType,
    AnyTorchListType,
    AnyTorchOptionalBoolType,
    AnyTorchOptionalDeviceType,
    AnyTorchOptionalFloatType,
    AnyTorchOptionalGeneratorType,
    AnyTorchOptionalIntType,
    AnyTorchOptionalStringType,
    # AnyTorchOptionalTensorType,

    AnyTorchOptionalType,

    # AnyTorchOptionalListOfTorchIntType,
    # AnyTorchTensorType,
    Torch_BoolType,
    Torch_DeviceType,
    Torch_DictType,
    Torch_FloatType,
    Torch_IntType,
    Torch_LinearParamsType,
    Torch_NnModuleType,
    Torch_NonValueTensorType,
    Torch_NoneType,
    Torch_NumberType,
    Torch_StringType,
    Torch_TupleType,
    Torch_ValueTensorType,
)
from ._mlir_libs._pi_mlir import (
    # AnyTorchDictKeyValue,
    # AnyTorchListOfOptionalIntValue,
    # AnyTorchListOfOptionalTensorValue,
    # AnyTorchListOfTensorValue,
    AnyTorchListOfTorchBoolValue,
    AnyTorchListOfTorchIntValue,
    AnyTorchListOfTorchStringValue,
    AnyTorchListValue,
    AnyTorchOptionalBoolValue,
    AnyTorchOptionalDeviceValue,
    AnyTorchOptionalFloatValue,
    AnyTorchOptionalGeneratorValue,
    AnyTorchOptionalIntValue,
    AnyTorchOptionalStringValue,
    # AnyTorchOptionalTensorValue,

    AnyTorchOptionalValue,

    # AnyTorchOptionalListOfTorchIntValue,
    # AnyTorchTensorValue,
    Torch_BoolValue,
    Torch_DeviceValue,
    Torch_DictValue,
    Torch_FloatValue,
    Torch_IntValue,
    Torch_LinearParamsValue,
    Torch_NnModuleValue,
    Torch_NonValueTensorValue,
    Torch_NoneValue,
    Torch_NumberValue,
    Torch_StringValue,
    Torch_TupleValue,
    Torch_ValueTensorValue,
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
            # if TorchAnyType.isinstance(val.type):
            #     return TorchAnyValue(val)
            if Torch_BoolType.isinstance(val.type):
                return Torch_BoolValue(val)
            if Torch_DeviceType.isinstance(val.type):
                return Torch_DeviceValue(val)
            if Torch_DictType.isinstance(val.type):
                return Torch_DictValue(val)
            if Torch_FloatType.isinstance(val.type):
                return Torch_FloatValue(val)
            # if Torch_GeneratorType.isinstance(val.type):
            #     return Torch_GeneratorValue(val)
            if Torch_IntType.isinstance(val.type):
                return Torch_IntValue(val)
            if Torch_LinearParamsType.isinstance(val.type):
                return Torch_LinearParamsValue(val)
            if AnyTorchListType.isinstance(val.type):
                return AnyTorchListValue(val)
            if Torch_NnModuleType.isinstance(val.type):
                return Torch_NnModuleValue(val)
            if Torch_NonValueTensorType.isinstance(val.type):
                return Torch_NonValueTensorValue(val)
            if Torch_NoneType.isinstance(val.type):
                return Torch_NoneValue(val)
            if Torch_NumberType.isinstance(val.type):
                return Torch_NumberValue(val)
            # if TorchOptionalType.isinstance(val.type):
            #     return TorchOptionalValue(val)
            # if TorchQInt8Type.isinstance(val.type):
            #     return TorchQInt8Value(val)
            # if TorchQUInt8Type.isinstance(val.type):
            #     return TorchQUInt8Value(val)
            if Torch_StringType.isinstance(val.type):
                return Torch_StringValue(val)
            if Torch_TupleType.isinstance(val.type):
                return Torch_TupleValue(val)
            # if TorchUnionType.isinstance(val.type):
            #     return TorchUnionValue(val)
            if Torch_ValueTensorType.isinstance(val.type):
                return Torch_ValueTensorValue(val)
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
