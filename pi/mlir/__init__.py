import atexit
import contextlib
import ctypes
import inspect
import sys


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

# noinspection PyUnresolvedReferences
from ._mlir_libs._pi_mlir import (
    # AnyTorchDictKeyType,
    # AnyTorchDictKeyValue,
    AnyTorchListOfOptionalTensorValue,
    AnyTorchOptionalListOfTorchIntValue,
    AnyTorchListOfTensorType,
    AnyTorchListOfTensorValue,
    AnyTorchListOfTorchBoolType,
    AnyTorchListOfTorchBoolValue,
    AnyTorchListOfTorchFloatType,
    AnyTorchListOfTorchFloatValue,
    AnyTorchListOfTorchIntType,
    AnyTorchListOfTorchIntValue,
    AnyTorchListOfTorchStringType,
    AnyTorchListOfTorchStringValue,
    AnyTorchListType,
    AnyTorchListValue,
    AnyTorchOptionalBoolType,
    AnyTorchOptionalBoolValue,
    AnyTorchOptionalDeviceType,
    AnyTorchOptionalDeviceValue,
    AnyTorchOptionalFloatType,
    AnyTorchOptionalFloatValue,
    AnyTorchOptionalGeneratorType,
    AnyTorchOptionalGeneratorValue,
    AnyTorchOptionalIntType,
    AnyTorchOptionalIntType,
    AnyTorchOptionalIntValue,
    AnyTorchOptionalStringType,
    AnyTorchOptionalStringValue,
    AnyTorchOptionalTensorType,
    AnyTorchOptionalTensorValue,
    AnyTorchOptionalType,
    AnyTorchOptionalValue,
    AnyTorchScalarType,
    AnyTorchScalarValue,
    AnyTorchTensorType,
    Torch_BoolType,
    Torch_BoolValue,
    Torch_DeviceType,
    Torch_DeviceValue,
    Torch_DictType,
    Torch_DictValue,
    Torch_FloatType,
    Torch_FloatValue,
    Torch_IntType,
    Torch_IntValue,
    Torch_LinearParamsType,
    Torch_LinearParamsValue,
    Torch_NnModuleType,
    Torch_NnModuleValue,
    Torch_NonValueTensorType,
    Torch_NonValueTensorValue,
    Torch_NoneType,
    Torch_NoneValue,
    Torch_NumberType,
    Torch_NumberValue,
    Torch_StringType,
    Torch_StringValue,
    Torch_TupleType,
    Torch_TupleValue,
    Torch_ValueTensorType,
    Torch_ValueTensorValue,
)

# noinspection PyUnresolvedReferences
from ._mlir_libs._pi_mlir import Tensor, dtype

Tensor = Tensor

from .dialects import _ods_common
from .dialects._ods_common import get_op_result_or_value, get_op_results_or_values
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


def type_caster(results):
    if len(results) == 1:
        val = results[0]
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
        if AnyTorchListOfTensorType.isinstance(val.type):
            return AnyTorchListOfTensorValue(val)
        if Torch_NnModuleType.isinstance(val.type):
            return Torch_NnModuleValue(val)
        if AnyTorchTensorType.isinstance(val.type):
            return Tensor(val)
        if AnyTorchScalarType.isinstance(val.type):
            return AnyTorchScalarValue(val)
        if Torch_NonValueTensorType.isinstance(val.type):
            return Torch_NonValueTensorValue(val)
        if Torch_NoneType.isinstance(val.type):
            return Torch_NoneValue(val)
        if Torch_NumberType.isinstance(val.type):
            return Torch_NumberValue(val)
        if AnyTorchOptionalType.isinstance(val.type):
            return AnyTorchOptionalValue(val)
        if AnyTorchOptionalIntType.isinstance(val.type):
            return AnyTorchOptionalIntValue(val)
        if AnyTorchOptionalBoolType.isinstance(val.type):
            return AnyTorchOptionalBoolValue(val)
        if AnyTorchOptionalDeviceType.isinstance(val.type):
            return AnyTorchOptionalDeviceValue(val)
        if AnyTorchOptionalFloatType.isinstance(val.type):
            return AnyTorchOptionalFloatValue(val)
        if AnyTorchOptionalTensorType.isinstance(val.type):
            return AnyTorchOptionalTensorValue(val)
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
        return val
    else:
        return tuple(type_caster((r,)) for r in results)


class ValueMeta(type(Value)):
    def __call__(cls, *args, **kwargs):
        cls_obj = cls.__new__(cls, *args, **kwargs)
        cls.__init__(cls_obj, *args, **kwargs)
        results = tuple(cls_obj.results)
        if results:
            return type_caster(results)
        else:
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
