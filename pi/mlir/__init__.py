import atexit
import contextlib
import ctypes
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
from ._mlir_libs._pi_mlir import Tensor, dtype, ops

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
