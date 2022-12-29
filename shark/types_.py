import builtins
import re
from enum import Enum
from typing import Union, List, Tuple, Any

from shark._mlir import (
    _Torch_IntType,
    _Torch_BoolType,
    _Torch_StringType,
    _Torch_FloatType,
    _Torch_ValueTensorType,
    _Torch_NonValueTensorType,
    is_a_torch_int_type,
    is_a_torch_string_type,
    is_a_torch_float_type,
    is_a_torch_value_tensor_type,
    is_a_torch_nonvalue_tensor_type,
)
from torch_mlir import ir
from torch_mlir.dialects._ods_common import (
    get_op_result_or_value,
)
from torch_mlir.ir import (
    Type as MLIRType,
    Value as MLIRValue,
)

# !torch.vtensor<[1,2,3],f32>
reg = re.compile(r"!torch.vtensor<\[(.*)\],(.*)>")


def parse_sizes_from_tensor_type_str(t: ir.OpView) -> List[int]:
    # TODO(max): pull straight from the ranked type
    t = get_op_result_or_value(t)
    sizes, dtype = reg.findall(str(t.type))[0]
    sizes = [s if s != "?" else "-1" for s in sizes.split(",")]
    return list(map(int, sizes)), dtype


def get_type(t: Union[MLIRType, MLIRValue]):
    if not isinstance(t, MLIRType):
        assert isinstance(t, MLIRValue)
        t = t.type
    return t


class Torch_IntType(_Torch_IntType):
    def __init__(self, type: Union[MLIRType, MLIRValue]):
        type = get_type(type)
        assert is_a_torch_int_type(type._CAPIPtr)
        super(Torch_IntType, self).__init__(type._CAPIPtr)


class Torch_BoolType(_Torch_BoolType):
    def __init__(self, type: Union[MLIRType, MLIRValue]):
        type = get_type(type)
        assert is_a_torch_int_type(type._CAPIPtr)
        super(Torch_BoolType, self).__init__(type._CAPIPtr)


class Torch_StringType(_Torch_StringType):
    def __init__(self, type: Union[MLIRType, MLIRValue]):
        type = get_type(type)
        assert is_a_torch_string_type(type._CAPIPtr)
        super(Torch_StringType, self).__init__(type._CAPIPtr)


class Torch_FloatType(_Torch_FloatType):
    def __init__(self, type: Union[MLIRType, MLIRValue]):
        type = get_type(type)
        assert is_a_torch_float_type(type._CAPIPtr)
        super(Torch_FloatType, self).__init__(type._CAPIPtr)


class Torch_ValueTensorType(_Torch_ValueTensorType):
    def __init__(self, type: Union[MLIRType, MLIRValue]):
        type = get_type(type)
        assert is_a_torch_value_tensor_type(type._CAPIPtr)
        super(Torch_ValueTensorType, self).__init__(type._CAPIPtr)


class Torch_NonValueTensorType(_Torch_NonValueTensorType):
    def __init__(self, type: Union[MLIRType, MLIRValue]):
        type = get_type(type)
        assert is_a_torch_nonvalue_tensor_type(type._CAPIPtr)
        super(Torch_NonValueTensorType, self).__init__(type._CAPIPtr)


def is_mlir_value(v):
    return isinstance(v, (ir.OpView, ir.Operation, ir.Value, ir.OpResultList))


# IntegerType.get_signless(32) -> i32
# IntegerType.get_signed(32) -> si32
# IntegerType.get_unsigned(32) -> ui32


class dtype(Enum):
    """
    |-------------------|--------------------|
    | Torch Type        | MLIR Type          |
    |-------------------|--------------------|
    | torch.bfloat16    | bf16               |
    | torch.bool        | i1                 |
    | torch.complex*    | complex<*>         |
    | torch.float16     | f16                |
    | torch.float32     | f32                |
    | torch.float64     | f64                |
    | torch.int16       | si16               |
    | torch.int32       | si32               |
    | torch.int64       | si64               |
    | torch.int8        | si8                |
    | torch.qint8       | !torch.qint8       |
    | torch.quint8      | !torch.quint8      |
    | torch.uint8       | ui8                |
    |-------------------|--------------------|
    """

    bfloat16 = "bfloat16"
    bool = "bool"
    complex32 = "complex32"
    complex64 = "complex64"
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    qint8 = "qint8"
    quint8 = "quint8"
    uint8 = "uint8"

    def to_mlir_type(self):
        # if ctx is None:
        #     ctx = get_default_loc_context()
        match self:
            case dtype.bfloat16:
                return ir.BF16Type.get()
            case dtype.bool:
                return ir.IntegerType.get_signless(1)
            case dtype.complex32:
                return ir.ComplexType.get(ir.F32Type.get())
            case dtype.complex64:
                return ir.ComplexType.get(ir.F64Type.get())
            case dtype.float16:
                return ir.F16Type.get()
            case dtype.float32:
                return ir.F32Type.get()
            case dtype.float64:
                return ir.F64Type.get()
            case dtype.int8:
                return ir.IntegerType.get_signed(8)
            case dtype.int16:
                return ir.IntegerType.get_signed(16)
            case dtype.int32:
                return ir.IntegerType.get_signed(32)
            case dtype.int64:
                return ir.IntegerType.get_signed(64)
            case dtype.uint8:
                return ir.IntegerType.get_unsigned(8)
            case _:
                raise NotImplementedError("Something's wrong with the internet")

    @staticmethod
    def from_mlir_type(t: str):
        match t:
            case "bf16":
                return dtype.bfloat16
            case "i1":
                return dtype.bool
            case "complex32":
                return dtype.complex32
            case "complex64":
                return dtype.complex64
            case "f16":
                return dtype.float16
            case "f32":
                return dtype.float32
            case "f64":
                return dtype.float64
            case "si8":
                return dtype.int8
            case "si16":
                return dtype.int16
            case "si32":
                return dtype.int32
            case "si64":
                return dtype.int64
            case "ui8":
                return dtype.uint8
            case _:
                raise NotImplementedError(f"Something's wrong with the internet {t}")


# attr = DenseFPElementsAttr(Attribute.parse("dense<0.0> : tensor<3x5xf32>"))


bfloat16 = dtype.bfloat16
bool = dtype.bool
complex32 = dtype.complex32
complex64 = dtype.complex64
float16 = dtype.float16
float32 = dtype.float32
float64 = dtype.float64
int8 = dtype.int8
int16 = dtype.int16
int32 = dtype.int32
int64 = dtype.int64
qint8 = dtype.qint8
quint8 = dtype.quint8
uint8 = dtype.uint8

# _int = builtins.int
# _float = builtins.float
# _bool = builtins.bool
size = Union[List[int], Tuple[int, ...]]

Number = Union[builtins.int, builtins.float, builtins.bool]
Generator = Device = Any
