from typing import (
    Union,
    Container,
    Annotated,
    NewType,
    TypeVar,
    Generic,
    List,
    Any,
    Optional,
    Dict,
    Tuple,
    Sequence,
)

T = TypeVar("T")
U = TypeVar("U")


class Torch_Value(Generic[T]):
    ...


class Torch_NonValueTensorType:
    ...


class Torch_ValueTensorType:
    ...


class Tensor:
    ...


class Torch_NoneType:
    ...


class Torch_IntType:
    ...


class Torch_FloatType:
    ...


class Torch_AnyType:
    ...


class Torch_BoolType:
    ...


class Torch_DeviceType:
    ...


class Torch_GeneratorType:
    ...


class Torch_LinearParamsType:
    ...


# class Torch_NumberType:
#     ...


class Torch_NnModuleType:
    ...


class Torch_StringType:
    ...


Torch_IntType = Torch_Value[Torch_IntType] | int
Torch_FloatType = Torch_Value[Torch_FloatType] | float
Torch_AnyType = Torch_Value[Torch_AnyType] | Any
Torch_BoolType = Torch_Value[Torch_BoolType] | bool
Torch_StringType = Torch_Value[Torch_StringType] | str
Torch_DeviceType = Torch_Value[Torch_DeviceType] | str

# AnyTorchTensorType = Union[Torch_NonValueTensorType, Torch_ValueTensorType]
AnyTorchTensorType = Tensor


# class Torch_OptionalType(Generic[T]):
#     ...
#
#
class Torch_List(Generic[T]):
    ...


class Torch_Dict(Generic[T, U]):
    ...


Torch_ListType = Torch_List
Torch_OptionalType = Optional[T]
Torch_DictType = Torch_Dict


Torch_TupleType = Tuple


class Torch_UnionType(Generic[T]):
    ...


AnyTypeOf = Union


def OptionalOf(type):
    return Optional[type]


AnyTorchOptionalTensorType = OptionalOf(AnyTorchTensorType)


AnyTorchOptionalIntType = OptionalOf(Torch_IntType)


AnyTorchOptionalFloatType = OptionalOf(Torch_FloatType)


AnyTorchOptionalBoolType = OptionalOf(Torch_BoolType)


AnyTorchOptionalStringType = OptionalOf(Torch_StringType)


AnyTorchOptionalDeviceType = OptionalOf(Torch_DeviceType)


AnyTorchOptionalGeneratorType = OptionalOf(Torch_GeneratorType)


def ListOf(allowedTypes):
    return Sequence[AnyTypeOf[allowedTypes]] | (
        Torch_ListType[allowedTypes.__args__[0].__args__[0]]
        if hasattr(allowedTypes, "__args__")
        and hasattr(allowedTypes.__args__[0], "__args__")
        else allowedTypes
    )


AnyTorchListOfTorchBoolType = ListOf(Torch_BoolType)


AnyTorchListOfTorchIntType = ListOf(Torch_IntType)


AnyTorchListOfTorchFloatType = ListOf(Torch_FloatType)


AnyTorchListOfTorchStringType = ListOf(Torch_StringType)


AnyTorchListOfTensorType = ListOf(AnyTorchTensorType)


AnyTorchListOfOptionalTensorType = ListOf(AnyTorchOptionalTensorType)


AnyTorchListOfOptionalIntType = ListOf(AnyTorchOptionalIntType)


AnyTorchOptionalListOfTorchIntType = OptionalOf(AnyTorchListOfTorchIntType)


AnyTorchOptionalListOfTorchFloatType = OptionalOf(AnyTorchListOfTorchFloatType)


class TorchNumber:
    ...


AnyTorchScalarType = TorchNumber


AnyTorchOptionalScalarType = OptionalOf(AnyTorchScalarType)


# AnyTorchDictKeyType = AnyTypeOf[
#     Torch_AnyType,
#     Torch_IntType,
#     Torch_BoolType,
#     Torch_FloatType,
#     Torch_StringType,
#     Torch_FloatType,
#     AnyTorchTensorType,
# ]
#
#
# AnyTorchType = AnyTypeOf[
#     AnyTorchScalarType,
#     AnyTorchTensorType,
#     Torch_AnyType,
#     Torch_BoolType,
#     Torch_DictType,
#     Torch_DeviceType,
#     Torch_GeneratorType,
#     Torch_ListType,
#     Torch_LinearParamsType,
#     Torch_NumberType,
#     Torch_NnModuleType,
#     Torch_NoneType,
#     Torch_OptionalType,
#     Torch_StringType,
#     Torch_TupleType,
#     Torch_UnionType,
# ]


AnyTorchListType = ListOf(Any)
