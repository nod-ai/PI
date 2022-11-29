import abc
import inspect
from typing import Callable, Tuple, Any, Optional, NamedTuple, Iterable

import torch

from shark._mlir_libs._mlir import ir

from shark.compiler import immutable_collections


class ArgsKwargsPair(NamedTuple):
    args: Tuple[Any, ...]
    kwargs: dict[str, Any]


def _args_kwargs_to_normalized_args_kwargs(
    sig: inspect.Signature,
    args: Tuple[Any, ...],
    kwargs: dict[str, Any],
    normalize_to_only_use_kwargs: bool,
) -> Optional[ArgsKwargsPair]:
    supported_parameter_types = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
        return None

    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    new_kwargs: dict[str, Any] = {}
    new_args: list[Any] = []
    for i, param in enumerate(sig.parameters):
        if not normalize_to_only_use_kwargs and i < len(args):
            new_args.append(bound_args.arguments[param])
        else:
            new_kwargs[param] = bound_args.arguments[param]

    return ArgsKwargsPair(tuple(new_args), new_kwargs)


def normalize_args_kwargs(target: Callable, args: Tuple[Any], kwargs: dict[str, Any]):
    """Fill in default values for optional args, which are dependent on the schema."""
    sig = inspect.signature(target)
    _, new_kwargs = _args_kwargs_to_normalized_args_kwargs(
        sig, args, kwargs, normalize_to_only_use_kwargs=True
    )
    if "self" in new_kwargs:
        new_kwargs["input"] = new_kwargs.pop("self")

    # Flatten lists of args for ops that takes lists, such as torch.cat.
    to_remove = set()
    to_add = {}
    for k, v in new_kwargs.items():
        if isinstance(v, (tuple, list)) and len(v) and isinstance(v[0], torch.Tensor):
            to_remove.add(k)
            for i, vv in enumerate(v):
                to_add[f"{k}_flattened_{i}"] = vv

    for rem in to_remove:
        del new_kwargs[rem]
    new_kwargs.update(**to_add)

    # Sort here in order to have consistency across TS graph and
    # MLIR module.
    sorted_kwargs = dict(sorted(new_kwargs.items()))
    return immutable_collections.immutable_dict(sorted_kwargs)


def str_to_ty(name):
    # if name[0] == "*":
    #     ty = str_to_ty(name[1:])
    #     return shark.language.pointer_type(ty)
    # tys = {
    #     "fp8": shark.language.float8,
    #     "fp16": shark.language.float16,
    #     "bf16": shark.language.bfloat16,
    #     "fp32": shark.language.float32,
    #     "fp64": shark.language.float64,
    #     "i1": shark.language.int1,
    #     "i8": shark.language.int8,
    #     "i16": shark.language.int16,
    #     "i32": shark.language.int32,
    #     "i64": shark.language.int64,
    #     "u8": shark.language.uint8,
    #     "u16": shark.language.uint16,
    #     "u32": shark.language.uint32,
    #     "u64": shark.language.uint64,
    #     "B": shark.language.int1,
    # }
    return tys[name]


class TorchMlirType(abc.ABC):
    """
    A `TorchMlirType` is an object that produces MLIR
    types in the `torch` dialect. The only requirement
    for a class to be a subclass of `TorchMlirType`  is
    to define a `to_mlir(self, ir.Context) -> ir.Type`.
    Each class is allowed to have different types of
    __init__ methods depending on the information they
    require to produce the given MLIR representation.
    """

    @abc.abstractmethod
    def to_mlir(self, context: ir.Context) -> ir.Type:
        pass


class TorchTensorTypeError(Exception):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return self.value


class TorchTensorType(TorchMlirType):
    """
    This class is used to generate types of the form
    !torch.tensor and !torch.vtensor<SHAPE, DTYPE>,
    where SHAPE is a list representing the shape of the tensor,
    and DTYPE is an MLIR data type.
    """

    def __init__(
        self,
        *,
        shape: Optional[Iterable[Optional[int]]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.shape = shape
        self.dtype = dtype

        if dtype is None and shape is not None:
            err = "If shape is specified, dtype must also be specified"
            raise TorchTensorTypeError(err)

    def __str__(self):
        return f"Torch Tensor (shape={self.shape}, dtype={self.dtype})"

    def to_mlir(self, context: ir.Context) -> ir.Type:
        if self.dtype is None:
            return ir.Type.parse("!torch.tensor", context=context)

        shape_asm = self._shape_to_mlir_asm()
        dtype_asm = self._dtype_to_mlir_asm()
        return ir.Type.parse(
            f"!torch.vtensor<{shape_asm},{dtype_asm}>", context=context
        )

    def _shape_to_mlir_asm(self) -> str:
        if self.shape is None:
            return "*"

        str_sizes = map(lambda x: "?" if x is None else str(x), self.shape)
        return f'[{",".join(str_sizes)}]'

    def _dtype_to_mlir_asm(self) -> str:
        if self.dtype in [torch.float64]:
            return "f64"
        if self.dtype in [torch.float, torch.float32]:
            return "f32"
        if self.dtype in [torch.int, torch.int32]:
            return "si32"
        if self.dtype in [torch.int64]:
            return "si64"
        if self.dtype in [torch.bool]:
            return "i1"

        raise NotImplementedError(f"Unsupported dtype: {self.dtype}")


class TorchNnModuleType(TorchMlirType):
    """This class is used to generate types for `!torch.nn.Module`s."""

    def __init__(self, module_name: str):
        self.module_name = module_name

    def __str__(self):
        return "torch.nn.Module"

    def to_mlir(self, context: ir.Context) -> ir.Type:
        return ir.Type.parse(f'!torch.nn.Module<"{self.module_name}">', context=context)


class PythonType(TorchMlirType):
    """
    This class is used to convert regular Python types
    into their corresponding `torch` dialect representation.
    The list of supported types can be found in the dictionary
    `_type_to_asm_dict`.
    """

    _type_to_asm_dict = {
        bool: "!torch.bool",
        int: "!torch.int",
        type(None): "!torch.none",
    }

    def __init__(self, type_: Any):
        self.type_ = type_

    def __str__(self):
        return str(self.type_)

    def to_mlir(self, context: ir.Context) -> ir.Type:
        asm = self._type_to_asm_dict.get(self.type_)
        if asm is None:
            raise NotImplementedError(f"Unsupported type: {self.type_}")
        return ir.Type.parse(asm, context=context)


def map_to_mlir_type(t):
    PythonType(t) if isinstance(t, type) else t
