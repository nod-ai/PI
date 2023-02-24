import builtins
import inspect
import logging
from typing import Any, Tuple, Optional

from typeguard import (
    TypeCheckMemo,
    TypeCheckError,
    check_type,
    TypeCheckerCallable,
    _config
)

from pi import Torch_Value, Torch_List, Tensor

logger = logging.getLogger(__name__)


def check_simple_torch_value(
    value: Any, origin_type: Any, type_var_args: Tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    assert len(type_var_args) == 1, f"multiple type var args to Torch_Value not handled"
    type_var_arg = type_var_args[0]
    if not isinstance(value, Torch_Value):
        raise TypeCheckError(f"Not Torch_Value: {value}")
    try:
        check_type(value.type, type_var_arg)
    except TypeCheckError:
        raise TypeCheckError(f"Not correct type param ({type_var_arg}): {value.type}")


def check_simple_torch_list(
    value: Any, origin_type: Any, type_var_args: Tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    assert len(type_var_args) == 1, f"multiple type var args to Torch_List not handled"
    type_var_arg = type_var_args[0]
    if not isinstance(value, Torch_List):
        raise TypeCheckError(f"Not Torch_Value: {value}")
    try:
        check_type(value.el_type, type_var_arg)
    except TypeCheckError:
        raise TypeCheckError(
            f"Not correct type param ({type_var_arg}): {value.el_type}"
        )


def check_tensor(
    value: Any, origin_type: Any, type_var_args: Tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    if not isinstance(value, Tensor):
        raise TypeCheckError(f"Not Torch_Tensor: {value}")


def check_number(
    value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    if origin_type is builtins.complex and not isinstance(
        value, (builtins.complex, builtins.float, builtins.int)
    ):
        raise TypeCheckError("is neither complex, float or int")
    elif origin_type is builtins.int and isinstance(value, builtins.bool):
        raise TypeCheckError("is neither float or int")
    elif origin_type is builtins.float and not isinstance(value, builtins.float):
        raise TypeCheckError("is neither float or int")
    elif origin_type is builtins.int and not isinstance(value, builtins.int):
        raise TypeCheckError("is neither float or int")


def torch_type_checker_lookup(
    origin_type: Any, type_var_args: Tuple[Any, ...], extras: Tuple[Any, ...]
) -> Optional[TypeCheckerCallable]:
    if inspect.isclass(origin_type) and issubclass(origin_type, Torch_List):
        return check_simple_torch_list
    elif inspect.isclass(origin_type) and issubclass(origin_type, Torch_Value):
        return check_simple_torch_value
    elif inspect.isclass(origin_type) and issubclass(origin_type, Tensor):
        return check_tensor
    elif inspect.isclass(origin_type) and issubclass(
        origin_type, (builtins.bool, builtins.float, builtins.int)
    ):
        return check_number

    return None


_config._config.checker_lookup_functions.insert(0, torch_type_checker_lookup)
