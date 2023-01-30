from typing import Dict, Any, Optional


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: Any) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(
                f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead."
            )
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: str, actual: Optional[Any], expected: Any) -> Any:
    if actual is not None:
        if actual != expected:
            raise ValueError(
                f"The parameter '{param}' expected value {expected} but got {actual} instead."
            )
    return expected
