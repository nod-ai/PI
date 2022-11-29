from typing import Any, Dict, Tuple, List


__all__ = ["immutable_list", "immutable_dict"]

from shark.compiler.pytree import _register_pytree_node


def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(
        f"'{type(self).__name__}' object does not support mutation."
    )


def _create_immutable_container(base, mutable_functions):
    container = type("immutable_" + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container


immutable_list = _create_immutable_container(
    list,
    [
        "__delitem__",
        "__iadd__",
        "__imul__",
        "__setitem__",
        "append",
        "clear",
        "extend",
        "insert",
        "pop",
        "remove",
    ],
)
immutable_list.__reduce__ = lambda self: (immutable_list, (tuple(iter(self)),))


immutable_dict = _create_immutable_container(
    dict, ["__delitem__", "__setitem__", "clear", "pop", "popitem", "update"]
)
immutable_dict.__reduce__ = lambda self: (immutable_dict, (iter(self.items()),))


# Register immutable collections for PyTree operations


def _immutable_dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Any]:
    return list(d.values()), list(d.keys())


def _immutable_dict_unflatten(values: List[Any], Any: Any) -> Dict[Any, Any]:
    return immutable_dict({key: value for key, value in zip(Any, values)})


def _immutable_list_flatten(d: List[Any]) -> Tuple[List[Any], Any]:
    return d, None


def _immutable_list_unflatten(values: List[Any], Any: Any) -> List[Any]:
    return immutable_list(values)


_register_pytree_node(
    immutable_dict, _immutable_dict_flatten, _immutable_dict_unflatten
)
_register_pytree_node(
    immutable_list, _immutable_list_flatten, _immutable_list_unflatten
)
