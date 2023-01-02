import inspect
import logging
from typing import Dict, Callable, List
from .function import Function


__all__ = ["Dispatcher", "dispatch"]


log = logging.getLogger(__name__)

_manual_overrides: Dict[Callable, List[inspect.Signature]] = {}


class Dispatcher:
    def __init__(self):
        self._functions = {}
        self._classes = {}

    def __call__(self, overload=None, precedence=0):
        if overload is None:

            def decorator(f_):
                return self(f_, precedence=precedence)

            return decorator

        def construct_function():
            return self._add_overload(
                overload,
                inspect.signature(overload),
                precedence=precedence,
            )

        return construct_function()

    def _get_function(self, overload) -> Function:
        name = overload.__name__
        namespace = self._functions
        if name not in namespace:
            namespace[name] = Function(overload)
        return namespace[name]

    def _add_overload(
        self,
        overload,
        signature,
        precedence,
    ):
        f = self._get_function(overload)
        f.register(signature, overload, precedence)
        return f

    def clear_cache(self):
        for f in self._functions.values():
            f.clear_cache()


dispatch = Dispatcher()
