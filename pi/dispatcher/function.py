import functools
from textwrap import indent
from typing import Tuple, Any, Optional, Dict

from typeguard import (
    check_argument_types,
    CallMemo,
    TypeCheckError,
)


class AmbiguousLookupError(LookupError):
    """A signature cannot be resolved due to ambiguity."""


class NotFoundLookupError(LookupError):
    """A signature cannot be resolved because no applicable method can be found."""


class Function:
    def __init__(self, f):
        self._f = f

        self._cache = {}
        self._overloads = {}

        # self.__name__ = "Dispatcher" + f.__name__
        # self.__qualname__ = "Dispatcher" + f.__qualname__
        # self.__module__ = "Dispatcher" + f.__module__
        self.__name__ = f.__name__
        self.__qualname__ = f.__qualname__
        self.__module__ = f.__module__

    def register(
        self,
        signature,
        f,
        precedence=0,
    ):
        assert (
            signature not in self._overloads
        ), f"{signature} already registered {self._overloads[signature]}"
        self._overloads[signature] = (f, precedence)

    def resolve_signature(
        self,
        args: Tuple[Any],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        for candidate_signature, (func, _precedence) in self._overloads.items():
            try:
                candidate_signature.bind(*args, **kwargs)
                memo = CallMemo(func, {}, args=args, kwargs=kwargs)
                check_argument_types(memo)
                return candidate_signature
            except (TypeError, TypeCheckError) as e:
                continue

        args = indent("\n".join(map(str, args)), "\t\t")
        kwargs = indent("\n".join(map(str, kwargs.items())), "\t\t")
        candidates = indent("\n".join(map(str, list(self._overloads.keys()))), "\t\t")
        raise NotFoundLookupError(
            f"Tried to normalize arguments to `{self._f.__name__}` but failed for\n args:\n{args}\n kwargs:\n{kwargs}\n tried candidates:\n{candidates}\n"
        )

    def resolve_overload(self, signature):
        f, _ = self._overloads[signature]
        return f

    def __call__(self, *args, **kwargs):
        signature = self.resolve_signature(args, kwargs)
        f = self.resolve_overload(signature)
        return f(*args, **kwargs)

    def __repr__(self):
        return f"<function {self._f} with " f"{len(self._overloads)} overload(s)>"


class ClassFunction:
    _pending = []

    def __init__(self, construct_function):
        self.function = construct_function

    def __get__(self, instance, owner):
        return functools.partial(self.function, instance)
