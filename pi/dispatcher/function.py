from typing import Tuple, Any, Optional, Dict, cast


from .operator_schemas import type_matches, create_type_hint, map_aggregate


class AmbiguousLookupError(LookupError):
    """A signature cannot be resolved due to ambiguity."""


class NotFoundLookupError(LookupError):
    """A signature cannot be resolved because no applicable method can be found."""


class Function:
    def __init__(self, f):
        self._f = f
        self._precedences = {}

        self._cache = {}
        self._overloads = {}

        self.__name__ = "Dispatcher" + f.__name__
        self.__qualname__ = "Dispatcher" + f.__qualname__
        self.__module__ = "Dispatcher" + f.__module__

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
        arg_types: Optional[Tuple[Any]] = None,
        kwarg_types: Optional[Dict[str, Any]] = None,
    ):
        candidates = []
        for candidate_signature in self._overloads:
            try:
                candidate_signature.bind(*args, **kwargs)
                candidates.append(candidate_signature)
            except TypeError as e:
                continue

        if len(candidates) == 0:
            raise NotFoundLookupError(
                f'For function "{self._f.__name__}", '
                f"signature {args}, {kwargs} could not be resolved."
            )
        elif len(candidates) == 1:
            return candidates[0]
        else:
            if arg_types is not None or kwarg_types is not None:
                arg_types = arg_types if arg_types else cast(Tuple[Any], ())
                kwarg_types = kwarg_types if kwarg_types else {}
                for candidate_signature in self._overloads.keys():
                    sig_matches = True
                    try:
                        bound_types = candidate_signature.bind(
                            *arg_types, **kwarg_types
                        )
                        for arg_name, arg_type in bound_types.arguments.items():
                            param = candidate_signature.parameters[arg_name]
                            sig_matches = sig_matches and type_matches(
                                param.annotation, arg_type
                            )
                    except TypeError as e:
                        print(e)
                        sig_matches = False
                    if sig_matches:
                        return candidate_signature
            else:
                # Matched more than one overload. In this situation, the caller must provide the types of
                # the arguments of the overload they expect.
                schema_printouts = "\n".join(str(sig) for sig in candidates)
                raise AmbiguousLookupError(
                    f"Tried to normalize arguments to {self._f.__name__} but "
                    f"the schema match was ambiguous! Please provide argument types to "
                    f"the normalize_arguments() call. Available schemas:\n{schema_printouts}"
                )

        raise AmbiguousLookupError(
            f"Tried to normalize arguments to {self._f.__name__} but failed for {args=} {kwargs=} {arg_types=} {kwarg_types=}; tried {candidates=}"
        )

    def resolve_overload(self, signature):
        f, _ = self._overloads[signature]
        return f

    def __call__(self, *args, **kwargs):
        args_types = map_aggregate(args, type)
        assert isinstance(args_types, tuple)
        args_types = tuple([create_type_hint(i) for i in args_types])
        kwargs_types = {k: type(v) for k, v in kwargs.items()}
        signature = self.resolve_signature(args, kwargs, args_types, kwargs_types)
        f = self.resolve_overload(signature)
        return f(*args, **kwargs)

    def __repr__(self):
        return f"<function {self._f} with " f"{len(self._overloads)} overload(s)>"
