import inspect
from functools import partial
from typing import Union, List, Tuple

# this is the right way to import in order to not screw up the tests (torch.dtype vs shark.type)
from ..types_ import dtype as shark_dtype

# import shark
from .._tensor import Tensor, empty


class Parameter(Tensor):
    def __repr__(self):
        return "Parameter containing:\n" + super(Parameter, self).__repr__()


class Uninitialized(partial):
    size: Union[List[int], Tuple[int, ...]]
    dtype: shark_dtype = None

    def __new__(cls, *args, **keywords):
        func = args[0]
        if not inspect.isfunction(func):
            func = empty
        else:
            args = args[1:]

        if isinstance(args[0], (tuple, list)):
            assert len(args) == 1, f"unknown len args {args}"
            args = args[0]

        assert all([isinstance(a, int) for a in args]), f"{args}"
        instance = super(Uninitialized, cls).__new__(cls, func, *args, **keywords)
        instance.size = args
        if "dtype" in keywords and keywords["dtype"] is not None:
            dtype = keywords["dtype"]
            assert isinstance(
                dtype, shark_dtype
            ), f"unknown dtype {type(dtype).__module__}.{type(dtype).__name__} (should be {shark_dtype.__module__}.{shark_dtype.__name__})"
            instance.dtype = dtype

        return instance

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        args = (*self.args, *args)
        return self.func(args, **keywords)


class UninitializedParameter(Uninitialized):
    cls_to_become = Parameter


class UninitializedBuffer(Uninitialized):
    cls_to_become = Tensor


def is_uninitialized(
    v: Union[Tensor, Parameter, UninitializedBuffer, UninitializedParameter]
) -> bool:
    return isinstance(v, Uninitialized)
