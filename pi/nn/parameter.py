import functools
import inspect
import warnings
from functools import partial
from typing import Union, List, Tuple


from pi import Tensor, dtype, empty
import pi


class Parameter(Tensor):
    def __repr__(self):
        return "Parameter containing:\n" + super(Parameter, self).__repr__()


class Uninitialized(partial):
    size: Union[List[int], Tuple[int, ...]]
    dtype_: dtype = None
    optional: bool = False

    def __new__(cls, *args, **keywords):
        if isinstance(args[0], Tensor):
            return args[0]
        else:
            func = args[0]
            if inspect.isfunction(func) or isinstance(func, functools.partial):
                args = args[1:]
            else:
                func = empty

            if isinstance(args[0], (tuple, list)):
                assert len(args) == 1, f"unknown len args {args}"
                args = args[0]

            assert all([isinstance(a, int) for a in args]), f"{args}"
            instance = super(Uninitialized, cls).__new__(cls, func, *args, **keywords)
            instance.size = args
            if "dtype_" in keywords and keywords["dtype_"] is not None:
                dtype_ = keywords["dtype_"]
                if not isinstance(dtype_, dtype):
                    warnings.warn(
                        f"unknown dtype_ {type(dtype_).__module__}.{type(dtype_).__name__} (should be {dtype.__module__}.{dtype.__name__})"
                    )
                instance.dtype_ = dtype_

        instance.optional = keywords.get("optional", False)
        return instance

    def fill_(self, val):
        pass

    def zero_(self):
        self.__setstate__((pi.zeros, self.args, self.keywords, self.__dict__))

    def ones_(self):
        self.__setstate__((pi.ones, self.args, self.keywords, self.__dict__))

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        # this is a little hack to make the numpy factory functions
        # in mlir/utils.py work
        args = (*self.args, *args)
        return self.func(args, **keywords)


class UninitializedParameter(Uninitialized):
    cls_to_become = Parameter


class UninitializedBuffer(Uninitialized):
    cls_to_become = Tensor


def is_uninitialized(
    v: Union[Tensor, Parameter, UninitializedBuffer, UninitializedParameter]
) -> bool:
    return isinstance(v, Uninitialized) and not v.optional
