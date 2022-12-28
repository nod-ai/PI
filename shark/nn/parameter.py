import inspect
from functools import partial
from typing import Union

import shark
from shark import Tensor


class Parameter(shark.Tensor):
    def __repr__(self):
        return "Parameter containing:\n" + super(Parameter, self).__repr__()


class Uninitialized(partial):
    def __new__(cls, *args, **keywords):
        func = args[0]
        if not inspect.isfunction(func):
            func = shark.empty
        else:
            args = args[1:]
        # TODO(max): set size here
        return super(Uninitialized, cls).__new__(cls, func, *args, **keywords)

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)


class UninitializedParameter(Uninitialized):
    cls_to_become = Parameter


class UninitializedBuffer(Uninitialized):
    cls_to_become = shark.Tensor


def is_uninitialized(
    v: Union[Tensor, Parameter, UninitializedBuffer, UninitializedParameter]
) -> bool:
    return isinstance(v, Uninitialized)
