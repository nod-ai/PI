from typing import Any, Callable, NamedTuple, Union

import numpy as np
from torch_mlir import ir

from ..types_ import float32, int64
from .. import nn
from ..utils.annotations import TensorPlaceholder


# Utilities for common testing trace generation.
# Also, resets the random seed for reproducibility.
# TODO: If generating in parallel, how to have manual_seed be local?


class TestUtils:
    def __init__(self):
        np.random.seed(0)

    # TODO: Add zeros/ones/etc. as convenient.
    def rand(self, *sizes, low=0.0, high=1.0):
        # return uniform(low, high, sizes)
        return TensorPlaceholder(sizes, dtype=float32)

    def randn(self, *sizes):
        # return randn(low, high, sizes)
        return TensorPlaceholder(sizes, dtype=float32)

    def randint(self, *sizes, low=0, high=10, dtype=int64):
        return TensorPlaceholder(sizes, dtype=dtype)


TestResult = Union[ir.OpView, ir.Operation, ir.Value, ir.OpResultList]


class Test(NamedTuple):
    unique_name: str
    program_factory: Callable[[], nn.Module]
    program_invoker: Callable[[Any, TestUtils], None]
