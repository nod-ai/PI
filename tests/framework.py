from typing import Any, Callable, NamedTuple, Union

import numpy as np
from torch_mlir import ir

import shark
from shark.compiler.annotations import TensorPlaceholder


# Utilities for common testing trace generation.
# Also, resets the random seed for reproducibility.
# TODO: If generating in parallel, how to have manual_seed be local?
class TestUtils:
    def __init__(self):
        np.random.seed(0)

    # TODO: Add zeros/ones/etc. as convenient.
    def rand(self, *sizes, low=0.0, high=1.0):
        # return shark.uniform(low, high, sizes)
        return TensorPlaceholder(sizes, dtype=shark.float32)

    def randn(self, *sizes):
        # return shark.randn(low, high, sizes)
        return TensorPlaceholder(sizes, dtype=shark.float32)

    def randint(self, *sizes, low=0, high=10):
        # return shark.randint(low, high, sizes)
        return TensorPlaceholder(sizes, dtype=shark.int64)


TestResult = Union[ir.OpView, ir.Operation, ir.Value, ir.OpResultList]

ModuleFactory = Callable[[], shark.nn.Module]
TestCase = Callable[[Any, TestUtils], TestResult]


class Test(NamedTuple):
    unique_name: str
    module_factory: ModuleFactory
    test_case: TestCase
