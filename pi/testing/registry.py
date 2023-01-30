# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Callable

from pi import nn

from .framework import Test

# The global registry of tests.
GLOBAL_TEST_REGISTRY = {}


def register_test_case(module_factory: Callable[[], nn.Module]):
    def decorator(f):
        # Ensure that there are no duplicate names in the global test registry.
        if f.__name__ in GLOBAL_TEST_REGISTRY:
            raise Exception(
                f"Duplicate test name: '{f.__name__}'. Please make sure that the function wrapped by `register_test_case` has a unique name."
            )

        # Store the test in the registry.
        GLOBAL_TEST_REGISTRY[f.__name__] = Test(
            unique_name=f.__name__,
            program_factory=module_factory,
            program_invoker=f,
        )
        return f

    return decorator
