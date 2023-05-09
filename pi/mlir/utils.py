# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import contextlib
from typing import Optional

from .ir import Context, Location, Module, InsertionPoint


@contextlib.contextmanager
def mlir_mod_ctx(
    src: Optional[str] = None, context: Context = None, location: Location = None
):
    if context is None:
        context = Context()
    with context:
        if location is None:
            location = Location.unknown()
        with location:
            if src is not None:
                module = Module.parse(src)
            else:
                module = Module.create()
            with InsertionPoint(module.body):
                yield module
