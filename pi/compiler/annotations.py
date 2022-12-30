# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import functools
import inspect
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import pi
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import (
    get_ods_type,
)
from torch_mlir import ir

PI_EXPORT_ATTR_NAME = "_PI_EXPORT"
PI_ARG_ANNOTATIONS_ATTR_NAME = "_PI_ARG_ANNOTATIONS"


def export(fn):
    # setattr(fn, PI_EXPORT_ATTR_NAME, True)
    return fn


ArgAnnotation = Union[type, Tuple[List[int], pi.dtype]]


class TensorPlaceholder:
    def __init__(self, shape: List[int], dtype: pi.dtype):
        self.shape = shape
        self.dtype = dtype

    def to_value_tensor_type(self):
        dtype = self.dtype.to_mlir_type()
        type = ir.Type.parse(
            f"!torch.vtensor<[{','.join(map(str, self.shape))}],{dtype}>"
        )
        return type

    def to(self, dtype: pi.dtype):
        self.dtype = dtype
        return self

    def type(self, dtype):
        return self.to(dtype)

    def bool(self):
        return self.to(pi.dtype.bool)

    def double(self):
        self.dtype = pi.dtype.float64
        return self


def annotations_to_placeholders(
    args: List[str], annotations: List[Optional[ArgAnnotation]]
) -> OrderedDict:
    placeholders = OrderedDict()
    for annotation, arg in zip(annotations, args):
        # Skip the "self" annotation.
        if annotation is None:
            assert arg == "self"
            continue
        shape, dtype, value_tensor = annotation
        assert value_tensor, f"non-value tensors not supported {arg}"
        placeholders[arg] = TensorPlaceholder(annotation[0], annotation[1])
    return placeholders


# TODO: Replace with py3 extended argument annotations when available.
# See https://www.python.org/dev/peps/pep-0593/
def annotate_args(annotations: List[Optional[ArgAnnotation]]):
    def decorator(fn):
        arg_spec = inspect.getfullargspec(fn)
        placeholders = annotations_to_placeholders(arg_spec.args, annotations)
        setattr(fn, "__placeholders__", placeholders)
        return fn

    return decorator


def convert_annotations_to_placeholders(forward_method):
    """Converts the annotations on a forward method into tensor placeholders.

    These placeholders are suitable for being passed to `torch_mlir.compile`.
    """
    annotations = getattr(forward_method, PI_ARG_ANNOTATIONS_ATTR_NAME)
    placeholders = []
    # Skip the "self" annotation.
    for annotation in annotations[1:]:
        placeholders.append(TensorPlaceholder(annotation[0], annotation[1]))
    return placeholders


def pipile(annotations: List[Optional[ArgAnnotation]]):
    def actual_decorator(func):
        func = export(func)
        if len(annotations):
            func = annotate_args(annotations)(func)
        return func

    return actual_decorator
