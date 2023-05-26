# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import inspect
import warnings
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
from pi import dtype, TensorPlaceholder

PI_EXPORT_ATTR_NAME = "_PI_EXPORT"
PI_ARG_ANNOTATIONS_ATTR_NAME = "_PI_ARG_ANNOTATIONS"


def export(fn):
    return fn


ArgAnnotation = Union[type, Tuple[List[int], dtype]]


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
        if not shape:
            warnings.warn(f"empty shape annotation: {shape}")
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
