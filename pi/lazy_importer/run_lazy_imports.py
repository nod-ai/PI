import sys
from textwrap import dedent

from pi.lazy_importer.lazy_imports import (
    LazyImportTracer,
    imports_file,
    import_uses_file,
    nn_modules_file,
    classes_file,
    capture_all_calls,
    functions_file,
)


class LazyImports(LazyImportTracer):
    def should_instrument_file(self, filename: str) -> bool:
        return not filename.endswith("lazy_imports.py")


# /home/mlevental/mambaforge/envs/PI/lib/python3.11/site-packages/pyccolo/ast_rewriter.py:161
PREFIX = f"""\
from __future__ import annotations
import functools
import importlib
import inspect
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import partial
from pathlib import PosixPath
from typing import Optional, Callable, Tuple, Union, Dict, List, Any, OrderedDict
import sys
import warnings

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
logger = logging.getLogger(__name__)

"""


def do_hand_imports(closure, closure2):
    with LazyImports.instance():
        closure()
    with open("imports.py", "w") as f:
        for n in sorted(imports_file):
            print(n, file=f, flush=True)
        for n in sorted(import_uses_file):
            print(
                dedent(
                    f"""\
            try: {n}()
            except: pass
            """
                ),
                file=f,
                flush=True,
            )
    with LazyImports.instance():
        closure2()


def do_package_imports(closure, prefix, name):
    with LazyImports.instance():
        closure()

    with open(f"{name}.py", "w") as f:
        print(PREFIX, file=f, flush=True)
        print(prefix, file=f, flush=True)
        for n in sorted(functions_file | classes_file | nn_modules_file):
            print(n, file=f, flush=True)
