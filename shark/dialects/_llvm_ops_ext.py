#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from ._ods_common import get_default_loc_context as _get_default_loc_context

  import inspect

  from typing import Any, List, Optional, Sequence, Union
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e
