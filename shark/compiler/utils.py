# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import ast
import os
import sys
import tempfile
from io import StringIO

from shark._mlir_libs._mlir.ir import (
    Operation,
    IntegerType,
)
from shark._mlir_libs._mlir.passmanager import PassManager
from shark.dialects import arith

ONE_SHOT_BUFFERIZATION_PIPELINE = [
    "func.func(linalg-init-tensor-to-alloc-tensor)",
    "one-shot-bufferize",
    "func-bufferize",
    "arith-bufferize",
    "func.func(finalizing-bufferize)",
]

LOWERING_PIPELINE = [
    # Lower to LLVM
    "convert-scf-to-cf",
    # "func.func(refback-expand-ops-for-llvm)",
    "func.func(arith-expand)",
    "func.func(convert-math-to-llvm)",
    # Handle some complex mlir::math ops (e.g. atan2)
    "convert-math-to-libm",
    "convert-linalg-to-llvm",
    "convert-memref-to-llvm",
    "func.func(convert-arith-to-llvm)",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
]


def run_pipeline_with_repro_report(module, pipeline: str, description: str = None):
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True
        )
        # Lower module in place to make it ready for compiler backends.
        with module.context:
            pm = PassManager.parse(pipeline)
            pm.run(module)
    except Exception as e:
        filename = os.path.join(tempfile.gettempdir(), "tmp.mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        description = description or f"tmp compile"

        message = f"""\
            {description} failed with the following diagnostics:
            {sys.stderr.getvalue()}

            For MLIR developers, the error can be reproduced with:
            $ mlir-opt -pass-pipeline='{pipeline}' {filename}
            Add '{debug_options}' to get the IR dump for debugging purpose.
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise Exception(trimmed_message) from None
    finally:
        sys.stderr = original_stderr


def traverse_op_region_block_iterators(op, handler):
    for i, region in enumerate(op.regions):
        for j, block in enumerate(region):
            for k, child_op in enumerate(block):
                res = handler(child_op)
                if res is not None and isinstance(res, Exception):
                    return res
                res = traverse_op_region_block_iterators(child_op, handler)
                if res is not None and isinstance(res, Exception):
                    return res


def parse_attrs_to_dict(attrs):
    d = {}
    for named_attr in attrs:
        if named_attr.name in {"lpStartTime", "value"}:
            d[named_attr.name] = ast.literal_eval(
                str(named_attr.attr).split(":")[0].strip()
            )
        elif named_attr.name in {"opr"}:
            d[named_attr.name] = ast.literal_eval(str(named_attr.attr))
        else:
            d[named_attr.name] = ast.literal_eval(str(named_attr.attr).replace('"', ""))
    return d


def make_i32_int(x):
    return arith.ConstantOp(IntegerType.get_signless(32), x).result


def add_dummy_value():
    return Operation.create(
        "custom.value", results=[IntegerType.get_signless(32)]
    ).result
