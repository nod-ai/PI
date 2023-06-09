# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import contextlib
import re
import warnings
from collections import OrderedDict
from io import StringIO
import os
import sys
import tempfile
from itertools import chain

from .dialects import _torch_ops_gen as torch_dialect
from .dialects import torch as torch_dialect, func as func_dialect
from . import ir
from .passmanager import PassManager
from .utils import disable_multithreading

from .. import nn, Tensor


def get_module_name_for_debug_dump(module):
    """Gets a name suitable for a debug dump.

    The name is not guaranteed to be unique.
    """
    if not "pi.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return ir.StringAttr(module.operation.attributes["pi.debug_module_name"]).value


class PIMlirCompilerError(Exception):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return self.value


def run_pipeline_with_repro_report(
    module,
    pipeline: str,
    description: str,
    enable_ir_printing=False,
    print_pipeline=False,
):
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True
        )
        # Lower module in place to make it ready for compiler backends.
        with module.context:
            pm = PassManager.parse(pipeline)
            if print_pipeline:
                print("pass-pipeline=", pm)
            if enable_ir_printing:
                with disable_multithreading():
                    pm.enable_ir_printing()
                    pm.run(module.operation)
            else:
                pm.run(module.operation)
    except Exception as e:
        print(e, file=sys.stderr)
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        # Put something descriptive here even if description is empty.
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:
            {sys.stderr.getvalue()}

            For Torch-MLIR developers, the error can be reproduced with:
            $ torch-mlir-opt -pass-pipeline='{pipeline}' {filename}
            Add '{debug_options}' to get the IR dump for debugging purpose.
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise PIMlirCompilerError(trimmed_message) from None
    finally:
        sys.stderr = original_stderr


def lower_pi_to_linalg(module, enable_ir_printing=False):
    run_pipeline_with_repro_report(
        module,
        "builtin.module("
        + ",".join(
            [
                "symbol-dce",
                "inline{default-pipeline= max-iterations=4}",
                "torch-adjust-calling-conventions",
                "torch-lower-to-backend-contract{decompose=true max-iterations=10}",
                "torch-backend-to-linalg-on-tensors-backend-pipeline",
            ]
        )
        + ")",
        "Lowering Torch MLIR -> Linalg",
        enable_ir_printing,
        print_pipeline=False,
    )
    return module


def lower_pi_to_torch_backend(module, enable_ir_printing=False, print_pipeline=False):
    run_pipeline_with_repro_report(
        module,
        "builtin.module("
        + ",".join(
            [
                "symbol-dce",
                "inline",
                "torch-adjust-calling-conventions",
                "torch-lower-to-backend-contract{decompose=true max-iterations=10}",
            ]
        )
        + ")",
        "Lowering to Torch MLIR backend contract",
        enable_ir_printing,
        print_pipeline=print_pipeline,
    )
    return module


def pipile(pi_module: nn.Module, example_args=None, module_name="pi.module_name"):
    if example_args is None:
        example_args = []

    mlir_module = ir.Module.create()
    mlir_module.operation.attributes[module_name] = ir.StringAttr.get(
        pi_module.__class__.__name__
    )
    with ir.InsertionPoint(mlir_module.body):

        placeholders = pi_module.forward.__dict__.get("__placeholders__")
        if placeholders:
            assert isinstance(placeholders, OrderedDict)
        func_op = func_dialect.FuncOp(
            name="forward",
            type=(
                [p.to_value_tensor_type() for p in placeholders.values()]
                if not example_args
                else [e.to_value_tensor_type() for e in example_args],
                [],
            ),
            # visibility="private",
        )

        if not example_args:
            arg_attrs = [p.to_value_tensor_type_bound() for p in placeholders.values()]
            func_op.arg_attrs = ir.ArrayAttr.get(arg_attrs)

        func_op_entry_block = func_op.add_entry_block()
        block_args = list(map(Tensor, func_op.arguments))

        def replace_block_args(self_, *args, **kwargs):
            assert not kwargs, f"kwargs not supported {kwargs}"
            assert len(args) == len(block_args)
            return block_args, kwargs

        pi_module.register_forward_pre_hook(replace_block_args, prepend=True)

        def move_buffers_params_into_func(self_, *args, **kwargs):
            for child in self_.all_children():
                for thing in chain(child._buffers.values(), child._parameters.values()):
                    if isinstance(thing, Tensor):
                        mlir_val = ir.Value._CAPICreate(thing._CAPIPtr)
                        ir.InsertionPoint.at_block_begin(func_op_entry_block).insert(
                            mlir_val.owner.detach_from_parent()
                        )

        pi_module.register_forward_pre_hook(move_buffers_params_into_func, prepend=True)

        results = []

        def collect_results(_self, result, *_args, **_kwargs):
            if len(results):
                warnings.warn(
                    f"results already collected {results} (new result {result}); overwriting"
                )
                results[0] = result
            else:
                results.append(result)
            return result

        pi_module.register_forward_post_hook(collect_results, prepend=True)

        with ir.InsertionPoint.at_block_begin(func_op_entry_block):
            pi_module(*example_args)
            if isinstance(results[0], (tuple, list)):
                results = results[0]

            assert all(isinstance(r, (Tensor, ir.Value)) for r in results), results
            # functions created from python can't return multiple results
            if len(results) > 1:
                el_type_reg = re.compile(r"!torch\.(.*)")
                el_types = []
                for r in results:
                    el_type = el_type_reg.findall(str(r.type))
                    assert len(el_type) == 1
                    el_types.append(el_type[0])
                res_type = ir.Type.parse(f"!torch.tuple<{', '.join(el_types)}>")
                results = [torch_dialect.PrimTupleConstructOp(res_type, results).result]

            canonical_func_type = ir.FunctionType.get(
                inputs=[b.type for b in block_args],
                results=[r.type for r in results],
            )
            func_op.attributes["function_type"] = ir.TypeAttr.get(canonical_func_type)

            func_dialect.ReturnOp(results)

    mlir_module = lower_pi_to_torch_backend(mlir_module)

    return mlir_module
