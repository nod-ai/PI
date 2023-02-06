import re
import warnings
from itertools import chain
from typing import Any, OrderedDict

import numpy as np
import torch
from torch_mlir import ir, OutputType
from torch_mlir.dialects import torch as torch_dialect, func as func_dialect

from .framework import Test, TestUtils
from ..mlir.utils import run_pipeline_with_repro_report, cm
from .._tensor import Tensor

FIXED = np.linspace(0, 0.1, 101)


def set_weights(
    mod, typ=torch.float32, val=1, requires_grad=False, fixed=False, random=False
):
    import torch
    from torch import nn

    for m in mod.modules():
        if hasattr(m, "weight"):
            if fixed:
                m.weight = torch.nn.Parameter(
                    torch.from_numpy(
                        np.random.choice(FIXED, m.weight.numel())
                        .astype(np.float16, casting="unsafe")
                        .reshape(m.weight.shape)
                    ).type(typ),
                    requires_grad=requires_grad,
                )
            elif random:
                nn.init.constant_(m.weight, np.random.randint(1, 100))
                m.weight.requires_grad_(False)
                m.weight = torch.nn.Parameter(
                    m.weight.type(typ), requires_grad=requires_grad
                )
            else:
                nn.init.constant_(m.weight, val)
                m.weight.requires_grad_(False)
                m.weight = torch.nn.Parameter(
                    m.weight.type(typ), requires_grad=requires_grad
                )
        if hasattr(m, "bias") and m.bias is not None:
            if fixed:
                m.bias = torch.nn.Parameter(
                    torch.from_numpy(
                        np.random.choice(FIXED, m.bias.numel())
                        .astype(np.float16, casting="unsafe")
                        .reshape(m.bias.shape)
                    ).type(typ),
                    requires_grad=requires_grad,
                )
            elif random:
                nn.init.constant_(m.bias, np.random.randint(1, 100))
                m.bias.requires_grad_(False)
                m.bias = torch.nn.Parameter(
                    m.bias.type(typ), requires_grad=requires_grad
                )
            else:
                nn.init.constant_(m.bias, val)
                m.bias.requires_grad_(False)
                m.bias = torch.nn.Parameter(
                    m.bias.type(typ), requires_grad=requires_grad
                )


class TorchDialectConfig:
    import torch

    """Base class for TestConfig's that are implemented with linalg-on-tensors.

    This class handles all the common lowering that torch-mlir does before
    reaching the linalg-on-tensors abstraction level.
    """

    def compile(self, program: torch.nn.Module, output_type=OutputType.TORCH) -> Any:
        from torch_mlir_e2e_test.utils import convert_annotations_to_placeholders
        import torch_mlir

        example_args = convert_annotations_to_placeholders(program.forward)
        module = torch_mlir.compile(program, example_args, output_type=output_type)

        return module


SMOKE_TEST = False


class PIConfig:
    def compile(self, test_case: Test) -> Any:
        tu = TestUtils()
        with cm() as module:
            test_module = test_case.program_factory()
            module_name = "torch.debug_module_name"
            module.operation.attributes[module_name] = ir.StringAttr.get(
                test_module.__class__.__name__ + ("SMOKE_TEST" if SMOKE_TEST else "")
            )
            # TODO(max): for some reason updated __call__ doesn't stick
            # (setattr doesn't work, gives 'method' object has no attribute '__annotations__'
            placeholders = test_module.forward.__dict__["__placeholders__"]
            if placeholders:
                assert isinstance(placeholders, OrderedDict)
            func_op = func_dialect.FuncOp(
                name="forward",
                type=(
                    [p.to_nonvalue_tensor_type() for p in placeholders.values()],
                    [],
                ),
                # visibility="private",
            )

            arg_attrs = [p.to_value_tensor_type_bound() for p in placeholders.values()]
            func_op.arg_attrs = ir.ArrayAttr.get(arg_attrs)
            func_op_entry_block = func_op.add_entry_block()
            block_args = list(map(Tensor, func_op.arguments))

            def replace_block_args(self_, *args, **kwargs):
                assert not kwargs, f"kwargs not supported {kwargs}"
                assert len(args) == len(block_args)
                return block_args, kwargs

            test_module.register_forward_pre_hook(replace_block_args, prepend=True)

            def move_buffers_params_into_func(self_, *args, **kwargs):
                for child in self_.all_children():
                    for thing in chain(
                        child._buffers.values(), child._parameters.values()
                    ):
                        if isinstance(thing, Tensor):
                            mlir_val = ir.Value._CAPICreate(thing._CAPIPtr)
                            ir.InsertionPoint.at_block_begin(
                                func_op_entry_block
                            ).insert(mlir_val.owner.detach_from_parent())

            test_module.register_forward_pre_hook(
                move_buffers_params_into_func, prepend=True
            )

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

            test_module.register_forward_post_hook(collect_results, prepend=True)

            with ir.InsertionPoint.at_block_begin(func_op_entry_block):
                test_case.program_invoker(test_module, tu)
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
                    res_type = ir.Type.parse(
                        f"!torch.tuple<{', '.join(el_types)}>"
                    )
                    results = [
                        torch_dialect.PrimTupleConstructOp(res_type, results).result
                    ]

                canonical_func_type = ir.FunctionType.get(
                    inputs=[b.type for b in block_args],
                    results=[r.type for r in results],
                )
                func_op.attributes["function_type"] = ir.TypeAttr.get(
                    canonical_func_type
                )

                func_dialect.ReturnOp(results)

        return module


def lower_torch_mlir_to_linalg(module):
    run_pipeline_with_repro_report(
        module,
        "builtin.module("
        + ",".join(
            [
                "cse",
                # "builtin.module(torchscript-module-to-torch-backend-pipeline)",
                "torch-backend-to-linalg-on-tensors-backend-pipeline",
            ]
        )
        + ")",
        # "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        "Lowering Torch Backend IR to Linalg",
    )
    return module
