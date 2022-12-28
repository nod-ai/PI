import difflib
from collections import OrderedDict
from typing import Any

# this is just to prevent pycharm from shuffling this below torch_mlir
try:
    import shark
except:
    import shark

# Available test configs.
import numpy as np
import torch
from torch_mlir import ir
from torch_mlir.dialects import func
from torch_mlir_e2e_test.configs import TorchScriptTestConfig
from torch_mlir_e2e_test.registry import (
    GLOBAL_TEST_REGISTRY as TORCH_MLIR_GLOBAL_TEST_REGISTRY,
)
from torch_mlir_e2e_test.test_suite import (
    register_all_tests as torch_mlir_register_all_tests,
)

from shark.compiler.mlir_utils import (
    run_pipeline_with_repro_report,
    lower_torch_mlir_to_linalg as lower_sharkpy_to_linalg,
)
from tests.framework import TestCase, TestUtils
from tests.test_suite import register_all_tests as sharkpy_register_all_tests
from tests.registry import GLOBAL_TEST_REGISTRY as SHARKPY_GLOBAL_TEST_REGISTRY

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

    def compile(self, program: torch.nn.Module) -> Any:
        from torch_mlir_e2e_test.utils import convert_annotations_to_placeholders
        import torch_mlir

        example_args = convert_annotations_to_placeholders(program.forward)
        module = torch_mlir.compile(program, example_args)

        return module


class SharkPyConfig:
    def compile(self, test_case: TestCase, test_module: shark.nn.Module) -> Any:
        tu = TestUtils()
        with shark.mlir_cm() as module:
            module.operation.attributes["torch.debug_module_name"] = ir.StringAttr.get(
                test_module.__class__.__name__
            )
            # TODO(max): for some reason updated __call__ doesn't stick
            # (setattr doesn't work, gives 'method' object has no attribute '__annotations__'
            placeholders = test_module.forward.__dict__["__placeholders__"]
            if placeholders:
                assert isinstance(placeholders, OrderedDict)
            func_op = func.FuncOp(
                name="forward",
                type=(
                    [p.to_value_tensor_type() for p in placeholders.values()],
                    [],
                ),
                # visibility="public",
            )
            func_op_entry_block = func_op.add_entry_block()
            block_args = list(map(shark.Tensor, func_op.arguments))

            def replace_block_args(self_, *args, **kwargs):
                assert not kwargs, f"kwargs not supported {kwargs}"
                assert len(args) == len(block_args)
                return block_args, kwargs

            test_module.register_forward_pre_hook(replace_block_args, prepend=True)

            with ir.InsertionPoint.at_block_begin(func_op_entry_block):
                ret = test_case(test_module, tu)
                canonical_func_type = ir.FunctionType.get(
                    inputs=[b.type for b in block_args], results=[ret.type]
                )
                func_op.attributes["function_type"] = ir.TypeAttr.get(
                    canonical_func_type
                )
                func.ReturnOp([ret])

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
        "Lowering TorchScript IR -> Torch Backend IR",
    )
    return module


# Import tests to register them in the global registry.
sharkpy_register_all_tests()
torch_mlir_register_all_tests()

TORCH_MLIR_GLOBAL_TEST_REGISTRY = {
    t.unique_name: t for t in TORCH_MLIR_GLOBAL_TEST_REGISTRY
}

torchscript_config = TorchScriptTestConfig()
sharkpy_config = SharkPyConfig()
torch_dialect_config = TorchDialectConfig()
tests = [test for test in SHARKPY_GLOBAL_TEST_REGISTRY]
for test in tests:
    print(test.unique_name)
    torch_mlir_test = TORCH_MLIR_GLOBAL_TEST_REGISTRY[test.unique_name]
    mod = torch_mlir_test.program_factory()
    set_weights(mod)
    mod.eval()
    torch_mlir_module = torch_dialect_config.compile(mod)
    torch_mlir_linalg_module_str = str(lower_torch_mlir_to_linalg(torch_mlir_module))

    test_module = test.module_factory()
    try:
        sharkpy_mlir_module = sharkpy_config.compile(test.test_case, test_module)
    except:
        torch_script_compiled = torchscript_config.compile(mod)
        frozen = torch.jit.freeze(torch_script_compiled)
        torch_mlir_module = torch_dialect_config.compile(mod)
        print("frozen.graph\n", frozen.graph)
        print("torch_mlir_module\n", torch_mlir_module)
        raise

    sharkpy_mlir_linalg_module_str = str(lower_sharkpy_to_linalg(sharkpy_mlir_module))

    diff = list(
        difflib.unified_diff(
            str(sharkpy_mlir_linalg_module_str).splitlines(),
            str(torch_mlir_linalg_module_str).splitlines(),
            lineterm="",
        )
    )

    if len(diff):
        print(f"\n{''.join('*' * 10)}\ndiff\n{''.join('*' * 10)}\n")
        print("\n".join(diff))
        print()
        print("torch_mlir_linalg_module_str:\n", torch_mlir_linalg_module_str)
        print("sharkpy_mlir_linalg_module_str:\n", sharkpy_mlir_linalg_module_str)

        torch_script_compiled = torchscript_config.compile(mod)
        frozen = torch.jit.freeze(torch_script_compiled)
        torch_mlir_module = torch_dialect_config.compile(mod)
        print("frozen.graph\n", frozen.graph)
        print("torch_mlir_module\n", torch_mlir_module)
    else:
        print("PASS")
