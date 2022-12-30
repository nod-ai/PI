import difflib
import sys
import traceback
from pathlib import Path


# noinspection PyUnresolvedReferences
import pi

# noinspection PyUnresolvedReferences
import torch_mlir
import torch_mlir_e2e_test

from xfail import SHARKPY_XFAIL_SET

shark_package_root_path = Path(pi.__file__).parent
torch_mlir_package_root_path = Path(torch_mlir.__file__).parent
torch_mlir_e2e_test_package_root_path = Path(torch_mlir_e2e_test.__file__).parent

from torch_mlir_e2e_test.test_suite import COMMON_TORCH_MLIR_LOWERING_XFAILS
from torch_mlir_e2e_test.test_suite import (
    register_all_tests as torch_mlir_register_all_tests,
)

from pi.dialects import (
    remove_modules,
    ImportOverload,
    BASE_OVERLOADS,
    patch_meta_path,
)
from pi.mlir_utils import lower_sharkpy_to_linalg
from pi.testing.util import (
    SharkPyConfig,
)
from pi.testing.util import (
    TorchDialectConfig,
    set_weights,
    lower_torch_mlir_to_linalg,
)


def run_torch_mlir_tests():
    torch_mlir_register_all_tests()
    import torch_mlir_e2e_test.registry
    import torch_mlir_e2e_test.framework

    tu = torch_mlir_e2e_test.framework.TestUtils()
    tests = sorted(
        torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY, key=lambda t: t.unique_name
    )

    torch_dialect_config = TorchDialectConfig()
    torch_mlir_linalg_module_strs = {}
    for test in tests:
        if test.unique_name in SHARKPY_XFAIL_SET | COMMON_TORCH_MLIR_LOWERING_XFAILS:
            continue

        mod = test.program_factory()
        set_weights(mod)
        mod.eval()
        torch_mlir_module = torch_dialect_config.compile(mod)
        torch_mlir_linalg_module_strs[test.unique_name] = str(
            lower_torch_mlir_to_linalg(torch_mlir_module)
        )
        # test.program_invoker(mod, tu)

    return torch_mlir_linalg_module_strs


def run_sharkpy_tests(torch_mlir_linalg_module_strs):
    torch_mlir_register_all_tests()
    # after remapping, this imports pi test registry
    import torch_mlir_e2e_test.registry

    tests = sorted(
        torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY, key=lambda t: t.unique_name
    )
    assert tests, "failed to load tests"

    from torch import nn

    assert (
        nn.__spec__.origin == f"{shark_package_root_path}/nn/__init__.py"
    ), f"monkey patch failed {nn.__spec__.origin}"
    # for compatibility
    nn.Module.train = lambda *args, **kwargs: None

    # torchscript_config = TorchScriptTestConfig()
    sharkpy_config = SharkPyConfig()
    torch_dialect_config = TorchDialectConfig()
    PASS, FAIL, TOTAL = 0, 0, 0
    for test in tests:
        if test.unique_name in SHARKPY_XFAIL_SET | COMMON_TORCH_MLIR_LOWERING_XFAILS:
            print(f"skipping {test.unique_name}")
            continue
        print(f"running {test.unique_name}")
        TOTAL += 1

        test_module = test.program_factory()
        torch_mlir_linalg_module_str = torch_mlir_linalg_module_strs[test.unique_name]

        try:
            sharkpy_mlir_module = sharkpy_config.compile(test, test_module)
        except NotImplementedError as e:
            print(traceback.format_exc(-2))
            print(f"{e}")
            print(f"FAIL sharkpy compile NotImplementedError")
            FAIL += 1
            continue
        except Exception as e:
            print(traceback.format_exc())
            print(f"{e}")
            print("\ntorch_mlir module\n")
            print(torch_mlir_linalg_module_str)
            # torch_script_compiled = torchscript_config.compile(mod)
            # frozen = torch.jit.freeze(torch_script_compiled)
            # torch_mlir_module = torch_dialect_config.compile(mod)
            # print("frozen.graph\n", frozen.graph)
            # print("torch_mlir_module\n", torch_mlir_module)
            print(f"FAIL sharkpy compile Exception")
            FAIL += 1
            raise e

        try:
            sharkpy_mlir_linalg_module_str = str(
                lower_sharkpy_to_linalg(sharkpy_mlir_module)
            )
        except Exception as e:
            print(traceback.format_exc())
            print("\ntorch_mlir module\n")
            print(torch_mlir_linalg_module_str)
            print(f"FAIL lower_sharkpy_to_linalg Exception")
            FAIL += 1
            raise e

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
            # print("torch_mlir_linalg_module_str:\n", torch_mlir_linalg_module_str)
            # print("sharkpy_mlir_linalg_module_str:\n", sharkpy_mlir_linalg_module_str)
            print(f"FAIL IR diff")
            FAIL += 1

            # torch_script_compiled = torchscript_config.compile(mod)
            # frozen = torch.jit.freeze(torch_script_compiled)
            # torch_mlir_module = torch_dialect_config.compile(mod)
            # print("frozen.graph\n", frozen.graph)
            # print("torch_mlir_module\n", torch_mlir_module)
        else:
            print("PASS")
            PASS += 1

    print(f"\n{''.join('*' * 10)}\n\n{PASS=} {FAIL=} out of {TOTAL=}\n\n{''.join('*' * 10)}\n")


def main():
    torch_mlir_linalg_module_strs = run_torch_mlir_tests()
    remove_modules(lambda mod: mod.startswith("torch_mlir_e2e_test"))
    remove_modules(lambda mod: mod == "torch" or mod.startswith("torch."))

    # remap to torch so that isintance works...
    # remove_modules(lambda mod: mod == "pi" or mod.startswith("pi."))

    overloads = [
        ImportOverload(
            "torch_mlir_e2e_test.framework",
            shark_package_root_path / "testing/framework.py",
            False,
        ),
        ImportOverload(
            "torch_mlir_e2e_test.registry",
            shark_package_root_path / "testing/registry.py",
            False,
        ),
        ImportOverload(
            "torch_mlir_e2e_test.annotations",
            shark_package_root_path / "compiler/annotations.py",
            False,
        ),
        ImportOverload(
            "torch",
            shark_package_root_path / "__init__.py",
            True,
        ),
        ImportOverload(
            "torch.jit._shape_functions",
            Path(""),
            False,
        ),
        # ImportOverload(
        #     "torch._functorch",
        #     shark_package_root_path / "_torch/_functorch/__init__.py",
        #     True,
        # ),
    ]
    overloads = {o.name: o for o in overloads}
    overloads.update(BASE_OVERLOADS)
    with patch_meta_path(overloads):
        run_sharkpy_tests(torch_mlir_linalg_module_strs)


if __name__ == "__main__":
    main()
