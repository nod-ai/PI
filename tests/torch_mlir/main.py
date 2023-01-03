import difflib
import functools
import logging
import operator
import traceback
from itertools import repeat
from pathlib import Path
import multiprocess as mp
from multiprocess.pool import Pool

FORMAT = "%(asctime)s, %(levelname)-8s [%(filename)s:%(module)s:%(funcName)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(FORMAT)


# logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

# Create handlers
# c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file.log")
# c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# noinspection PyUnresolvedReferences
import pi

# noinspection PyUnresolvedReferences
import torch_mlir
import torch_mlir_e2e_test

from xfail import PI_XFAIL_SET

pi_package_root_path = Path(pi.__file__).parent
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
from pi.mlir_utils import lower_pi_to_linalg
from pi.testing.util import (
    PIConfig,
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

    tests = sorted(
        torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY, key=lambda t: t.unique_name
    )
    tests = list(
        filter(
            lambda t: t.unique_name
            not in PI_XFAIL_SET | COMMON_TORCH_MLIR_LOWERING_XFAILS,
            tests,
        )
    )
    num_processes = min(int(mp.cpu_count() * 1.1), len(tests))

    torch_dialect_config = TorchDialectConfig()
    pool = Pool(num_processes)

    def compile_and_run_test(test):
        mod = test.program_factory()
        set_weights(mod)
        mod.eval()
        torch_mlir_module = torch_dialect_config.compile(mod)
        return test.unique_name, str(lower_torch_mlir_to_linalg(torch_mlir_module))

    handles = pool.map_async(compile_and_run_test, tests)
    # handles = map(compile_and_run_test, tests)
    torch_mlir_linalg_module_strs = {}
    for name, s in handles.get():
        # for name, s in handles:
        torch_mlir_linalg_module_strs[name] = s

    return torch_mlir_linalg_module_strs


def run_pi_tests(torch_mlir_linalg_module_strs):
    torch_mlir_register_all_tests()
    # after remapping, this imports pi test registry
    import torch_mlir_e2e_test.registry

    tests = sorted(
        torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY, key=lambda t: t.unique_name
    )
    assert tests, "failed to load tests"

    from torch import nn
    from pi.dispatcher.function import NotFoundLookupError, AmbiguousLookupError
    from torch.dispatcher.function import (
        NotFoundLookupError as torch_NotFoundLookupError,
        AmbiguousLookupError as torch_AmbiguousLookupError,
    )

    assert (
        nn.__spec__.origin == f"{pi_package_root_path}/nn/__init__.py"
    ), f"monkey patch failed {nn.__spec__.origin}"
    # for compatibility
    nn.Module.train = lambda *args, **kwargs: None

    # torchscript_config = TorchScriptTestConfig()
    pi_config = PIConfig()
    torch_dialect_config = TorchDialectConfig()
    (
        PASS,
        NotImplementedErrorFAIL,
        NotFoundLookupErrorFAIL,
        AmbiguousLookupErrorFAIL,
        compileFAIL,
        lower_to_linalg_FAIL,
        irFAIL,
        TOTAL,
        SKIP,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    for test in tests:
        TOTAL += 1
        if test.unique_name in PI_XFAIL_SET | COMMON_TORCH_MLIR_LOWERING_XFAILS:
            print(f"skipping {test.unique_name}")
            SKIP += 1
            continue
        print(f"running {test.unique_name}")

        test_module = test.program_factory()
        torch_mlir_linalg_module_str = torch_mlir_linalg_module_strs[test.unique_name]

        try:
            pi_mlir_module = pi_config.compile(test, test_module)
        except NotImplementedError as e:
            print(traceback.format_exc(-2))
            print(f"{e}")
            print(f"FAIL pi compile NotImplementedError")
            NotImplementedErrorFAIL += 1
            print()
            continue
        except (NotFoundLookupError, torch_NotFoundLookupError) as e:
            print(traceback.format_exc())
            print(f"{e}")
            print(f"FAIL dispatcher error")
            NotFoundLookupErrorFAIL += 1
            print()
            continue
        except (AmbiguousLookupError, torch_AmbiguousLookupError) as e:
            print(traceback.format_exc())
            print(f"{e}")
            print(f"FAIL dispatcher error")
            AmbiguousLookupErrorFAIL += 1
            print()
            continue
        except Exception as e:
            print("\ntorch_mlir module\n")
            print(torch_mlir_linalg_module_str)
            # torch_script_compiled = torchscript_config.compile(mod)
            # frozen = torch.jit.freeze(torch_script_compiled)
            # torch_mlir_module = torch_dialect_config.compile(mod)
            # print("frozen.graph\n", frozen.graph)
            # print("torch_mlir_module\n", torch_mlir_module)
            print(f"FAIL pi compile Exception")
            compileFAIL += 1
            raise e

        try:
            pi_mlir_linalg_module_str = str(lower_pi_to_linalg(pi_mlir_module))
        except Exception as e:
            print(traceback.format_exc())
            print("\ntorch_mlir module\n")
            print(torch_mlir_linalg_module_str)
            print(f"FAIL lower_pi_to_linalg Exception")
            lower_to_linalg_FAIL += 1
            print()
            continue

        diff = list(
            difflib.unified_diff(
                str(pi_mlir_linalg_module_str).splitlines(),
                str(torch_mlir_linalg_module_str).splitlines(),
                lineterm="",
            )
        )

        if len(diff):
            print(f"\n{''.join('*' * 10)}\ndiff\n{''.join('*' * 10)}\n")
            print("\n".join(diff))
            print()
            # print("torch_mlir_linalg_module_str:\n", torch_mlir_linalg_module_str)
            # print("pi_mlir_linalg_module_str:\n", pi_mlir_linalg_module_str)
            print(f"FAIL IR diff")
            irFAIL += 1

            # torch_script_compiled = torchscript_config.compile(mod)
            # frozen = torch.jit.freeze(torch_script_compiled)
            # torch_mlir_module = torch_dialect_config.compile(mod)
            # print("frozen.graph\n", frozen.graph)
            # print("torch_mlir_module\n", torch_mlir_module)
        else:
            print("PASS")
            PASS += 1
        print()

    assert (
        functools.reduce(
            operator.add,
            (
                PASS,
                NotImplementedErrorFAIL,
                NotFoundLookupErrorFAIL,
                AmbiguousLookupErrorFAIL,
                compileFAIL,
                lower_to_linalg_FAIL,
                irFAIL,
                SKIP,
            ),
            0,
        )
        == TOTAL
    )
    print(
        f"\n{''.join('*' * 10)}\n\n{PASS=}\n{NotImplementedErrorFAIL=}\n{NotFoundLookupErrorFAIL=}\n{AmbiguousLookupErrorFAIL=}\n{compileFAIL=}\n{lower_to_linalg_FAIL=}\n{irFAIL=}\n{SKIP=}\nout of {TOTAL=}\n\n{''.join('*' * 10)}\n"
    )


def main():
    torch_mlir_linalg_module_strs = run_torch_mlir_tests()
    remove_modules(lambda mod: mod.startswith("torch_mlir_e2e_test"))
    remove_modules(lambda mod: mod == "torch" or mod.startswith("torch."))

    # remap to torch so that isintance works...
    # remove_modules(lambda mod: mod == "pi" or mod.startswith("pi."))

    overloads = [
        ImportOverload(
            "torch_mlir_e2e_test.framework",
            pi_package_root_path / "testing/framework.py",
            False,
        ),
        ImportOverload(
            "torch_mlir_e2e_test.registry",
            pi_package_root_path / "testing/registry.py",
            False,
        ),
        ImportOverload(
            "torch_mlir_e2e_test.annotations",
            pi_package_root_path / "compiler/annotations.py",
            False,
        ),
        ImportOverload(
            "torch",
            pi_package_root_path / "__init__.py",
            True,
        ),
        ImportOverload(
            "torch.jit._shape_functions",
            Path(""),
            False,
        ),
        # ImportOverload(
        #     "torch._functorch",
        #     pi_package_root_path / "_torch/_functorch/__init__.py",
        #     True,
        # ),
    ]
    overloads = {o.name: o for o in overloads}
    overloads.update(BASE_OVERLOADS)
    with patch_meta_path(overloads):
        run_pi_tests(torch_mlir_linalg_module_strs)


if __name__ == "__main__":
    main()
