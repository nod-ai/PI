import difflib
import inspect
import logging
import sys
from collections import defaultdict
from pathlib import Path

FORMAT = "[%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(FORMAT)
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

from multiprocessing import cpu_count, Manager
from multiprocess.pool import Pool

# noinspection PyUnresolvedReferences
import torch_mlir

from xfail import CRASHING, PI_XFAIL_SET

from torch_mlir_e2e_test.test_suite import COMMON_TORCH_MLIR_LOWERING_XFAILS
from torch_mlir_e2e_test.test_suite import (
    register_all_tests as torch_mlir_register_all_tests,
)

from infra.path_hacks import (
    remove_modules,
    RewriteOverload,
    patch_meta_path,
)
from pi.mlir.compile import lower_pi_to_linalg
import pi

ONLY = {
    # "ChunkListUnpackUneven_Module_basic"
}


def run_torch_mlir_tests(sequential=False):
    torch_mlir_register_all_tests()
    import torch_mlir_e2e_test.registry
    import torch_mlir_e2e_test.framework
    from infra.util import (
        TorchDialectConfig,
        set_weights,
    )

    tests = sorted(
        torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY, key=lambda t: t.unique_name
    )
    tests = list(
        filter(
            lambda t: t.unique_name not in CRASHING | COMMON_TORCH_MLIR_LOWERING_XFAILS,
            tests,
        )
    )

    if ONLY:
        tests = list(
            filter(
                lambda t: t.unique_name in ONLY,
                tests,
            )
        )

    num_processes = min(int(cpu_count() * 1.1), len(tests))
    torch_dialect_config = TorchDialectConfig()
    pool = Pool(num_processes)

    def compile_and_run_test(test):
        from torch_mlir import run_pipeline_with_repro_report

        # logger.info(f"running TorchMLIR {test.unique_name}")

        mod = test.program_factory()
        set_weights(mod)
        mod.eval()
        torch_mlir_module_raw = str(
            torch_dialect_config.compile(mod, output_type=torch_mlir.OutputType.RAW)
        )
        # logger.info(torch_mlir_module_raw)
        torch_mlir_module = torch_dialect_config.compile(mod)
        torch_mlir_module_str = str(
            torch_mlir_module.operation.get_asm(large_elements_limit=10)
        )
        run_pipeline_with_repro_report(
            torch_mlir_module,
            "builtin.module(cse,torch-backend-to-linalg-on-tensors-backend-pipeline)",
            "Lowering Torch Backend IR to Linalg",
        )
        linalg_module = str(
            torch_mlir_module.operation.get_asm(large_elements_limit=10)
        )
        return (
            test.unique_name,
            linalg_module,
            torch_mlir_module_str,
            torch_mlir_module_raw,
        )

    torch_mlir_module_strs = {}
    if sequential:
        handles = map(compile_and_run_test, tests)
    else:

        def error_callback(*args, **kwargs):
            print("uncaught error inside torch-mlir test", args, kwargs)

        handles = pool.map_async(
            compile_and_run_test, tests, error_callback=error_callback
        ).get()
    for (
        name,
        linalg_module,
        torch_mlir_module_str,
        torch_mlir_module_raw,
    ) in handles:
        torch_mlir_module_strs[name] = (
            linalg_module,
            torch_mlir_module_str,
            torch_mlir_module_raw,
        )

    return torch_mlir_module_strs


def run_pi_tests(torch_mlir_module_strs, sequential=False):
    from infra.util import PIConfig, GLOBAL_TEST_REGISTRY

    tests = sorted(GLOBAL_TEST_REGISTRY.values(), key=lambda t: t.unique_name)
    assert tests, "failed to load tests"

    pi.nn.Module.train = lambda *args, **kwargs: None
    pi_config = PIConfig()
    num_processes = min(int(cpu_count() * 1.1), len(tests))
    pool = Pool(num_processes)
    manager = Manager()
    XFAILs, Exception_FAILs, lower_to_linalg_FAILs, ir_FAILs, SKIPs = (
        [],
        [],
        [],
        [],
        [],
    )
    if not sequential:
        XFAILs, Exception_FAILs, lower_to_linalg_FAILs, ir_FAILs, SKIPs = map(
            manager.list,
            [XFAILs, Exception_FAILs, lower_to_linalg_FAILs, ir_FAILs, SKIPs],
        )

    def compile_and_run_test(test):
        if test.unique_name in CRASHING | COMMON_TORCH_MLIR_LOWERING_XFAILS or (
            ONLY and test.unique_name not in ONLY
        ):
            SKIPs.append(test.unique_name)
            return
        logger.info(f"running {test.unique_name}")

        (
            torch_linalg_module,
            torch_dialect_module,
            torch_dialect_module_raw,
        ) = torch_mlir_module_strs[test.unique_name]

        try:
            pi_mlir_module = pi_config.compile(test)
        except Exception as e:
            Exception_FAILs.append((test.unique_name, str(e)))
            return

        pi_torch_dialect_module_str = str(
            pi_mlir_module.operation.get_asm(large_elements_limit=10)
        )
        try:
            pi_linalg_module_str = str(
                lower_pi_to_linalg(
                    pi_mlir_module, enable_ir_printing=False
                ).operation.get_asm(large_elements_limit=10)
            )
        except Exception as e:
            lower_to_linalg_FAILs.append(
                (test.unique_name, str(e), pi_torch_dialect_module_str)
            )
            return

        sorted_diff = list(
            difflib.unified_diff(
                sorted(str(pi_linalg_module_str).splitlines()),
                sorted(str(torch_linalg_module).splitlines()),
                lineterm="",
            )
        )

        if len(sorted_diff) and test.unique_name in PI_XFAIL_SET:
            XFAILs.append(test.unique_name)
        elif len(sorted_diff):
            diff = list(
                difflib.unified_diff(
                    str(pi_linalg_module_str).splitlines(),
                    str(torch_linalg_module).splitlines(),
                    lineterm="",
                )
            )
            ir_FAILs.append((test.unique_name, diff))
        else:
            logger.info("PASS")
            return 1

    if sequential:
        handles = map(compile_and_run_test, tests)
    else:
        handles = pool.map_async(compile_and_run_test, tests).get()

    PASS = 0
    for r in handles:
        r = 0 if r is None else 1
        PASS += r

    print("\n", "".join("*" * 80), "\n", "SKIPs", "\n", "".join("*" * 80))
    for test_name in SKIPs:
        print(test_name)

    print("\n", "".join("*" * 80), "\n", "XFAILs", "\n", "".join("*" * 80))
    for test_name in XFAILs:
        print(test_name)

    print(
        "\n", "".join("*" * 80), "\n", "lower_to_linalg_FAILs", "\n", "".join("*" * 80)
    )
    for test_name, e, pi_torch_dialect_module_str in lower_to_linalg_FAILs:
        print(test_name, "\n", e, "\n", pi_torch_dialect_module_str, "\n")

    print("\n", "".join("*" * 80), "\n", "ir_FAILs", "\n", "".join("*" * 80))
    for test_name, diff in ir_FAILs:
        print(test_name, "\n", "\n".join(diff), "\n")

    print("\n", "".join("*" * 80), "\n", "Exception_FAILs", "\n", "".join("*" * 80))
    fails = defaultdict(list)
    for test_name, e in Exception_FAILs:
        fails[e].append(test_name)
    for e, test_names in sorted(fails.items(), key=lambda e_ts: len(e_ts[1])):
        print(len(test_names), ":", ", ".join(test_names))
        print(e, "\n")

    print(f"total # of tests: {len(tests)}")
    print(f"total # of skips: {len(SKIPs)}")
    print(f"total # of passes: {PASS}")
    print(f"total # of xfails: {len(XFAILs)}")
    print(f"total # of exceptions: {len(Exception_FAILs)}")
    print(f"total # of lower to linalg failures: {len(lower_to_linalg_FAILs)}")
    print(f"total # of ir differences: {len(ir_FAILs)}")


class TestMain:
    def test_suite(self, sequential=False):
        torch_mlir_linalg_module_strs = run_torch_mlir_tests(sequential=sequential)

        import torch_mlir_e2e_test

        overloads = [
            # stupid thing in torch_mlir_e2e_test.test_suite.__init__.py that imports torch
            RewriteOverload(
                f"torch_mlir_e2e_test.test_suite",
                {
                    "from torch_mlir._version": "from pi._version",
                },
            )
        ] + [
            RewriteOverload(
                f"torch_mlir_e2e_test.test_suite.{k}",
                {
                    "torch.": "pi.",
                    "from torch import nn": "from pi import nn",
                    "torch_mlir_e2e_test.annotations": "pi.mlir.utils",
                    "torch_mlir_e2e_test.registry": "infra.util",
                    "torch_mlir_e2e_test.framework": "infra.util",
                    "import torchvision.models as models": "",
                    "import torch": "import pi",
                    "import functorch": "",
                },
            )
            for k, v in torch_mlir_e2e_test.test_suite.__dict__.items()
            if inspect.ismodule(v)
        ]
        remove_modules(lambda mod: mod.startswith("torch_mlir_e2e_test"))
        remove_modules(lambda mod: mod == "torch" or mod.startswith("torch."))

        # reload torch_mlir tests but patch all paths to point here
        overloads = {o.name: o for o in overloads}
        with patch_meta_path(overloads):
            torch_mlir_register_all_tests()

        run_pi_tests(torch_mlir_linalg_module_strs, sequential=False)


if __name__ == "__main__":
    # just to be sure
    sys.path.insert(0, str(Path(__file__).parent))
    TestMain().test_suite()
