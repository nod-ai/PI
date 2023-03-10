import difflib
import functools
import inspect
import logging

FORMAT = "[%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(FORMAT)
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

import operator
from multiprocessing import cpu_count
from multiprocess.pool import Pool

# noinspection PyUnresolvedReferences
import pi

# noinspection PyUnresolvedReferences
import torch_mlir
import torch_mlir_e2e_test
import torch_mlir_e2e_test.registry

from xfail import PI_XFAIL_SET, CASTS

from torch_mlir_e2e_test.test_suite import COMMON_TORCH_MLIR_LOWERING_XFAILS
from torch_mlir_e2e_test.test_suite import (
    register_all_tests as torch_mlir_register_all_tests,
)

from pi.dialects import (
    remove_modules,
    RewriteOverload,
    patch_meta_path,
)
from pi.mlir.utils import lower_pi_to_linalg
from pi.testing.util import (
    PIConfig,
    TorchDialectConfig,
    set_weights,
    lower_torch_mlir_to_linalg,
)
from pi.testing.registry import GLOBAL_TEST_REGISTRY as PI_GLOBAL_TEST_REGISTRY
from pi.dispatcher.function import NotFoundLookupError, AmbiguousLookupError


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
    num_processes = min(int(cpu_count() * 1.1), len(tests))

    torch_dialect_config = TorchDialectConfig()
    pool = Pool(num_processes)

    def compile_and_run_test(test):
        mod = test.program_factory()
        set_weights(mod)
        mod.eval()
        torch_mlir_module_raw = str(
            torch_dialect_config.compile(mod, output_type=torch_mlir.OutputType.RAW)
        )
        # logger.info(torch_mlir_module_raw)
        torch_mlir_module = torch_dialect_config.compile(mod)
        torch_mlir_module_str = str(torch_mlir_module)
        return (
            test.unique_name,
            str(lower_torch_mlir_to_linalg(torch_mlir_module)),
            torch_mlir_module_str,
            torch_mlir_module_raw,
        )

    torch_mlir_module_strs = {}
    # handles = map(compile_and_run_test, tests)
    # for name, linalg_module, torch_module in handles:
    handles = pool.map_async(compile_and_run_test, tests)
    for (
        name,
        linalg_module,
        torch_mlir_module_str,
        torch_mlir_module_raw,
    ) in handles.get():
        torch_mlir_module_strs[name] = (
            linalg_module,
            torch_mlir_module_str,
            torch_mlir_module_raw,
        )

    return torch_mlir_module_strs


ONLY = {
    # "BoolIntConstantModule_basic"
    # "BatchNorm3DModule_basic"
    # "AtenIntBoolOpConstFalseModule_basic"
}
# ONLY = CASTS


def run_pi_tests(torch_mlir_module_strs):
    torch_mlir_register_all_tests()
    for cast in CASTS:
        assert cast in PI_GLOBAL_TEST_REGISTRY, f"missing expected test case {cast}"
        PI_GLOBAL_TEST_REGISTRY.pop(cast)
    # noinspection PyUnresolvedReferences
    import casts

    for cast in CASTS:
        assert cast in PI_GLOBAL_TEST_REGISTRY, f"missing expected test case {cast}"

    tests = sorted(PI_GLOBAL_TEST_REGISTRY.values(), key=lambda t: t.unique_name)
    if ONLY:
        tests = list(filter(lambda t: t.unique_name in ONLY, tests))
    assert tests, "failed to load tests"

    pi.nn.Module.train = lambda *args, **kwargs: None

    pi_config = PIConfig()
    (
        PASS,
        NotImplementedErrorFAIL,
        NotFoundLookupErrorFAIL,
        AmbiguousLookupErrorFAIL,
        lower_to_linalg_FAIL,
        irFAIL,
        TOTAL,
        SKIP,
    ) = (0, 0, 0, 0, 0, 0, 0, 0)
    for test in tests:
        TOTAL += 1
        if test.unique_name in PI_XFAIL_SET | COMMON_TORCH_MLIR_LOWERING_XFAILS:
            logger.info(f"skipping {test.unique_name}")
            SKIP += 1
            continue
        print(f"running {test.unique_name}")

        (
            torch_linalg_module,
            torch_dialect_module,
            torch_dialect_module_raw,
        ) = torch_mlir_module_strs[test.unique_name]

        try:
            pi_mlir_module = pi_config.compile(test)
        except NotImplementedError as e:
            # logger.info(traceback.format_exc(-2))
            # logger.info(f"{e}")
            logger.info(f"FAIL pi compile NotImplementedError")
            NotImplementedErrorFAIL += 1
            continue
        except NotFoundLookupError as e:
            # logger.info(traceback.format_exc())
            # logger.info(f"{e}")
            logger.info(f"FAIL dispatcher error")
            NotFoundLookupErrorFAIL += 1

            continue
        except AmbiguousLookupError as e:
            # logger.info(traceback.format_exc())
            # logger.info(f"{e}")
            logger.info(f"FAIL dispatcher error")
            AmbiguousLookupErrorFAIL += 1
            # logger.info("\ntorch_mlir module\n")
            # logger.info(torch_dialect_module)
            # logger.info(torch_linalg_module)

            continue
        except Exception as e:
            # logger.info("\ntorch_mlir module\n")
            # logger.info(torch_dialect_module)
            # logger.info(torch_linalg_module)
            # torch_script_compiled = torchscript_config.compile(mod)
            # frozen = torch.jit.freeze(torch_script_compiled)
            # torch_mlir_module = torch_dialect_config.compile(mod)
            # logger.info("frozen.graph\n", frozen.graph)
            # logger.info("torch_mlir_module\n", torch_mlir_module)
            logger.info(f"FAIL pi compile Exception")
            raise e

        pi_torch_dialect_module_str = str(pi_mlir_module)
        try:
            pi_linalg_module_str = str(
                lower_pi_to_linalg(pi_mlir_module, enable_ir_printing=False)
            )
        except Exception as e:
            # logger.info(traceback.format_exc())
            # logger.info("\npi_torch module\n")
            # logger.info(pi_torch_dialect_module_str)
            # logger.info("\npi linalg module\n")
            # logger.info(pi_mlir_module)
            # logger.info("\ntorch_mlir raw module\n")
            # logger.info(torch_dialect_module_raw)
            # logger.info("\ntorch_mlir module\n")
            # logger.info(torch_dialect_module)
            # logger.info("\nlinalg module\n")
            # logger.info(torch_linalg_module)
            logger.info(f"FAIL lower_pi_to_linalg Exception")
            lower_to_linalg_FAIL += 1

            continue

        diff = list(
            difflib.unified_diff(
                str(pi_linalg_module_str).splitlines(),
                str(torch_linalg_module).splitlines(),
                lineterm="",
            )
        )
        for i, d in enumerate(diff):
            diff[i] = d[:100]

        if len(diff):
            logger.info(f"\n{''.join('*' * 10)}\ndiff\n{''.join('*' * 10)}\n")
            logger.info("\n".join(diff))

            # logger.info("torch_mlir_linalg_module_str:\n", torch_mlir_linalg_module_str)
            # logger.info("pi_mlir_linalg_module_str:\n", pi_mlir_linalg_module_str)
            logger.info(f"FAIL IR diff")
            irFAIL += 1

            # torch_script_compiled = torchscript_config.compile(mod)
            # frozen = torch.jit.freeze(torch_script_compiled)
            # torch_mlir_module = torch_dialect_config.compile(mod)
            # logger.info("frozen.graph\n", frozen.graph)
            # logger.info("torch_mlir_module\n", torch_mlir_module)
        else:
            logger.info("PASS")
            PASS += 1

    assert (
        functools.reduce(
            operator.add,
            (
                PASS,
                NotImplementedErrorFAIL,
                NotFoundLookupErrorFAIL,
                AmbiguousLookupErrorFAIL,
                lower_to_linalg_FAIL,
                irFAIL,
                SKIP,
            ),
            0,
        )
        == TOTAL
    )
    print(
        f"\n{''.join('*' * 10)}\n\n{PASS=}\n{NotImplementedErrorFAIL=}\n{NotFoundLookupErrorFAIL=}\n{AmbiguousLookupErrorFAIL=}\n{lower_to_linalg_FAIL=}\n{irFAIL=}\n{SKIP=}\nout of {TOTAL=}\n\n{''.join('*' * 10)}\n"
    )
    assert (
        NotImplementedErrorFAIL == 0
    ), f"missing torch_wrappers impl; you probably need to run generate_torch_mlir_extensions.py: {NotImplementedErrorFAIL=}"
    assert (
        NotFoundLookupErrorFAIL == 0
    ), f"pytorch api changed; good luck: {NotFoundLookupErrorFAIL=}"


class TestMain:
    def test_suite(self):
        torch_mlir_linalg_module_strs = run_torch_mlir_tests()
        overloads = [
            RewriteOverload(
                f"torch_mlir_e2e_test.test_suite.{k}",
                {
                    "torch.": "pi.",
                    "from torch import nn": "from pi import nn",
                    "torch_mlir_e2e_test.annotations": "pi.utils.annotations",
                    "torch_mlir_e2e_test": "pi.testing",
                    "import torchvision.models as models": "",
                    "import torch": "import pi",
                    "import functorch": "",
                },
            )
            for k, v in torch_mlir_e2e_test.test_suite.__dict__.items()
            if inspect.ismodule(v)
        ]
        torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY = []
        torch_mlir_e2e_test.registry._SEEN_UNIQUE_NAMES = set()
        remove_modules(lambda mod: mod.startswith("torch_mlir_e2e_test"))
        remove_modules(lambda mod: mod == "torch" or mod.startswith("torch."))

        overloads = {o.name: o for o in overloads}
        with patch_meta_path(overloads):
            run_pi_tests(torch_mlir_linalg_module_strs)


if __name__ == "__main__":
    TestMain().test_suite()
