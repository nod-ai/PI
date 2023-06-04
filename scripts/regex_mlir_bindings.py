import argparse
import os.path
import re
from difflib import unified_diff


def patch_bindings_header(fp, debug=False):
    with open(fp) as f:
        f = f.read()

    if not os.path.exists(f"{fp}.orig"):
        with open(f"{fp}.bkup", "w") as bkup:
            bkup.write(f)

    with open(f"{fp}.bkup", "w") as bkup:
        bkup.write(f)

    with open(fp, "w") as ff:
        res = re.sub(
            r"^class ((?!PYBIND11_EXPORT).*) {$",
            r"class PYBIND11_EXPORT \1 {",
            f,
            flags=re.MULTILINE,
        )
        res = re.sub(
            r",.?py::module_local\(\)",
            "",
            res,
            flags=re.MULTILINE,
        )
        res = res.replace(
            "class DefaultingPyMlirContext",
            "class PYBIND11_EXPORT DefaultingPyMlirContext",
        )
        diff = list(
            unified_diff(
                f.splitlines(keepends=True),
                res.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            )
        )
        ff.write(res)

    if debug:
        print("".join(diff))


def patch_bindings_cpp(fp, debug=False):
    with open(fp) as f:
        f = f.read()
    if not os.path.exists(f"{fp}.orig"):
        with open(f"{fp}.bkup", "w") as bkup:
            bkup.write(f)
    with open(f"{fp}.bkup", "w") as bkup:
        bkup.write(f)
    with open(fp, "w") as ff:
        res = re.sub(
            r",.?py::module_local\(\)",
            "",
            f,
            flags=re.MULTILINE,
        )
        diff = list(
            unified_diff(
                f.splitlines(keepends=True),
                res.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            )
        )
        ff.write(res)
    if debug:
        print("".join(diff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    fp = args.fp
    debug = args.debug
    if fp.endswith(".h"):
        patch_bindings_header(fp, debug)
    elif fp.endswith(".cpp"):
        patch_bindings_cpp(fp, debug)
    else:
        raise NotImplementedError(f"unknown fp {fp=}")

# TODO(max): some day, figure out how to actually compile C extensions the correct way so that these regex hacks aren't necessary
# somewhere around torch_mlir_install/torch_mlir_install/lib/cmake/mlir/AddMLIRPython.cmake:605 (e.g., by adding SHARED)
