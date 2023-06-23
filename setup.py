import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pip._internal.req import parse_requirements
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        ext_build_lib_dir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cmake_args = [
            # f"-DCMAKE_C_COMPILER:FILEPATH=clang",
            # f"-DCMAKE_CXX_COMPILER:FILEPATH=clang++",
            f"-DCMAKE_INSTALL_PREFIX={ext_build_lib_dir}/{PACKAGE_NAME}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        torch_mlir_install_dir = os.environ.get("TORCH_MLIR_INSTALL_DIR", None)
        if torch_mlir_install_dir is not None:
            cmake_args.append(f"-DTORCH_MLIR_INSTALL_DIR={torch_mlir_install_dir}")
        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja  # noqa: F401

                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except ImportError:
                pass

        if sys.platform.lower().startswith("darwin") and os.environ.get(
            "CMAKE_OSX_ARCHITECTURES", False
        ):
            cmake_args += [
                f"-DCMAKE_OSX_ARCHITECTURES={os.getenv('CMAKE_OSX_ARCHITECTURES')}"
            ]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "install"] + build_args,
            cwd=build_temp,
            check=True,
        )

        if platform.system() == "Darwin":
            shlib_ext = "dylib"
        elif platform.system() == "Linux":
            shlib_ext = "so"
        else:
            raise NotImplementedError(f"unknown platform {platform.system()}")

        mlir_libs_dir = Path(f"{ext_build_lib_dir}/{PACKAGE_NAME}/mlir/_mlir_libs")
        shlibs = [
            # "LTO",
            # "MLIR-C",
            # "Remarks",
            "mlir_async_runtime",
            "mlir_c_runner_utils",
            "mlir_float16_utils",
            "mlir_runner_utils",
        ]

        for shlib in shlibs:
            shlib_name = f"lib{shlib}.{shlib_ext}"
            torch_mlir_install_dir = (
                Path(".").parent / "torch_mlir_install/torch_mlir_install"
            ).absolute()
            assert torch_mlir_install_dir.exists(), f"missing {torch_mlir_install_dir=}"
            torch_mlir_install_fp = (
                torch_mlir_install_dir / "lib" / shlib_name
            ).absolute()
            assert torch_mlir_install_fp.exists(), f"missing {torch_mlir_install_fp=}"
            dst_path = mlir_libs_dir / shlib_name
            shutil.copyfile(torch_mlir_install_fp, dst_path)
            if platform.system() == "Linux":
                shutil.copyfile(torch_mlir_install_fp, f"{dst_path}.17git")
                subprocess.run(
                    ["patchelf", "--set-rpath", "$ORIGIN", dst_path],
                    cwd=build_temp,
                    check=True,
                )
                subprocess.run(
                    ["patchelf", "--set-rpath", "$ORIGIN", f"{dst_path}.17git"],
                    cwd=build_temp,
                    check=True,
                )


PACKAGE_NAME = "pi"

packages = find_namespace_packages(
    include=[
        PACKAGE_NAME,
        f"{PACKAGE_NAME}.*",
    ],
)
VERSION = "0.0.4"

if len(sys.argv) > 1 and sys.argv[1] == "--version":
    print(VERSION)
else:
    install_reqs = parse_requirements("requirements.txt", session="hack")
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author="Maksim Levental",
        author_email="maksim.levental@gmail.com",
        description="A lightweight MLIR Python frontend with PyTorch like syntax",
        ext_modules=[CMakeExtension("_pi_mlir")],
        cmdclass={"build_ext": CMakeBuild},
        packages=packages,
        zip_safe=False,
        install_requires=[str(ir.requirement) for ir in install_reqs],
    )
