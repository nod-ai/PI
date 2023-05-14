import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pip._internal.req import parse_requirements
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext


def get_torch_mlir_url():
    system = {"Linux": "ubuntu", "Darwin": "macos"}[platform.system()]
    url = "https://github.com/nod-ai/PI/releases/expanded_assets/torch-mlir-latest"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    download_link = soup.find("a", href=re.compile(rf".*{system}.*"))
    assert download_link, "couldn't find correct torch-mlir distro download link"
    return f"https://github.com/{download_link['href']}"


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
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_build_lib_dir}{os.sep}pi",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        llvm_install_dir = os.environ.get("LLVM_INSTALL_DIR", None)
        if llvm_install_dir is not None:
            cmake_args.append(f"-DLLVM_INSTALL_DIR={llvm_install_dir}")
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

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

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
            ["cmake", "--build", "."] + build_args, cwd=build_temp, check=True
        )


packages = find_namespace_packages(
    include=[
        "pi",
        "pi.*",
    ],
)

VERSION = "0.0.3"

if len(sys.argv) > 1 and sys.argv[1] == "--version":
    print(VERSION)
if len(sys.argv) > 1 and sys.argv[1] == "--torch-mlir-url":
    print(get_torch_mlir_url())
else:
    install_reqs = parse_requirements("requirements.txt", session="hack")
    setup(
        name="PI",
        version=VERSION,
        author="Maksim Levental",
        author_email="maksim.levental@gmail.com",
        description="A lightweight MLIR Python frontend with PyTorch like syntax",
        ext_modules=[CMakeExtension("_mlir")],
        cmdclass={"build_ext": CMakeBuild},
        packages=packages,
        zip_safe=False,
        install_requires=[str(ir.requirement) for ir in install_reqs],
    )
