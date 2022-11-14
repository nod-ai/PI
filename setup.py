#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import platform
import sys
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from distutils.command.build import build as _build
from pathlib import Path

from setuptools import find_namespace_packages, setup, Extension, Distribution
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


# Build phase discovery is unreliable. Just tell it what phases to run.
class CustomBuild(_build):
    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def get_llvm_package():
    # download if nothing is installed
    system = platform.system()
    system_suffix = {"Linux": "linux-gnu-ubuntu-20.04", "Darwin": "apple-darwin"}[
        system
    ]
    LIB_ARCH = os.environ.get("LIB_ARCH", platform.machine())
    assert LIB_ARCH is not None
    print(f"ARCH {LIB_ARCH}")
    name = f"llvm+mlir+python-{sys.version_info.major}.{sys.version_info.minor}-15.0.4-{LIB_ARCH}-{system_suffix}-release"
    here = Path(__file__).parent
    if not (here / "llvm_install").exists():
        url = f"https://github.com/makslevental/llvm-releases/releases/latest/download/{name}.tar.xz"
        print(f"downloading and extracting {url} ...")
        ftpstream = urllib.request.urlopen(url)
        file = tarfile.open(fileobj=ftpstream, mode="r|*")
        file.extractall(path=str(here))

    TBLGEN_ARCH = os.environ.get("TBLGEN_ARCH", platform.machine())
    tblgen_name = f"mlir-tblgen-15.0.4-{TBLGEN_ARCH}-{system_suffix}-release.exe"
    if not (here / "llvm_install" / "bin" / tblgen_name).exists():
        url = f"https://github.com/makslevental/llvm-releases/releases/latest/download/{tblgen_name}"
        print(f"downloading and extracting {url} ...")
        ftpstream = urllib.request.urlopen(url)
        os.remove(here / "llvm_install" / "bin" / "mlir-tblgen")
        with open(here / "llvm_install" / "bin" / "mlir-tblgen", "wb") as f:
            f.write(ftpstream.read())

        make_executable(str(here / "llvm_install" / "bin" / "mlir-tblgen"))

    print("done downloading")
    return str((here / "llvm_install").absolute())


class CMakeBuild(build_py):
    def run(self):
        target_dir = self.build_lib
        here = os.path.abspath(os.path.dirname(__file__))
        cmake_build_dir = os.path.join(here, "build")
        cmake_install_dir = os.path.join(cmake_build_dir, "install")
        llvm_install_dir = get_llvm_package()
        cmake_args = os.environ.get("CMAKE_ARGS", "")
        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={cmake_install_dir}",
            f"-DCMAKE_PREFIX_PATH={llvm_install_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DMLIR_ENABLE_EXECUTION_ENGINE=ON",
            f"-DLLVM_TABLEGEN_EXE={llvm_install_dir}/bin/llvm-tblgen",
            f"-DMLIR_TABLEGEN_EXE={llvm_install_dir}/bin/mlir-tblgen",
            f"-DMLIR_LINALG_ODS_YAML_GEN_EXE={llvm_install_dir}/bin/mlir-linalg-ods-yaml-gen",
        ] + cmake_args.split(";")

        build_args = []
        os.makedirs(cmake_build_dir, exist_ok=True)
        if os.path.exists(cmake_install_dir):
            shutil.rmtree(cmake_install_dir)
        cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
        if os.path.exists(cmake_cache_file):
            os.remove(cmake_cache_file)
        subprocess.check_call(
            ["cmake", "-G Ninja", here] + cmake_args, cwd=cmake_build_dir
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"] + build_args,
            cwd=cmake_build_dir,
        )
        shutil.copytree(
            os.path.join(cmake_install_dir, "python_packages", "mlir_core"),
            target_dir,
            symlinks=False,
            dirs_exist_ok=True,
        )


class NoopBuildExtension(build_ext):
    def build_extension(self, ext):
        pass


packages = find_namespace_packages(
    include=[
        "shark",
        "shark.*",
    ],
)


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(foo):
        return True


setup(
    name="SharkPy",
    version="0.0.1",
    author="Maksim Levental",
    author_email="maksim.levental@gmail.com",
    description="Python bindings for the Shark MLIR dialect",
    include_package_data=True,
    ext_modules=[
        CMakeExtension("shark._mlir_libs._mlir"),
    ],
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuild,
    },
    zip_safe=False,
    packages=packages,
    distclass=BinaryDistribution,
)
