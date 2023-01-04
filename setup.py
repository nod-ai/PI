import os
import platform
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from pip._internal.req import parse_requirements
from setuptools import Extension, setup, find_namespace_packages
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


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

    print("done downloading")
    return str((here / "llvm_install").absolute())


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

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        llvm_install_dir = os.environ.get("LLVM_INSTALL_DIR", None)
        if llvm_install_dir is None:
            llvm_install_dir = get_llvm_package()
        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            # f"-DCMAKE_BUILD_TYPE=Debug",
            # f"-DCMAKE_C_COMPILER=clang",
            # f"-DCMAKE_CXX_COMPILER=clang++",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_build_lib_dir}{os.sep}pi",
            f"-DCMAKE_PREFIX_PATH={llvm_install_dir}",
            f"-DCMAKE_MODULE_LINKER_FLAGS=-L{llvm_install_dir}/lib",
            f"-DCMAKE_SHARED_LINKER_FLAGS=-L{llvm_install_dir}/lib",
            f"-DCMAKE_EXE_LINKER_FLAGS=-L{llvm_install_dir}/lib",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        # Using Ninja-build since it a) is available as a wheel and b)
        # multithreads automatically. MSVC would require all variables be
        # exported for Ninja to pick it up, which is a little tricky to do.
        # Users can override the generator with CMAKE_GENERATOR in CMake
        # 3.15+.
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

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
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

VERSION = "0.0.2"

if len(sys.argv) > 1 and sys.argv[1] == "--version":
    print(VERSION)
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
        # install_requires=["PyYAML", "pyccolo", "torch-mlir", "multiprocess", "numpy"]
        install_requires=[str(ir.requirement) for ir in install_reqs],
    )
