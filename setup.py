#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import find_namespace_packages, setup

packages = find_namespace_packages(
    include=[
        "shark",
        "shark.*",
    ],
)

setup(
    name="SharkPy",
    version="0.0.1",
    author="Maksim Levental",
    author_email="maksim.levental@gmail.com",
    description="Python bindings for the Shark MLIR dialect",
    # include_package_data=True,
    # ext_modules=[
    #     CMakeExtension("shark._mlir_libs._mlir"),
    # ],
    # cmdclass={
    #     "build": CustomBuild,
    #     "built_ext": NoopBuildExtension,
    #     "build_py": CMakeBuild,
    # },
    zip_safe=False,
    packages=packages,
    # distclass=BinaryDistribution,
)
