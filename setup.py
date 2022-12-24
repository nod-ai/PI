#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys

from setuptools import find_namespace_packages, setup

packages = find_namespace_packages(
    include=[
        "shark",
        "shark.*",
    ],
)

VERSION = "0.0.1"

if len(sys.argv) > 1 and sys.argv[1] == "--version":
    print(VERSION)
else:
    setup(
        name="SharkPy",
        version=VERSION,
        author="Maksim Levental",
        author_email="maksim.levental@gmail.com",
        description="Python frontend for MLIR (and torch-mlir)",
        zip_safe=False,
        packages=packages,
        install_requires=["PyYAML", "pyccolo", "torch-mlir"],
    )
