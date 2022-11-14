#!/bin/bash

set -e -x

# The absolute path to the directory of this script.
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash "$HERE/check_path.sh"

OS=${OS:-"ubuntu-latest"}
PY_VERSION=${PY_VERSION:-"3.11"}

if [ x"$OS" == x"ubuntu-latest" ]; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
else
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
fi
bash ~/miniconda.sh -b -p llvm_miniconda

CONDA_EXE=llvm_miniconda/bin/conda
$CONDA_EXE create -n mlir -c conda-forge python="$PY_VERSION" -y

python3 -m pip install pybind11==2.10.1 numpy wheel PyYAML dataclasses ninja==1.10.2 cmake==3.24.0 -U --force