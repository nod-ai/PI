#!/bin/bash
set -eu -o pipefail

OS=ubuntu-latest
ARCH=AArch64
GITHUB_WORKSPACE=$PWD/..

TORCH_MLIR_COMMIT=$(git ls-tree HEAD $GITHUB_WORKSPACE/externals/torch-mlir --object-only --abbrev=8)
#TORCH_MLIR_COMMIT=173050ec

wget "https://github.com/nod-ai/PI/releases/download/torch-mlir-${TORCH_MLIR_COMMIT}/torch-mlir-${TORCH_MLIR_COMMIT}-${OS}-X86.tar.xz" -O host-torch-mlir-install.tar.xz
mkdir host-torch-mlir-install && tar -xvf host-torch-mlir-install.tar.xz -C host-torch-mlir-install
export TORCH_MLIR_HOST_MAIN_INSTALL_DIR=${PWD}/host-torch-mlir-install/torch_mlir_install

# for testing purposes override TORCH_MLIR_INSTALL_DIR
#wget "https://github.com/nod-ai/PI/releases/download/torch-mlir-${TORCH_MLIR_COMMIT}/torch-mlir-${TORCH_MLIR_COMMIT}-${OS}-${ARCH}.tar.xz" -O torch-mlir-install.tar.xz
#mkdir torch-mlir-install && tar -xvf torch-mlir-install.tar.xz -C torch-mlir-install
#export TORCH_MLIR_INSTALL_DIR=${PWD}/torch-mlir-install/torch_mlir_install

export TORCH_MLIR_DISTRO_ARCHITECTURE=$ARCH

PY_VERSION=$(python -c "import sys; print('{0[0]}{0[1]}'.format(sys.version_info))")

if [ x"$OS" == x"macos-latest" ]; then
  export CXX_COMPILER=clang++
  export C_COMPILER=clang
  export LLVM_DEFAULT_TARGET_TRIPLE=arm64-apple-darwin21.6.0
  export LLVM_HOST_TRIPLE=arm64-apple-darwin21.6.0
  export ARCH=AArch64
  export PYTHON_MODULE_EXTENSION=".cpython-${PY_VERSION}-darwin.dylib"
  export PLAT_NAME="macosx_12_0_arm64"
elif [ x"$OS" == x"ubuntu-latest" ] && [ x"$ARCH" == x"AArch64" ]; then
  export CXX_COMPILER=aarch64-linux-gnu-g++
  export C_COMPILER=aarch64-linux-gnu-gcc
  export LLVM_DEFAULT_TARGET_TRIPLE=aarch64-linux-gnu
  export LLVM_HOST_TRIPLE=aarch64-linux-gnu
  export ARCH=AArch64
  export PYTHON_MODULE_EXTENSION=".cpython-${PY_VERSION}-aarch64-linux-gnu.so"
  export PLAT_NAME="linux_aarch64"
else
  export CXX_COMPILER=g++
  export C_COMPILER=gcc
  export LLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-linux-gnu
  export LLVM_HOST_TRIPLE=x86_64-unknown-linux-gnu
  export ARCH=X86
  export PYTHON_MODULE_EXTENSION=".cpython-${PY_VERSION}-x86_64-linux-gnu.so"
  export PLAT_NAME="linux_x86_64"
fi

CMAKE_ARGS="\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
  -DCMAKE_C_COMPILER=$C_COMPILER \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_DEFAULT_TARGET_TRIPLE=$LLVM_DEFAULT_TARGET_TRIPLE \
  -DLLVM_HOST_TRIPLE=$LLVM_HOST_TRIPLE \
  -DLLVM_TARGETS_TO_BUILD=$ARCH \
  -DLLVM_TARGET_ARCH=$ARCH \
  -DLLVM_USE_HOST_TOOLS=ON \
  -DPYTHON_MODULE_EXTENSION=$PYTHON_MODULE_EXTENSION"

if [ x"$OS" == x"macos-latest" ]; then
  CMAKE_ARGS="${CMAKE_ARGS} \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DMACOSX_DEPLOYMENT_TARGET=12.0"
elif [ x"$OS" == x"ubuntu-latest" ] && [ x"$ARCH" == x"AArch64" ]; then
  CMAKE_ARGS="${CMAKE_ARGS} \
    -DLLVM_TABLEGEN=$TORCH_MLIR_HOST_MAIN_INSTALL_DIR/bin/llvm-tblgen \
    -DMLIR_LINALG_ODS_YAML_GEN=$TORCH_MLIR_HOST_MAIN_INSTALL_DIR/bin/mlir-linalg-ods-yaml-gen \
    -DMLIR_LINALG_ODS_YAML_GEN_EXE=$TORCH_MLIR_HOST_MAIN_INSTALL_DIR/bin/mlir-linalg-ods-yaml-gen \
    -DMLIR_PDLL_TABLEGEN=$TORCH_MLIR_HOST_MAIN_INSTALL_DIR/bin/mlir-pdll \
    -DMLIR_TABLEGEN=$TORCH_MLIR_HOST_MAIN_INSTALL_DIR/bin/mlir-tblgen"
fi

echo $CMAKE_ARGS

export CMAKE_GENERATOR=Ninja
export CMAKE_ARGS=$CMAKE_ARGS
export WHEELHOUSE_DIR=$PWD/wheelhouse
cd $GITHUB_WORKSPACE && python3 setup.py bdist_wheel --plat-name=$PLAT_NAME --dist-dir $WHEELHOUSE_DIR
